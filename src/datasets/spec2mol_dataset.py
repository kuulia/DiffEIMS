"""
spec2mol_dataset.py — DDP-compatible data module for spec2mol training.

Design principles:
  - __init__: stores config only, zero I/O
  - prepare_data(): called by Lightning on global rank 0 only; ensures stat
    files exist (computes them if missing) and touches a sentinel file so other
    ranks can proceed without a distributed barrier
  - setup(stage): called by Lightning on every rank; loads pickle/spectra with
    LOCAL_RANK-based I/O staggering so at most one process per node hits the
    filesystem at a time
  - Dataloaders: return plain DataLoaders with sampler=None so Lightning's DDP
    strategy can inject DistributedSampler automatically
"""

import os
import math
import time
import pathlib
import logging
import random
from functools import partial
from typing import Any, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.mist.data import datasets, splitter, featurizers
from src.mist.data.datasets import get_paired_loader_graph, SpectraMolDataset
from src.datasets.abstract_dataset import (
    AbstractDatasetInfos,
    ATOM_TO_VALENCY,
    ATOM_TO_WEIGHT,
    ATOM_DECODER,
)
from src.diffusion.distributions import DistributionNodes
import src.utils as utils

# ---------------------------------------------------------------------------
# Helpers shared by DataModule and DatasetInfos
# ---------------------------------------------------------------------------

valency = [ATOM_TO_VALENCY.get(atom, 0) for atom in ATOM_DECODER]

SENTINEL_FILENAME = ".stat_files_ready"
STAT_FILENAMES = {
    "n_nodes": "n_counts.txt",
    "node_types": "atom_types.txt",
    "edge_types": "edge_types.txt",
    "valency_distribution": "valencies.txt",
}

# Seconds to wait between retries when polling for the sentinel file
_SENTINEL_POLL_INTERVAL = 5.0
# How long (s) non-zero ranks wait for rank 0 before raising
_SENTINEL_TIMEOUT = 7200


def _stat_root(cfg) -> str:
    parent_dir = getattr(cfg.general, "parent_dir", ".")
    return os.path.join(parent_dir, cfg.dataset.datadir)


def _sentinel_path(cfg) -> str:
    return os.path.join(_stat_root(cfg), SENTINEL_FILENAME)


def _stat_file_path(cfg, key: str) -> str:
    """Return path for a named stat file, respecting optional stats_dir override."""
    root = _stat_root(cfg)
    stats_dir = getattr(cfg.dataset, "stats_dir", None)
    if stats_dir and key in ("node_types", "edge_types"):
        base = stats_dir
    else:
        base = root
    return os.path.join(base, STAT_FILENAMES[key])


def _all_stat_files_exist(cfg) -> bool:
    return all(os.path.exists(_stat_file_path(cfg, k)) for k in STAT_FILENAMES)


def _wait_for_sentinel(cfg):
    """Block until the sentinel file appears (called on non-zero ranks)."""
    path = _sentinel_path(cfg)
    start = time.time()
    while not os.path.exists(path):
        elapsed = time.time() - start
        if elapsed > _SENTINEL_TIMEOUT:
            raise TimeoutError(
                f"Rank {os.environ.get('RANK', '?')} waited {_SENTINEL_TIMEOUT}s "
                f"for stat-file sentinel '{path}'. "
                f"Ensure rank 0 completed successfully."
            )
        time.sleep(_SENTINEL_POLL_INTERVAL)


# ---------------------------------------------------------------------------
# Static dimension computation (no data loading required)
# ---------------------------------------------------------------------------


def _compute_static_dims(cfg):
    """
    Compute model input/output feature dimensions purely from the config,
    without touching any data files.

    Breakdown:
      Base: 12 atom types, 5 edge types (no-bond + 4 bond orders),
            morgan_nbits fingerprint for y
      ExtraFeatures (from cfg.model.extra_features):
        None          → (X=0,  E=0, y=0)
        'cycles'      → (X=3,  E=0, y=5)   k3/k4/k5 per node; n+k3y+k4y+k5y+k6y
        'eigenvalues' → (X=3,  E=0, y=11)  same X; n+kcycles(4)+n_comp+eigvals(5)
        'all'         → (X=6,  E=0, y=11)  cycles+nonlcc(1)+eigvec(2); same y
      ExtraMolecularFeatures: (X=2, E=0, y=1)  charge+valency per node; weight
      input y gets +1 for time conditioning
    """
    num_atom_types = len(ATOM_DECODER)  # 12
    num_edge_types = 5  # one-hot: [no-bond, single, double, triple, aromatic]
    morgan_nbits = cfg.dataset.morgan_nbits

    feat_type = cfg.model.extra_features
    if feat_type is None:
        X_extra, E_extra, y_extra = 0, 0, 0
    elif feat_type == "cycles":
        X_extra, E_extra, y_extra = 3, 0, 5
    elif feat_type == "eigenvalues":
        X_extra, E_extra, y_extra = 3, 0, 11
    elif feat_type == "all":
        X_extra, E_extra, y_extra = 6, 0, 11
    else:
        raise ValueError(f"Unknown extra_features type: '{feat_type}'")

    # ExtraMolecularFeatures: charge(1) + valency(1) per node, weight(1) global
    X_mol, E_mol, y_mol = 2, 0, 1

    input_dims = {
        "X": num_atom_types + X_extra + X_mol,
        "E": num_edge_types + E_extra + E_mol,
        "y": morgan_nbits + 1 + y_extra + y_mol,  # +1 = time conditioning
    }
    output_dims = {
        "X": num_atom_types,
        "E": num_edge_types,
        "y": morgan_nbits,
    }
    return input_dims, output_dims


# ---------------------------------------------------------------------------
# Stat-file computation (rank-0 only, called from prepare_data / main sentinel)
# ---------------------------------------------------------------------------


def _build_splits(cfg):
    """Load raw spectra/mol pairs and return (train, val, test) dataset objects."""
    data_splitter = splitter.PresetSpectraSplitter(split_file=cfg.dataset.split_file)

    paired_featurizer = featurizers.PairedFeaturizer(
        spec_featurizer=featurizers.PeakFormula(**cfg.dataset),
        mol_featurizer=featurizers.FingerprintFeaturizer(
            fp_names=["morgan4096"], **cfg.dataset
        ),
        graph_featurizer=featurizers.GraphFeaturizer(**cfg.dataset),
    )

    spectra_mol_pairs = datasets.get_paired_spectra(
        atom_decoder=ATOM_DECODER, **cfg.dataset
    )
    spectra_mol_pairs = list(zip(*spectra_mol_pairs))

    _, (train_pairs, val_pairs, test_pairs) = data_splitter.get_splits(
        spectra_mol_pairs
    )

    random.seed(42)
    random.shuffle(test_pairs)

    def _make_dataset(pairs):
        return SpectraMolDataset(
            spectra_mol_list=pairs,
            featurizer=paired_featurizer,
            **cfg.dataset,
        )

    if len(train_pairs) == 0 and len(val_pairs) == 0:
        logging.info(f"Only test data found: {len(test_pairs)} samples")
        return None, None, _make_dataset(test_pairs)

    return (
        _make_dataset(train_pairs),
        _make_dataset(val_pairs),
        _make_dataset(test_pairs),
    )


def _collate_fn_for_stats(cfg):
    """Build a minimal collate_fn for iterating graphs during stat computation."""
    from src.mist.data.datasets import _collate_pairs_graph

    dummy_featurizer = featurizers.PairedFeaturizer(
        spec_featurizer=featurizers.PeakFormula(**cfg.dataset),
        mol_featurizer=featurizers.FingerprintFeaturizer(
            fp_names=["morgan4096"], **cfg.dataset
        ),
        graph_featurizer=featurizers.GraphFeaturizer(**cfg.dataset),
    )
    return partial(
        _collate_pairs_graph,
        mol_collate_fn=dummy_featurizer.get_mol_collate(),
        spec_collate_fn=dummy_featurizer.get_spec_collate(),
        graph_collate_fn=dummy_featurizer.get_graph_collate(),
    )


def compute_dataset_stats(cfg):
    """
    Load the full dataset and compute all stat files that are missing.
    Writes results to the dataset root directory.
    Only safe to call from a single process (rank 0).
    """
    root = _stat_root(cfg)
    os.makedirs(root, exist_ok=True)

    stats_dir = getattr(cfg.dataset, "stats_dir", None)
    if stats_dir:
        os.makedirs(stats_dir, exist_ok=True)

    if _all_stat_files_exist(cfg):
        logging.info("All stat files already exist; skipping computation.")
        return

    logging.info("Computing dataset statistics (rank 0 only, one-time setup)...")
    train_ds, val_ds, test_ds = _build_splits(cfg)

    # Use the same collate function as get_paired_loader_graph
    from src.mist.data.datasets import _collate_pairs_graph

    paired_featurizer = featurizers.PairedFeaturizer(
        spec_featurizer=featurizers.PeakFormula(**cfg.dataset),
        mol_featurizer=featurizers.FingerprintFeaturizer(
            fp_names=["morgan4096"], **cfg.dataset
        ),
        graph_featurizer=featurizers.GraphFeaturizer(**cfg.dataset),
    )
    collate_fn = partial(
        _collate_pairs_graph,
        mol_collate_fn=paired_featurizer.get_mol_collate(),
        spec_collate_fn=paired_featurizer.get_spec_collate(),
        graph_collate_fn=paired_featurizer.get_graph_collate(),
    )

    batch_size = min(128, getattr(cfg.train, "batch_size", 128))
    num_workers = min(4, getattr(cfg.train, "num_workers", 4))

    def _loader(ds):
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )

    loaders_for_train_stats = []
    if train_ds is not None:
        loaders_for_train_stats.append(_loader(train_ds))
    if val_ds is not None:
        loaders_for_train_stats.append(_loader(val_ds))
    if not loaders_for_train_stats:
        loaders_for_train_stats = [_loader(test_ds)]

    # --- n_nodes ---
    n_nodes_path = _stat_file_path(cfg, "n_nodes")
    if not os.path.exists(n_nodes_path):
        max_nodes = 150
        all_counts = torch.zeros(max_nodes)
        for loader in loaders_for_train_stats:
            for batch in loader:
                data = batch["graph"]
                _, counts = torch.unique(data.batch, return_counts=True)
                for c in counts:
                    if c < max_nodes:
                        all_counts[c] += 1
        max_idx = int(all_counts.nonzero().max())
        all_counts = all_counts[: max_idx + 1] / all_counts[: max_idx + 1].sum()
        np.savetxt(n_nodes_path, all_counts.numpy())
        logging.info(f"Saved n_nodes → {n_nodes_path}")

    # --- node_types ---
    node_types_path = _stat_file_path(cfg, "node_types")
    if not os.path.exists(node_types_path):
        num_classes = None
        counts = None
        for loader in loaders_for_train_stats[:1]:
            for batch in loader:
                data = batch["graph"]
                num_classes = data.x.shape[1]
                break
        if num_classes is not None:
            counts = torch.zeros(num_classes)
            for loader in loaders_for_train_stats[:1]:
                for batch in loader:
                    counts += batch["graph"].x.sum(dim=0)
            counts = counts / counts.sum()
            np.savetxt(node_types_path, counts.numpy())
            logging.info(f"Saved node_types → {node_types_path}")

    # --- edge_types ---
    edge_types_path = _stat_file_path(cfg, "edge_types")
    if not os.path.exists(edge_types_path):
        num_classes = None
        d = None
        for loader in loaders_for_train_stats[:1]:
            for batch in loader:
                data = batch["graph"]
                num_classes = data.edge_attr.shape[1]
                break
        if num_classes is not None:
            d = torch.zeros(num_classes, dtype=torch.float)
            for loader in loaders_for_train_stats[:1]:
                for batch in loader:
                    data = batch["graph"]
                    _, counts = torch.unique(data.batch, return_counts=True)
                    all_pairs = sum(c * (c - 1) for c in counts)
                    num_edges = data.edge_index.shape[1]
                    num_non_edges = all_pairs - num_edges
                    edge_types = data.edge_attr.sum(dim=0)
                    assert num_non_edges >= 0
                    d[0] += num_non_edges
                    d[1:] += edge_types[1:]
            d = d / d.sum()
            np.savetxt(edge_types_path, d.numpy())
            logging.info(f"Saved edge_types → {edge_types_path}")

    # --- valency_distribution ---
    val_dist_path = _stat_file_path(cfg, "valency_distribution")
    if not os.path.exists(val_dist_path):
        # Need max_n_nodes for valency histogram size
        n_nodes_data = torch.tensor(np.loadtxt(n_nodes_path))
        max_n_nodes = len(n_nodes_data) - 1
        valencies = torch.zeros(3 * max_n_nodes - 2)
        multiplier = torch.tensor([0, 1, 2, 3, 1.5])
        for loader in loaders_for_train_stats[:1]:
            for batch in loader:
                data = batch["graph"]
                n = data.x.shape[0]
                for atom in range(n):
                    edges = data.edge_attr[data.edge_index[0] == atom]
                    edges_total = edges.sum(dim=0)
                    val = (edges_total * multiplier).sum()
                    idx = val.long().item()
                    if 0 <= idx < len(valencies):
                        valencies[idx] += 1
        valencies = valencies / valencies.sum()
        np.savetxt(val_dist_path, valencies.numpy())
        logging.info(f"Saved valency_distribution → {val_dist_path}")

    logging.info("Dataset statistics computation complete.")


# ---------------------------------------------------------------------------
# Main DataModule
# ---------------------------------------------------------------------------


class Spec2MolDataModule(pl.LightningDataModule):
    """
    DDP-compatible LightningDataModule for spec2mol.

    Lifecycle:
      __init__  → store config only, zero I/O
      prepare_data() → rank-0 only: compute stat files if missing, touch sentinel
      setup()   → all ranks: load dataset pickles with LOCAL_RANK-based staggering
    """

    # Seconds per LOCAL_RANK index to stagger dataset pickle reads within a node.
    # Rank 0 reads immediately; rank 7 waits 7 × STAGGER_SECONDS before opening files.
    STAGGER_SECONDS: float = 3.0

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.remove_h = getattr(cfg.dataset, "remove_h", False) or False
        self.datadir = cfg.dataset.datadir
        self.filter_dataset = cfg.dataset.filter

        # DataLoader hyperparameters
        self.batch_size = cfg.train.batch_size
        self.eval_batch_size = cfg.train.eval_batch_size
        self.num_workers = cfg.train.num_workers
        self.pin_memory = getattr(cfg.train, "pin_memory", True)
        self.random_seed = getattr(cfg.train, "seed", None)
        self.fraction_train_epoch = getattr(cfg.train, "fraction_train_epoch", 1.0)

        # Populated by setup()
        self.train_dataset: Optional[SpectraMolDataset] = None
        self.val_dataset: Optional[SpectraMolDataset] = None
        self.test_dataset: Optional[SpectraMolDataset] = None

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def prepare_data(self):
        """Called by Lightning on global rank 0 only (guaranteed).

        Ensures all stat files exist and writes the sentinel file so that
        non-zero ranks in the manual sentinel-wait in main() can proceed.
        Idempotent: no-op if sentinel already exists.
        """
        sentinel = _sentinel_path(self.cfg)
        if os.path.exists(sentinel):
            return
        compute_dataset_stats(self.cfg)
        pathlib.Path(sentinel).touch()
        logging.info(f"Sentinel written: {sentinel}")

    def setup(self, stage: Optional[str] = None):
        """Called by Lightning on every rank after prepare_data().

        Loads dataset splits with I/O staggering: LOCAL_RANK k sleeps
        k × STAGGER_SECONDS before opening pickle files, so only one
        process per node reads at a time.
        """
        # Guard: don't reload splits that are already loaded for this stage
        needs_train = stage in ("fit", "validate", None) and self.train_dataset is None
        needs_test = stage in ("test", "predict", None) and self.test_dataset is None
        if not needs_train and not needs_test:
            return

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if local_rank > 0:
            delay = local_rank * self.STAGGER_SECONDS
            logging.info(
                f"Rank {os.environ.get('RANK', '?')}: "
                f"staggering dataset load by {delay:.1f}s (LOCAL_RANK={local_rank})"
            )
            time.sleep(delay)

        logging.info(f"Rank {os.environ.get('RANK', '?')}: loading dataset splits...")
        train_ds, val_ds, test_ds = _build_splits(self.cfg)

        if needs_train:
            self.train_dataset = train_ds
            self.val_dataset = val_ds
        if needs_test:
            self.test_dataset = test_ds
            if getattr(self.cfg.train, "sort_test_by_size", False):
                # Round the prefix up to a whole number of per-rank batches, using the
                # same batches_to_sample arithmetic as test_step. If it only covered
                # num_test_samples exactly, the last sampled batch would straddle the
                # boundary and pull in unselected molecules.
                world_size = int(os.environ.get("WORLD_SIZE", 1))
                per_round = self.eval_batch_size * world_size
                n_batches = math.ceil(
                    self.cfg.general.num_test_samples / per_round
                )
                self.test_dataset.arrange_for_eval(
                    n_sample=n_batches * per_round,
                    seed=getattr(self.cfg.train, "seed", 42),
                )

    # ------------------------------------------------------------------
    # DataLoaders
    # NOTE: Lightning's DDPStrategy intercepts these and wraps the
    # DataLoader's sampler with DistributedSampler automatically because
    # we always pass sampler=None here.
    # ------------------------------------------------------------------

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            return None
        return get_paired_loader_graph(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            return None
        return get_paired_loader_graph(
            self.val_dataset,
            shuffle=False,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            return None
        return get_paired_loader_graph(
            self.test_dataset,
            shuffle=False,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    # Keep these for compatibility with any code that iterates the module directly
    def __getitem__(self, idx):
        return self.train_dataset[idx]


# ---------------------------------------------------------------------------
# DatasetInfos — loaded from stat files, no data iteration
# ---------------------------------------------------------------------------


class Spec2MolDatasetInfos(AbstractDatasetInfos):
    """
    Holds dataset metadata for model construction.

    All information is loaded from pre-computed stat files on disk.
    Input/output dimensions are derived statically from the config.
    No data loading or DataLoader iteration is performed here.
    """

    def __init__(self, cfg, recompute_statistics: bool = False):
        self.name = getattr(cfg.dataset, "name", "canopus")
        self.remove_h = getattr(cfg.dataset, "remove_h", False) or False

        self.atom_decoder = ATOM_DECODER
        self.atom_encoder = {atom: i for i, atom in enumerate(self.atom_decoder)}
        self.atom_weights = {
            i: ATOM_TO_WEIGHT.get(atom, 0) for i, atom in enumerate(self.atom_decoder)
        }
        self.valencies = valency
        self.num_atom_types = len(self.atom_decoder)
        self.max_weight = max(self.atom_weights.values())

        # --- Load stat files ---
        if recompute_statistics:
            logging.warning(
                "recompute_statistics=True is not supported in DDP mode. "
                "Run compute_dataset_stats() on a single process first."
            )

        for key in STAT_FILENAMES:
            path = _stat_file_path(cfg, key)
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Stat file '{path}' not found.\n"
                    f"Either run a single-GPU pass first so rank 0 can compute it,\n"
                    f"or call compute_dataset_stats(cfg) explicitly before launching DDP."
                )
            setattr(self, key, torch.tensor(np.loadtxt(path)))

        self.max_n_nodes = len(self.n_nodes) - 1

        # --- Static dimension computation ---
        self.input_dims, self.output_dims = _compute_static_dims(cfg)

        # complete_infos sets num_classes, nodes_dist, max_n_nodes, and
        # also resets input_dims/output_dims to None — we restore them after.
        self.complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)
        self.input_dims, self.output_dims = _compute_static_dims(cfg)
