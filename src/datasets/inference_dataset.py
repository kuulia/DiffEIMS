"""
inference_dataset.py

Dataset classes for inference-only scenarios where no molecular structure
(SMILES, InChI, fingerprints) is known — only spectra + molecular formulas.

Designed for the Yee et al. dataset and similar inference use cases.
"""

import os
import re
import logging
from typing import List, Tuple, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from src.mist.data import featurizers, splitter
from src.mist.data.data import Spectra, Mol
from src.mist.data.datasets import SpectraMolDataset, get_paired_loader_graph
from src.mist.data.featurizers import (
    Featurizer,
    PairedFeaturizer,
    PeakFormula,
    NoneFeaturizer,
)
from src.datasets.abstract_dataset import (
    AbstractDatasetInfos,
    MolecularDataModule,
    ATOM_DECODER,
    ATOM_TO_VALENCY,
    ATOM_TO_WEIGHT,
)
from src.diffusion.distributions import DistributionNodes
import src.utils as utils

TYPES = {atom: i for i, atom in enumerate(ATOM_DECODER)}
BONDS_COUNT = 5  # no-bond + single + double + triple + aromatic


class InferenceGraphFeaturizer(Featurizer):
    """Creates placeholder Data objects from formula-only Mol objects.

    Since no SMILES/InChI is available, graphs are constructed with the correct
    atom count (derived from the molecular formula) but empty edges. The edges
    and fingerprints will be replaced during inference anyway:
    - sample_batch() replaces edges with noise
    - data.y is overwritten by the encoder's predicted fingerprint
    """

    def __init__(self, remove_h=True, morgan_nbits=2048, **kwargs):
        super().__init__(**kwargs)
        self.remove_h = remove_h
        self.morgan_nbits = morgan_nbits

    def _encode(self, mol: Mol) -> str:
        return mol.get_molform() or ""

    @staticmethod
    def collate_fn(graphs: List[Data]) -> Batch:
        return Batch.from_data_list(graphs)

    def _featurize(self, mol: Mol) -> Data:
        rdkit_mol = mol.get_rdkit_mol()
        if rdkit_mol is None:
            # Fallback: parse formula directly to get atom types
            return self._featurize_from_formula(mol.get_molform())

        type_idx = []
        for atom in rdkit_mol.GetAtoms():
            symbol = atom.GetSymbol()
            if self.remove_h and symbol == "H":
                continue
            if symbol not in TYPES:
                continue
            type_idx.append(TYPES[symbol])

        if len(type_idx) == 0:
            # Fallback for edge case
            return self._featurize_from_formula(mol.get_molform())

        x = F.one_hot(torch.tensor(type_idx), num_classes=len(TYPES)).float()
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, BONDS_COUNT), dtype=torch.float)
        y = torch.zeros(1, self.morgan_nbits, dtype=torch.int8)
        formula = mol.get_molform() or ""
        inchi = f"InChI=1S/{formula}"

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, inchi=inchi)

    def _featurize_from_formula(self, formula: str) -> Data:
        """Fallback: parse formula string directly to build atom features."""
        pattern = r"([A-Z][a-z]*)(\d*)"
        matches = re.findall(pattern, formula)

        type_idx = []
        for element, count in matches:
            if self.remove_h and element == "H":
                continue
            if element not in TYPES:
                continue
            count = int(count) if count else 1
            type_idx.extend([TYPES[element]] * count)

        if len(type_idx) == 0:
            # Absolute minimum: single carbon atom
            type_idx = [TYPES["C"]]

        x = F.one_hot(torch.tensor(type_idx), num_classes=len(TYPES)).float()
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, BONDS_COUNT), dtype=torch.float)
        y = torch.zeros(1, self.morgan_nbits, dtype=torch.int8)
        inchi = f"InChI=1S/{formula}"

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, inchi=inchi)


def get_inference_spectra(
    labels_file: str,
    spec_folder: str = None,
    collated_pkl: bool = False,
    collated_pkl_file: str = None,
    max_count: Optional[int] = None,
    prog_bars: bool = True,
    **kwargs,
) -> Tuple[List[Spectra], List[Mol]]:
    """Load spectra for inference where no SMILES/InChI is available.

    Args:
        labels_file: Path to labels.tsv with columns: spec, formula, ionization, instrument.
                     smiles/inchikey columns may be present but contain "na".
        spec_folder: Directory containing spectrum files or collated pkl.
        collated_pkl: If True, load spectra from a pickled DataFrame.
        collated_pkl_file: Filename of the collated pkl within spec_folder.
        max_count: Maximum number of spectra to load.

    Returns:
        Tuple of (spectra_list, mol_list) where mol_list contains
        formula-only Mol objects (no SMILES/bonds).
    """
    compound_id_file = pd.read_csv(labels_file, sep="\t").astype(str)
    name_to_formula = dict(compound_id_file[["spec", "formula"]].values)

    name_to_instrument = {}
    if "instrument" in compound_id_file.columns:
        name_to_instrument = dict(compound_id_file[["spec", "instrument"]].values)

    name_to_ionization = {}
    if "ionization" in compound_id_file.columns:
        name_to_ionization = dict(compound_id_file[["spec", "ionization"]].values)

    spec_names = list(name_to_formula.keys())

    tq = tqdm if prog_bars else lambda x: x

    if collated_pkl:
        logging.info("Loading inference spectra from collated pickle file")
        spec_folder_path = Path(spec_folder)
        df_pkl = pd.read_pickle(f"{spec_folder_path}/{collated_pkl_file}")

        # Map UIDs to spectrum arrays — handle both UID column and index-based
        if "UID" in df_pkl.columns:
            spec_to_array = dict(zip(df_pkl["UID"], df_pkl["spec"]))
        else:
            spec_to_array = dict(zip(compound_id_file["spec"], df_pkl["spec"]))

        # Only keep spec names that exist in the pkl
        spec_names = [n for n in spec_names if n in spec_to_array]

        if max_count is not None:
            spec_names = spec_names[:max_count]

        spectra_list = [
            Spectra(
                spectra_name=spec_name,
                spectra_formula=name_to_formula.get(spec_name, ""),
                instrument=name_to_instrument.get(spec_name, ""),
                spectrum_array=spec_to_array[spec_name],
                **kwargs,
            )
            for spec_name in tq(spec_names)
        ]
    else:
        logging.info("Loading inference spectra from .ms files")
        spec_folder_path = Path(spec_folder) if spec_folder is not None else None

        if spec_folder_path is not None and spec_folder_path.exists():
            spectra_files = [Path(i).resolve() for i in spec_folder_path.glob("*.ms")]
        else:
            logging.warning(f"Spec folder {spec_folder} not found, using placeholders")
            spectra_files = [Path(f"{i}.ms") for i in spec_names]

        spectra_files = [f for f in spectra_files if f.stem in name_to_formula]

        if max_count is not None:
            spectra_files = spectra_files[:max_count]

        spec_names = [f.stem for f in spectra_files]

        spectra_list = [
            Spectra(
                spectra_name=spec_name,
                spectra_file=str(spec_file),
                spectra_formula=name_to_formula.get(spec_name, ""),
                instrument=name_to_instrument.get(spec_name, ""),
                **kwargs,
            )
            for spec_name, spec_file in tq(zip(spec_names, spectra_files))
        ]

    # Create formula-only Mol objects
    mol_list = []
    valid_spectra = []
    for spec_name, spec in zip(spec_names, spectra_list):
        formula = name_to_formula.get(spec_name, "")
        mol = Mol.MolFromFormula(formula)
        if mol is not None:
            # Filter by allowed atom types
            all_allowed = True
            for atom in mol.get_rdkit_mol().GetAtoms():
                if atom.GetSymbol() not in ATOM_DECODER:
                    all_allowed = False
                    break
            if all_allowed:
                mol_list.append(mol)
                valid_spectra.append(spec)
            else:
                logging.warning(
                    f"Skipping {spec_name}: contains unsupported atom types"
                )
        else:
            logging.warning(
                f"Skipping {spec_name}: could not parse formula '{formula}'"
            )

    logging.info(
        f"Loaded {len(mol_list)} inference spectra (from {len(spec_names)} total)"
    )
    return (valid_spectra, mol_list)


class InferenceDataModule(MolecularDataModule):
    """DataModule for inference-only spectra where no molecular structure is known.

    All data is assigned to all three splits (train=val=test) since the parent
    class requires all three, and compute_input_output_dims uses train_dataloader.
    """

    def __init__(self, cfg):
        self.remove_h = getattr(cfg.dataset, "remove_h", False) or False
        self.datadir = cfg.dataset.datadir
        self.train_smiles = []
        self.dataset_name = getattr(cfg.dataset, "name", "inference")

        paired_featurizer = PairedFeaturizer(
            spec_featurizer=PeakFormula(**cfg.dataset),
            mol_featurizer=NoneFeaturizer(),
            graph_featurizer=InferenceGraphFeaturizer(
                remove_h=self.remove_h,
                morgan_nbits=cfg.dataset.morgan_nbits,
            ),
        )

        spectra_mol_pairs = get_inference_spectra(**cfg.dataset)
        spectra_mol_pairs = list(zip(*spectra_mol_pairs))

        dataset = SpectraMolDataset(
            spectra_mol_list=spectra_mol_pairs,
            featurizer=paired_featurizer,
            **cfg.dataset,
        )

        # All data goes to all splits (parent class requires train/val/test)
        datasets = {"train": dataset, "val": dataset, "test": dataset}
        super().__init__(cfg, datasets)

    def train_dataloader(self) -> DataLoader:
        return get_paired_loader_graph(
            self.train_dataset, shuffle=False, batch_size=self.batch_size, **self.kwargs
        )

    def val_dataloader(self) -> DataLoader:
        return get_paired_loader_graph(
            self.val_dataset,
            shuffle=False,
            batch_size=self.eval_batch_size,
            **self.kwargs,
        )

    def test_dataloader(self) -> DataLoader:
        return get_paired_loader_graph(
            self.test_dataset,
            shuffle=False,
            batch_size=self.eval_batch_size,
            **self.kwargs,
        )

    def node_counts(self, max_nodes_possible=150):
        all_counts = torch.zeros(max_nodes_possible)
        for batch in self.train_dataloader():
            data = batch["graph"]
            unique, counts = torch.unique(data.batch, return_counts=True)
            for count in counts:
                all_counts[count] += 1
        max_index = max(all_counts.nonzero())
        all_counts = all_counts[: max_index + 1]
        all_counts = all_counts / all_counts.sum()
        return all_counts

    def node_types(self):
        num_classes = None
        for batch in self.train_dataloader():
            data = batch["graph"]
            num_classes = data.x.shape[1]
            break
        counts = torch.zeros(num_classes)
        for batch in self.train_dataloader():
            data = batch["graph"]
            counts += data.x.sum(dim=0)
        counts = counts / counts.sum()
        return counts

    def edge_counts(self):
        num_classes = None
        for batch in self.train_dataloader():
            data = batch["graph"]
            num_classes = data.edge_attr.shape[1]
            break
        d = torch.zeros(num_classes, dtype=torch.float)
        for batch in self.train_dataloader():
            data = batch["graph"]
            unique, counts = torch.unique(data.batch, return_counts=True)
            all_pairs = 0
            for count in counts:
                all_pairs += count * (count - 1)
            num_edges = data.edge_index.shape[1]
            num_non_edges = all_pairs - num_edges
            edge_types = data.edge_attr.sum(dim=0)
            assert num_non_edges >= 0
            d[0] += num_non_edges
            d[1:] += edge_types[1:]
        d = d / d.sum()
        return d

    def valency_count(self, max_n_nodes):
        valencies = torch.zeros(3 * max_n_nodes - 2)
        multiplier = torch.tensor([0, 1, 2, 3, 1.5])
        for batch in self.train_dataloader():
            data = batch["graph"]
            n = data.x.shape[0]
            for atom in range(n):
                edges = data.edge_attr[data.edge_index[0] == atom]
                edges_total = edges.sum(dim=0)
                valency = (edges_total * multiplier).sum()
                valencies[valency.long().item()] += 1
        valencies = valencies / valencies.sum()
        return valencies


valency = [ATOM_TO_VALENCY.get(atom, 0) for atom in ATOM_DECODER]


class InferenceDatasetInfos(AbstractDatasetInfos):
    """Dataset infos for inference, loading statistics from a reference training dataset.

    Requires `stats_dir` in cfg.dataset pointing to a directory with:
    n_counts.txt, atom_types.txt, edge_types.txt, valencies.txt
    """

    def __init__(self, datamodule, cfg, recompute_statistics=False, meta=None):
        self.name = getattr(cfg.dataset, "name", "inference")
        self.input_dims = None
        self.output_dims = None
        self.remove_h = getattr(cfg.dataset, "remove_h", False) or False

        self.atom_decoder = ATOM_DECODER
        self.atom_encoder = {atom: i for i, atom in enumerate(self.atom_decoder)}
        self.atom_weights = {
            i: ATOM_TO_WEIGHT.get(atom, 0) for i, atom in enumerate(self.atom_decoder)
        }
        self.valencies = valency
        self.num_atom_types = len(self.atom_decoder)
        self.max_weight = max(self.atom_weights.values())

        stats_dir = getattr(cfg.dataset, "stats_dir", None)
        if stats_dir is None:
            raise ValueError(
                "InferenceDatasetInfos requires 'stats_dir' in dataset config, "
                "pointing to a directory with pre-computed training statistics "
                "(n_counts.txt, atom_types.txt, edge_types.txt, valencies.txt)"
            )

        meta_files = dict(
            n_nodes=f"{stats_dir}/n_counts.txt",
            node_types=f"{stats_dir}/atom_types.txt",
            edge_types=f"{stats_dir}/edge_types.txt",
            valency_distribution=f"{stats_dir}/valencies.txt",
        )

        self.n_nodes = None
        self.node_types = None
        self.edge_types = None
        self.valency_distribution = None

        for k, v in meta_files.items():
            if os.path.exists(v):
                setattr(self, k, torch.tensor(np.loadtxt(v)))
            else:
                raise FileNotFoundError(
                    f"Stats file not found: {v}. "
                    f"Ensure stats_dir points to a valid training stats directory."
                )

        self.max_n_nodes = len(self.n_nodes) - 1

        if recompute_statistics:
            logging.warning(
                "recompute_statistics=True ignored for InferenceDatasetInfos; using pre-computed stats."
            )

        self.complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)

    def compute_input_output_dims(self, datamodule, extra_features, domain_features):
        example_batch = next(iter(datamodule.train_dataloader()))["graph"]
        ex_dense, node_mask = utils.to_dense(
            example_batch.x,
            example_batch.edge_index,
            example_batch.edge_attr,
            example_batch.batch,
        )
        example_data = {
            "X_t": ex_dense.X,
            "E_t": ex_dense.E,
            "y_t": example_batch["y"],
            "node_mask": node_mask,
        }

        self.input_dims = {
            "X": example_batch["x"].size(1),
            "E": (
                example_batch["edge_attr"].size(1)
                if example_batch["edge_attr"].numel() > 0
                else BONDS_COUNT
            ),
            "y": example_batch["y"].size(1) + 1,  # +1 for time conditioning
        }

        ex_extra_feat = extra_features(example_data)
        self.input_dims["X"] += ex_extra_feat.X.size(-1)
        self.input_dims["E"] += ex_extra_feat.E.size(-1)
        self.input_dims["y"] += ex_extra_feat.y.size(-1)

        ex_extra_molecular_feat = domain_features(example_data)
        self.input_dims["X"] += ex_extra_molecular_feat.X.size(-1)
        self.input_dims["E"] += ex_extra_molecular_feat.E.size(-1)
        self.input_dims["y"] += ex_extra_molecular_feat.y.size(-1)

        self.output_dims = {
            "X": example_batch["x"].size(1),
            "E": (
                example_batch["edge_attr"].size(1)
                if example_batch["edge_attr"].numel() > 0
                else BONDS_COUNT
            ),
            "y": example_batch["y"].size(1),
        }
