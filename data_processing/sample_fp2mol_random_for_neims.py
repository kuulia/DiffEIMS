"""
Diversity-ablation counterpart of sample_fp2mol_for_neims.py: a uniform random
sample from the combined fp2mol pool, sized to mixed_augment's train+val, with
none of the atmospheric filter / MW stratification / ATMOMACCS clustering.

The only filters are those of build_fp2mol_datasets.py (single fragment,
16 <= MW < 1500, atoms in FILTER_ATOMS, stereo stripped), re-applied to every
written SMILES. Every atmomaccs / atmomaccs_tms molecule (all splits) and the
test/val molecules of every spectral dataset are excluded, matching on the
InChIKey connectivity block as build_guarded_splits.py does; the sample is also
unique at that level, like mixed_augment.

Output: one-column CSV ('smiles') for NEIMS spectrum simulation.

Usage (from data_processing/):
    python sample_fp2mol_random_for_neims.py
"""

import argparse
import random
from pathlib import Path
from typing import Set

import pandas as pd
from rdkit import Chem
from tqdm import tqdm

from build_fp2mol_datasets import (
    LEAKAGE_GUARD_DATASETS,
    _build_registry,
    _canonicalize,
    _passes_filter,
    _to_inchi,
)

RANDOM_SEED = 42


def _target_size(split_path: Path) -> int:
    split = pd.read_csv(split_path, sep="\t")
    return int(split["split"].isin(["train", "val"]).sum())


def _exclusions(data: Path) -> Set[str]:
    spectral, _, _ = _build_registry(data, data / "fp2mol")
    excluded: Set[str] = set()
    for ds in spectral:
        # Guard datasets: every split. Others: test/val, as in a full
        # build_fp2mol_datasets.py run.
        splits = None if ds.name in LEAKAGE_GUARD_DATASETS else ("test", "val")
        try:
            collected = ds.get_exclusions(splits=splits)
        except Exception as e:
            if ds.name in LEAKAGE_GUARD_DATASETS:
                raise RuntimeError(f"Leakage guard dataset '{ds.name}' unreadable: {e}")
            print(f"[WARN] Could not collect exclusions from '{ds.name}': {e}")
            continue
        print(f"[EXCL] {ds.name}: {len(collected)} molecules")
        excluded |= collected
    print(f"[EXCL] {len(excluded)} unique molecules in total")
    return excluded


def _skeleton(inchi: str) -> str:
    """InChIKey connectivity block, the key build_guarded_splits.py matches on."""
    return Chem.InchiToInchiKey(inchi)[:14]


def _inchi_to_smiles(inchi: str):
    """InChI -> (stereo-free canonical SMILES, its InChI) if it passes the filter."""
    try:
        mol = Chem.MolFromInchi(inchi)
        if mol is None:
            return None
        mol, smi = _canonicalize(Chem.MolToSmiles(mol, isomericSmiles=False))
    except Exception:
        return None
    if not _passes_filter(mol, filter_atoms=True):
        return None
    return smi, _to_inchi(mol)


def main(args: argparse.Namespace) -> None:
    data = Path(args.data_dir)
    n_target = args.n or _target_size(data / args.size_from)
    print(f"[INFO] Target: {n_target} molecules")

    excluded = _exclusions(data)
    # Tautomer/isotope layers can differ between otherwise identical molecules,
    # so exclude and deduplicate on the connectivity block as well.
    excluded_skel = {_skeleton(i) for i in excluded}

    pool = pd.read_csv(data / args.input)["inchi"].dropna().unique().tolist()
    print(f"[INFO] Pool: {len(pool)} InChIs from {args.input}")
    random.Random(RANDOM_SEED).shuffle(pool)

    smiles, seen = [], set()
    n_excluded = n_failed = 0
    with tqdm(total=n_target, desc="Sampling") as pbar:
        for inchi in pool:
            if len(smiles) == n_target:
                break
            if inchi in excluded:
                n_excluded += 1
                continue
            res = _inchi_to_smiles(inchi)
            if res is None or res[1] is None:
                n_failed += 1
                continue
            smi, inchi_rt = res
            # Check the round-tripped InChI too: that is what any later
            # leakage check on the simulated dataset will compute.
            skel = _skeleton(inchi_rt)
            if inchi_rt in excluded or skel in excluded_skel:
                n_excluded += 1
                continue
            if skel in seen:
                continue
            seen.add(skel)
            smiles.append(smi)
            pbar.update(1)

    print(f"[INFO] Skipped {n_excluded} excluded, {n_failed} failed conversion/filter")
    if len(smiles) < n_target:
        raise RuntimeError(f"Pool exhausted: only {len(smiles)} of {n_target} sampled")

    out = data / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"smiles": smiles}).to_csv(out, index=False)
    print(f"[DONE] Wrote {len(smiles)} SMILES to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Uniform random fp2mol sample for the NEIMS diversity ablation."
    )
    parser.add_argument("--data-dir", default="../data", metavar="PATH")
    parser.add_argument(
        "--input",
        default="fp2mol/combined/preprocessed/combined_train.csv",
        help="Pool to sample from, relative to --data-dir (column: inchi).",
    )
    parser.add_argument(
        "--size-from",
        default="neims/mixed_augment/mist_inputs/mixed_augment/split_random.tsv",
        help="Split file whose train+val row count sets the sample size.",
    )
    parser.add_argument("--n", type=int, default=None, help="Override sample size.")
    parser.add_argument(
        "--output",
        default="fp2mol/neims_random_ablation/random_sampled.csv",
        help="Output CSV, relative to --data-dir.",
    )
    main(parser.parse_args())
