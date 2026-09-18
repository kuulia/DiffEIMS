"""
Write leakage-guarded copies of the spectral datasets' split files.

spec2mol (encoder / end-to-end training) does not read the preprocessed InChI CSVs
that build_fp2mol_datasets.py guards -- it needs spectrum-molecule pairs, so it reads
labels.tsv + the split file + the collated pkls directly. The atmomaccs leakage
guard therefore has to be applied to the split file itself.

For every spectral dataset except the guard datasets, this writes
    <mist_inputs dir>/split_random_guarded.tsv
which is split_random.tsv with every train/val row removed whose molecule is also
an atmomaccs or atmomaccs_tms molecule (any split). Test rows are never touched --
for the *_atmomaccs_test datasets they ARE the atmomaccs pool. Removed rows are
listed in split_random_guarded.removed.tsv next to it for auditing.

Molecules are compared by InChIKey connectivity block (first 14 chars). That is
slightly broader than the evaluation's own criterion (full InChI of a stereo-free
graph): it also treats stereo/isotope/charge variants of one skeleton as the same
molecule, which is the safe direction for leakage.

Run from data_processing/ (same convention as build_fp2mol_datasets.py):
    python build_guarded_splits.py                 # all datasets
    python build_guarded_splits.py -d gecko_new    # one dataset
    python build_guarded_splits.py --check         # verify existing files only
"""

import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Iterable, Optional, Set

import pandas as pd
from rdkit import Chem, RDLogger

from build_fp2mol_datasets import (
    GUARDED_SPLIT_FILE,
    LEAKAGE_GUARD_DATASETS,
    SPECTRAL_LAYOUT,
)

RDLogger.DisableLog("rdApp.*")

SOURCE_SPLIT_FILE = "split_random.tsv"
REMOVED_SUFFIX = ".removed.tsv"


def connectivity_key(smi: str) -> Optional[str]:
    try:
        mol = Chem.MolFromSmiles(smi)
        return Chem.MolToInchiKey(mol).split("-")[0] if mol is not None else None
    except Exception:
        return None


def compute_keys(smiles: Iterable[str], workers: int) -> Dict[str, Optional[str]]:
    uniq = sorted({s for s in smiles if isinstance(s, str)})
    with Pool(workers) as pool:
        keys = pool.map(connectivity_key, uniq, chunksize=2000)
    return dict(zip(uniq, keys))


def read_split(path: Path) -> pd.DataFrame:
    # The upstream pipeline writes two formats: some split files have a leading
    # unnamed index column ("\tname\tsplit"), some don't ("name\tsplit"). Read it
    # as a plain column either way so write_split can reproduce the same format.
    return pd.read_csv(path, sep="\t")


def write_split(df: pd.DataFrame, path: Path):
    # pandas reads an empty header field as "Unnamed: 0"; writing it back as ""
    # restores the leading-tab header of the original.
    df.rename(columns={"Unnamed: 0": ""}).to_csv(path, sep="\t", index=False)


def read_labels(d: Path) -> pd.DataFrame:
    return pd.read_csv(d / "labels.tsv", sep="\t", usecols=["spec", "smiles"])


def guarded_frame(split: pd.DataFrame, spec_key: pd.Series, guard: Set[str]) -> pd.Series:
    """Boolean mask of rows to DROP."""
    keys = split["name"].map(spec_key)
    return split["split"].isin(["train", "val"]) & keys.isin(guard)


def verify(d: Path, spec_key: pd.Series, guard: Set[str]) -> Optional[str]:
    """Return an error message, or None if the guarded file is sound."""
    src, out = d / SOURCE_SPLIT_FILE, d / GUARDED_SPLIT_FILE
    if not out.exists():
        return f"missing {out}"
    orig, g = read_split(src), read_split(out)
    trainval = g[g["split"].isin(["train", "val"])]
    leaked = trainval["name"].map(spec_key).isin(guard).sum()
    if leaked:
        return f"{leaked} guarded molecules still in train/val"
    # Compare contents, not the pandas row index, which shifts after any removal.
    test_of = lambda df: df[df["split"] == "test"].reset_index(drop=True)
    if not test_of(orig).equals(test_of(g)):
        return "test rows differ from split_random.tsv"
    if not set(g["name"]) <= set(orig["name"]):
        return "contains specs not in split_random.tsv"
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-dir", default="../data", metavar="PATH")
    parser.add_argument("--dataset", "-d", metavar="NAME", help="Only this dataset.")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument(
        "--check", action="store_true", help="Verify existing guarded files, write nothing."
    )
    args = parser.parse_args()

    neims = Path(args.data_dir) / "neims"
    layout = dict(SPECTRAL_LAYOUT)
    targets = [n for n in layout if n not in LEAKAGE_GUARD_DATASETS]
    if args.dataset:
        if args.dataset not in targets:
            parser.error(
                f"'{args.dataset}' is not a guardable dataset. Choose from: {', '.join(targets)}"
            )
        targets = [args.dataset]

    labels = {n: read_labels(neims / layout[n]) for n in list(LEAKAGE_GUARD_DATASETS) + targets}
    all_smiles = pd.concat([l["smiles"] for l in labels.values()])
    print(f"[KEYS] computing connectivity keys for {all_smiles.nunique()} unique SMILES "
          f"({args.workers} workers)")
    smi_key = compute_keys(all_smiles, args.workers)

    guard: Set[str] = set()
    guard_source: Dict[str, str] = {}
    for n in LEAKAGE_GUARD_DATASETS:
        ks = {smi_key.get(s) for s in labels[n]["smiles"]} - {None}
        for k in ks:
            guard_source.setdefault(k, n)
        guard |= ks
        print(f"[GUARD] {n}: {len(ks)} molecules")
    print(f"[GUARD] {len(guard)} molecules in total")

    failures = 0
    for n in targets:
        d = neims / layout[n]
        spec_key = labels[n].set_index("spec")["smiles"].map(smi_key)
        split = read_split(d / SOURCE_SPLIT_FILE)

        missing = (~split["name"].isin(spec_key.index)).sum()
        if missing:
            print(f"[WARN] {n}: {missing} split rows have no labels.tsv entry (kept)")

        if not args.check:
            drop = guarded_frame(split, spec_key, guard)
            removed = split[drop].copy()
            removed["smiles"] = removed["name"].map(labels[n].set_index("spec")["smiles"])
            removed["key"] = removed["name"].map(spec_key)
            removed["guard_dataset"] = removed["key"].map(guard_source)

            out = d / GUARDED_SPLIT_FILE
            tmp = out.with_suffix(".tmp")
            write_split(split[~drop], tmp)
            os.replace(tmp, out)
            removed.to_csv(d / (GUARDED_SPLIT_FILE[: -len(".tsv")] + REMOVED_SUFFIX),
                           sep="\t", index=False)

            by_split = removed["split"].value_counts().to_dict()
            print(f"[WRITE] {n}: removed {len(removed)} rows "
                  f"(train={by_split.get('train', 0)}, val={by_split.get('val', 0)}) "
                  f"-> {out.relative_to(neims)}")

        err = verify(d, spec_key, guard)
        if err:
            failures += 1
            print(f"  [FAIL] {n}: {err}")
        else:
            print(f"  [OK]   {n}: no guarded molecules in train/val; test rows unchanged")

    if failures:
        print(f"\n{failures} dataset(s) FAILED verification -- do not train on them.")
        sys.exit(1)
    print("\nAll guarded split files verified.")


if __name__ == "__main__":
    main()
