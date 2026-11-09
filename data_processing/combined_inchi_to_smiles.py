"""
Write the molecules of the combined fp2mol dataset (train + val) as one file of 2D SMILES.

Run from data_processing/:
    python combined_inchi_to_smiles.py --workers 8
"""

import argparse
import os
from multiprocessing import Pool
from pathlib import Path
from typing import Optional

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


def to_2d_smiles(inchi: str) -> Optional[str]:
    mol = Chem.MolFromInchi(inchi)
    return Chem.MolToSmiles(mol, isomericSmiles=False) if mol is not None else None


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-dir", default="../data", metavar="PATH")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args()

    combined = Path(args.data_dir) / "fp2mol" / "combined"
    # Not in preprocessed/, which holds only the split files the datasets read.
    output = combined / "combined_2d_smiles.csv"

    inchis = []
    for split in ("train", "val"):
        rows = pd.read_csv(combined / "preprocessed" / f"combined_{split}.csv")["inchi"]
        print(f"[READ] {split}: {len(rows)} InChIs")
        inchis.extend(rows.dropna())

    with Pool(args.workers) as pool:
        converted = pool.map(to_2d_smiles, inchis, chunksize=2000)

    failed = [inchi for inchi, smi in zip(inchis, converted) if smi is None]
    if failed:
        print(f"[WARN] {len(failed)} InChIs could not be converted, e.g. {failed[:3]}")

    smiles = pd.Series([s for s in converted if s is not None], name="smiles")
    unique = smiles.drop_duplicates()
    print(f"[DONE] {len(smiles)} converted, {len(unique)} unique SMILES")

    # Written under a temporary name so a killed job can't leave a partial file behind.
    tmp = output.with_suffix(".csv.tmp")
    unique.to_csv(tmp, index=False)
    os.replace(tmp, output)
    print(f"[DONE] wrote {output}")


if __name__ == "__main__":
    main()
