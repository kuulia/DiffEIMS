import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from tqdm import tqdm

random.seed(42)

RDLogger.logger().setLevel(RDLogger.CRITICAL)

FILTER_ATOMS = {'C', 'N', 'S', 'O', 'F', 'Cl', 'H', 'P', 'B', 'Br', 'I', 'Si'}


# ---------------------------------------------------------------------------
# Shared mol utilities
# ---------------------------------------------------------------------------

def _canonicalize(smi: str):
    """Return (canonical_mol, canonical_smi) with stereo stripped, or (None, None)."""
    try:
        mol = Chem.MolFromSmiles(smi)
        smi = Chem.MolToSmiles(mol, isomericSmiles=False)
        return Chem.MolFromSmiles(smi), smi
    except Exception:
        return None, None


def _passes_filter(mol, filter_atoms: bool = True) -> bool:
    if mol is None:
        return False
    try:
        if len(Chem.GetMolFrags(mol, asMols=False)) != 1:
            return False
        mw = Descriptors.MolWt(mol)
        if not (16.0 <= mw < 1500):
            return False
        if filter_atoms:
            if any(a.GetSymbol() not in FILTER_ATOMS for a in mol.GetAtoms()):
                return False
    except Exception:
        return False
    return True


def _to_inchi(mol) -> Optional[str]:
    try:
        return Chem.MolToInchi(mol)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class DataPreprocessor(ABC):
    """Base class for fp2mol dataset preprocessing."""

    def _smiles_to_inchi(self, smi: str, filter_atoms: bool = True) -> Optional[str]:
        mol, _ = _canonicalize(smi)
        if mol is None or not _passes_filter(mol, filter_atoms):
            return None
        return _to_inchi(mol)

    def _save_split(self, inchis: Union[List[str], Set[str]], path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(list(inchis), columns=["inchi"]).to_csv(path, index=False)

    @abstractmethod
    def process(self, excluded_inchis: Set[str]) -> Set[str]:
        """Run preprocessing. Returns the test+val InChIs to be excluded downstream."""
        ...


# ---------------------------------------------------------------------------
# Spectral datasets  (labels.tsv + split.tsv, pre-defined train/test/val)
# ---------------------------------------------------------------------------

class SpectralDataPreprocessor(DataPreprocessor):
    """
    Handles datasets that ship with labels.tsv and split.tsv.
    Train split is filtered and deduplicated; test/val are kept as-is.
    """

    def __init__(self, name: str, data_dir: Union[str, Path], output_dir: Union[str, Path],
                 split_file: str = "split.tsv"):
        self.name = name
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.split_file = split_file

    def _load(self) -> pd.DataFrame:
        labels = pd.read_csv(self.data_dir / "labels.tsv", sep="\t")
        labels["name"] = labels["spec"]
        labels = labels[["name", "smiles"]].reset_index(drop=True)
        split = pd.read_csv(self.data_dir / self.split_file, sep="\t")
        return labels.merge(split, on="name")

    def process(self, excluded_inchis: Set[str]) -> Set[str]:
        df = self._load()
        train, test, val = [], [], []

        for _, row in tqdm(df.iterrows(), total=len(df),
                           desc=f"Processing {self.name}", leave=False):
            inchi = self._smiles_to_inchi(row["smiles"], filter_atoms=True)
            if inchi is None:
                continue
            if row["split"] == "train" and inchi not in excluded_inchis:
                train.append(inchi)
            elif row["split"] == "test":
                test.append(inchi)
            elif row["split"] == "val":
                val.append(inchi)

        self._save_split(set(train), self.output_dir / f"{self.name}_train.csv")
        self._save_split(test,       self.output_dir / f"{self.name}_test.csv")
        self._save_split(val,        self.output_dir / f"{self.name}_val.csv")

        return set(test) | set(val)


# ---------------------------------------------------------------------------
# Structure-only datasets  (raw SMILES, auto 95/5 split)
# ---------------------------------------------------------------------------

class StructureDataPreprocessor(DataPreprocessor):
    """
    Handles structure-only datasets with no pre-defined splits.
    Subclasses implement load_raw_smiles(); this class owns filtering and splitting.
    """

    def __init__(self, name: str, output_dir: Union[str, Path], val_frac: float = 0.05):
        self.name = name
        self.output_dir = Path(output_dir)
        self.val_frac = val_frac

    @abstractmethod
    def load_raw_smiles(self) -> Set[str]:
        """Return raw SMILES strings (may be noisy)."""
        ...

    def process(self, excluded_inchis: Set[str]) -> Set[str]:
        raw = self.load_raw_smiles()

        inchis = set()
        for smi in tqdm(raw, desc=f"Filtering {self.name}", leave=False):
            inchi = self._smiles_to_inchi(smi, filter_atoms=True)
            if inchi:
                inchis.add(inchi)

        inchis = list(inchis)
        random.shuffle(inchis)
        split_idx = int((1 - self.val_frac) * len(inchis))
        train = [i for i in inchis[:split_idx] if i not in excluded_inchis]
        val   = inchis[split_idx:]

        self._save_split(train, self.output_dir / f"{self.name}_train.csv")
        self._save_split(val,   self.output_dir / f"{self.name}_val.csv")

        return set()  # structure-only datasets don't contribute to exclusions


# ---------------------------------------------------------------------------
# Concrete structure dataset loaders
# ---------------------------------------------------------------------------

class HMDBPreprocessor(StructureDataPreprocessor):
    def __init__(self, sdf_path: Union[str, Path], output_dir: Union[str, Path]):
        super().__init__("hmdb", output_dir)
        self.sdf_path = Path(sdf_path)

    def load_raw_smiles(self) -> Set[str]:
        smiles, append_next = [], False
        with open(self.sdf_path) as f:
            for line in tqdm(f, desc="Loading HMDB SDF", leave=False):
                if append_next:
                    smiles.append(line.strip())
                    append_next = False
                if line.startswith("> <SMILES>"):
                    append_next = True
        return set(smiles)


class DSSToxPreprocessor(StructureDataPreprocessor):
    def __init__(self, raw_dir: Union[str, Path], output_dir: Union[str, Path], n_files: int = 13):
        super().__init__("dss", output_dir)
        self.raw_dir = Path(raw_dir)
        self.n_files = n_files

    def load_raw_smiles(self) -> Set[str]:
        smiles = set()
        for i in tqdm(range(1, self.n_files + 1), desc="Loading DSSTox", leave=False):
            df = pd.read_excel(self.raw_dir / f"DSSToxDump{i}.xlsx")
            smiles.update(df[df["SMILES"].notnull()]["SMILES"])
        return smiles


class CSVSmilesPreprocessor(StructureDataPreprocessor):
    """Generic loader for a single CSV with a named SMILES column."""

    def __init__(self, name: str, csv_path: Union[str, Path], smiles_col: str,
                 output_dir: Union[str, Path]):
        super().__init__(name, output_dir)
        self.csv_path = Path(csv_path)
        self.smiles_col = smiles_col

    def load_raw_smiles(self) -> Set[str]:
        df = pd.read_csv(self.csv_path)
        return set(df[self.smiles_col].dropna())


class ATMOMACSPreprocessor(StructureDataPreprocessor):
    def __init__(self, data_dir: Union[str, Path], output_dir: Union[str, Path],
                 datasets: Tuple[str, ...] = ("wang", "li", "ferraz-caetano", "kruger-broad")):
        super().__init__("atmomaccs", output_dir)
        self.data_dir = Path(data_dir)
        self.datasets = datasets

    def load_raw_smiles(self) -> Set[str]:
        smiles = set()
        for ds in self.datasets:
            arr = np.loadtxt(self.data_dir / f"{ds}-smiles.txt",
                             dtype=np.str_, comments=None)
            smiles.update(arr.tolist())
        return smiles


# ---------------------------------------------------------------------------
# Combined dataset — merges all structure sources
# ---------------------------------------------------------------------------

class CombinedPreprocessor(DataPreprocessor):
    """Merges InChI pools from multiple StructureDataPreprocessors into one dataset."""

    def __init__(self, sources: List[StructureDataPreprocessor], output_dir: Union[str, Path],
                 val_frac: float = 0.05):
        self.sources = sources
        self.output_dir = Path(output_dir)
        self.val_frac = val_frac

    def process(self, excluded_inchis: Set[str]) -> Set[str]:
        all_inchis: Set[str] = set()
        for src in self.sources:
            raw = src.load_raw_smiles()
            for smi in tqdm(raw, desc=f"Filtering {src.name} for combined", leave=False):
                inchi = src._smiles_to_inchi(smi, filter_atoms=True)
                if inchi:
                    all_inchis.add(inchi)

        inchis = list(all_inchis)
        random.shuffle(inchis)
        split_idx = int((1 - self.val_frac) * len(inchis))
        train = [i for i in inchis[:split_idx] if i not in excluded_inchis]
        val   = inchis[split_idx:]

        self._save_split(train, self.output_dir / "combined_train.csv")
        self._save_split(val,   self.output_dir / "combined_val.csv")

        return set()


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    DATA   = Path("../data")
    FP2MOL = DATA / "fp2mol"

    spectral_datasets = [
        SpectralDataPreprocessor(
            name="canopus",
            data_dir=DATA / "canopus",
            output_dir=FP2MOL / "canopus/preprocessed",
            split_file="splits/canopus_hplus_100_0.tsv",
        ),
        SpectralDataPreprocessor(
            name="msg",
            data_dir=DATA / "msg",
            output_dir=FP2MOL / "msg/preprocessed",
        ),
        SpectralDataPreprocessor(
            name="neims_tms",
            data_dir=DATA / "neims_tms",
            output_dir=DATA / "neims_tms/neims_tms/preprocessed",
        ),
        SpectralDataPreprocessor(
            name="msg_neims",
            data_dir=DATA / "msg_neims",
            output_dir=DATA / "msg_neims/msg_neims/preprocessed",
        ),
        SpectralDataPreprocessor(
            name="gecko_atmomaccs",
            data_dir=DATA / "gecko_atmomaccs",
            output_dir=DATA / "gecko_atmomaccs/gecko_atmomaccs/preprocessed",
        ),
        SpectralDataPreprocessor(
            name="neims",
            data_dir=DATA / "neims",
            output_dir=DATA / "neims/neims/preprocessed",
        ),
        SpectralDataPreprocessor(
            name="mixed_augment_test",
            data_dir=DATA / "mixed_augment_test",
            output_dir=DATA / "mixed_augment_test/preprocessed",
        ),
    ]

    structure_datasets: List[StructureDataPreprocessor] = [
        HMDBPreprocessor(
            sdf_path=FP2MOL / "raw/structures.sdf",
            output_dir=FP2MOL / "hmdb/preprocessed",
        ),
        DSSToxPreprocessor(
            raw_dir=FP2MOL / "raw",
            output_dir=FP2MOL / "dss/preprocessed",
        ),
        CSVSmilesPreprocessor(
            name="coconut",
            csv_path=FP2MOL / "raw/coconut_csv-03-2025.csv",
            smiles_col="canonical_smiles",
            output_dir=FP2MOL / "coconut/preprocessed",
        ),
        CSVSmilesPreprocessor(
            name="moses",
            csv_path=FP2MOL / "raw/moses.csv",
            smiles_col="SMILES",
            output_dir=FP2MOL / "moses/preprocessed",
        ),
        ATMOMACSPreprocessor(
            data_dir=DATA / "atmomaccs",
            output_dir=FP2MOL / "atmomaccs/preprocessed",
        ),
    ]

    # Spectral test/val sets are excluded from all downstream training splits
    excluded_inchis: Set[str] = set()
    for ds in spectral_datasets:
        try:
            excluded_inchis |= ds.process(excluded_inchis)
        except Exception as e:
            print(f"[SKIP] {ds.name}: {e}")

    for ds in structure_datasets:
        try:
            ds.process(excluded_inchis)
        except Exception as e:
            print(f"[SKIP] {ds.name}: {e}")

    try:
        CombinedPreprocessor(
            sources=structure_datasets,
            output_dir=FP2MOL / "combined/preprocessed",
        ).process(excluded_inchis)
    except Exception as e:
        print(f"[SKIP] combined: {e}")
