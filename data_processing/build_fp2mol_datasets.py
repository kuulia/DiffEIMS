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


def _passes_filter(mol, filter_atoms: bool = True,
                   max_heavy_atoms: Optional[int] = None) -> bool:
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
        if max_heavy_atoms is not None and mol.GetNumHeavyAtoms() > max_heavy_atoms:
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

    max_heavy_atoms: Optional[int] = None  # set per-instance to cap heavy atom count

    def _smiles_to_inchi(self, smi: str, filter_atoms: bool = True) -> Optional[str]:
        mol, _ = _canonicalize(smi)
        if mol is None or not _passes_filter(mol, filter_atoms, self.max_heavy_atoms):
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

    def get_exclusions(self) -> Set[str]:
        """Return test+val InChIs without writing any output."""
        df = self._load()
        exclusions: Set[str] = set()
        for _, row in tqdm(df.iterrows(), total=len(df),
                           desc=f"Collecting exclusions from {self.name}", leave=False):
            if row["split"] in ("test", "val"):
                inchi = self._smiles_to_inchi(row["smiles"], filter_atoms=True)
                if inchi:
                    exclusions.add(inchi)
        return exclusions


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
# Dataset registry
# ---------------------------------------------------------------------------

def _build_registry(data: Path, fp2mol: Path):
    spectral: List[SpectralDataPreprocessor] = [
        SpectralDataPreprocessor(
            name="canopus",
            data_dir=data / "canopus",
            output_dir=fp2mol / "canopus/preprocessed",
            split_file="splits/canopus_hplus_100_0.tsv",
        ),
        SpectralDataPreprocessor(
            name="msg",
            data_dir=data / "msg",
            output_dir=fp2mol / "msg/preprocessed",
        ),
        SpectralDataPreprocessor(
            name="neims_tms",
            data_dir=data / "neims_tms",
            output_dir=data / "neims_tms/neims_tms/preprocessed",
        ),
        SpectralDataPreprocessor(
            name="msg_neims",
            data_dir=data / "msg_neims",
            output_dir=data / "msg_neims/msg_neims/preprocessed",
        ),
        SpectralDataPreprocessor(
            name="gecko_atmomaccs",
            data_dir=data / "gecko_atmomaccs",
            output_dir=data / "gecko_atmomaccs/gecko_atmomaccs/preprocessed",
        ),
        SpectralDataPreprocessor(
            name="neims",
            data_dir=data / "neims",
            output_dir=data / "neims/neims/preprocessed",
        ),
        SpectralDataPreprocessor(
            name="mixed_augment_test",
            data_dir=data / "mixed_augment_test",
            output_dir=data / "mixed_augment_test/preprocessed",
        ),
        SpectralDataPreprocessor(
            name="mixed_augment",
            data_dir=data / "mixed_augment",
            output_dir=data / "mixed_augment/preprocessed",
        ),
        SpectralDataPreprocessor(
            name="mixed_atmomaccs",
            data_dir=data / "mixed_atmomaccs",
            output_dir=data / "mixed_atmomaccs/preprocessed",
            split_file="split_random.tsv",
        ),
    ]
    structure: List[StructureDataPreprocessor] = [
        HMDBPreprocessor(
            sdf_path=fp2mol / "raw/structures.sdf",
            output_dir=fp2mol / "hmdb/preprocessed",
        ),
        DSSToxPreprocessor(
            raw_dir=fp2mol / "raw",
            output_dir=fp2mol / "dss/preprocessed",
        ),
        CSVSmilesPreprocessor(
            name="coconut",
            csv_path=fp2mol / "raw/coconut_csv-03-2025.csv",
            smiles_col="canonical_smiles",
            output_dir=fp2mol / "coconut/preprocessed",
        ),
        CSVSmilesPreprocessor(
            name="moses",
            csv_path=fp2mol / "raw/moses.csv",
            smiles_col="SMILES",
            output_dir=fp2mol / "moses/preprocessed",
        ),
        ATMOMACSPreprocessor(
            data_dir=data / "atmomaccs",
            output_dir=fp2mol / "atmomaccs/preprocessed",
        ),
    ]
    combined = CombinedPreprocessor(
        sources=structure,
        output_dir=fp2mol / "combined/preprocessed",
    )
    return spectral, structure, combined


def _run(ds: DataPreprocessor, excluded_inchis: Set[str]) -> Set[str]:
    try:
        return ds.process(excluded_inchis)
    except Exception as e:
        print(f"[SKIP] {getattr(ds, 'name', type(ds).__name__)}: {e}")
        return set()


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess fp2mol datasets.")
    parser.add_argument(
        "--dataset", "-d",
        metavar="NAME",
        help="Run only this dataset. Omit to run all.",
    )
    parser.add_argument(
        "--exclude-from", "-e",
        nargs="+",
        metavar="NAME",
        dest="exclude_from",
        help="Spectral dataset names whose test/val InChIs are excluded from training "
             "(default when running a single dataset: none).",
    )
    parser.add_argument(
        "--data-dir",
        default="../data",
        metavar="PATH",
        help="Root data directory (default: ../data).",
    )
    parser.add_argument(
        "--max-heavy-atoms",
        type=int,
        default=None,
        metavar="N",
        dest="max_heavy_atoms",
        help="Drop molecules with more than N heavy (non-hydrogen) atoms. "
             "Omit to keep all sizes (default: no cap).",
    )
    args = parser.parse_args()

    DATA   = Path(args.data_dir)
    FP2MOL = DATA / "fp2mol"

    spectral_datasets, structure_datasets, combined = _build_registry(DATA, FP2MOL)

    if args.max_heavy_atoms is not None:
        for ds in spectral_datasets + structure_datasets + [combined]:
            ds.max_heavy_atoms = args.max_heavy_atoms
        print(f"[INFO] Heavy atom cap: {args.max_heavy_atoms}")

    spectral_by_name: dict = {ds.name: ds for ds in spectral_datasets}
    all_datasets: dict = {ds.name: ds for ds in spectral_datasets + structure_datasets}
    all_datasets["combined"] = combined

    # Build exclusion set from requested spectral sources
    excluded_inchis: Set[str] = set()
    exclude_sources = args.exclude_from
    if exclude_sources is None and args.dataset is None:
        # Full run: collect exclusions from every spectral dataset
        exclude_sources = list(spectral_by_name.keys())

    for name in (exclude_sources or []):
        if name not in spectral_by_name:
            print(f"[WARN] --exclude-from '{name}' is not a known spectral dataset, skipping.")
            continue
        try:
            excluded_inchis |= spectral_by_name[name].get_exclusions()
        except Exception as e:
            print(f"[WARN] Could not collect exclusions from '{name}': {e}")

    if args.dataset:
        if args.dataset not in all_datasets:
            parser.error(
                f"Unknown dataset '{args.dataset}'. "
                f"Available: {', '.join(sorted(all_datasets))}"
            )
        _run(all_datasets[args.dataset], excluded_inchis)
    else:
        # Full pipeline: spectral process() also writes output, so run them again
        for ds in spectral_datasets:
            _run(ds, excluded_inchis)
        for ds in structure_datasets:
            _run(ds, excluded_inchis)
        _run(combined, excluded_inchis)
