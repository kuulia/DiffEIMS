import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Set, Tuple, Union

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from tqdm import tqdm

random.seed(42)

RDLogger.logger().setLevel(RDLogger.CRITICAL)

FILTER_ATOMS = {"C", "N", "S", "O", "F", "Cl", "H", "P", "B", "Br", "I", "Si"}


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


def _passes_filter(
    mol, filter_atoms: bool = True, max_heavy_atoms: Optional[int] = None
) -> bool:
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
# Spectral datasets  (labels.tsv + split_random.tsv, pre-defined train/test/val)
# ---------------------------------------------------------------------------


class SpectralDataPreprocessor(DataPreprocessor):
    """
    Handles datasets that ship with labels.tsv and a split file.
    Train split is filtered and deduplicated; test/val are kept as-is.
    """

    def __init__(
        self,
        name: str,
        data_dir: Union[str, Path],
        output_dir: Union[str, Path],
        split_file: str = "split_random.tsv",
    ):
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

        for _, row in tqdm(
            df.iterrows(), total=len(df), desc=f"Processing {self.name}", leave=False
        ):
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
        self._save_split(test, self.output_dir / f"{self.name}_test.csv")
        self._save_split(val, self.output_dir / f"{self.name}_val.csv")

        # NeimsDataset loads all three splits, so a header-only CSV breaks training
        # later with an opaque collate error. Say so here instead. Expected and
        # harmless for atmomaccs / atmomaccs_tms, which have no val rows and are
        # never trained on (see LEAKAGE_GUARD_DATASETS).
        for split_name, rows in (("train", train), ("test", test), ("val", val)):
            if not rows:
                n_raw = int((df["split"] == split_name).sum())
                reason = (
                    f"{self.data_dir / self.split_file} has no '{split_name}' rows"
                    if n_raw == 0
                    else f"all {n_raw} '{split_name}' rows were filtered out or excluded "
                    f"(e.g. a full run excludes every *_atmomaccs_test dataset's test "
                    f"split, which is the whole atmomaccs pool)"
                )
                print(
                    f"[WARN] {self.name}: {split_name} split is empty -- {reason}. "
                    f"{self.name}_{split_name}.csv contains only a header."
                )

        return set(test) | set(val)

    def get_exclusions(
        self, splits: Optional[Tuple[str, ...]] = ("test", "val")
    ) -> Set[str]:
        """
        Return the InChIs of this dataset without writing any output.

        splits=("test", "val") keeps the usual held-out hygiene; splits=None takes
        every row, which is what the atmomaccs leakage guard needs (see
        LEAKAGE_GUARD_DATASETS).
        """
        df = self._load()
        exclusions: Set[str] = set()
        for _, row in tqdm(
            df.iterrows(),
            total=len(df),
            desc=f"Collecting exclusions from {self.name}",
            leave=False,
        ):
            if splits is not None and row["split"] not in splits:
                continue
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
        val = inchis[split_idx:]

        self._save_split(train, self.output_dir / f"{self.name}_train.csv")
        self._save_split(val, self.output_dir / f"{self.name}_val.csv")

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
    def __init__(
        self, raw_dir: Union[str, Path], output_dir: Union[str, Path], n_files: int = 13
    ):
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

    def __init__(
        self,
        name: str,
        csv_path: Union[str, Path],
        smiles_col: str,
        output_dir: Union[str, Path],
    ):
        super().__init__(name, output_dir)
        self.csv_path = Path(csv_path)
        self.smiles_col = smiles_col

    def load_raw_smiles(self) -> Set[str]:
        df = pd.read_csv(self.csv_path)
        return set(df[self.smiles_col].dropna())


# ---------------------------------------------------------------------------
# Combined dataset — merges all structure sources
# ---------------------------------------------------------------------------


class CombinedPreprocessor(DataPreprocessor):
    """Merges InChI pools from multiple StructureDataPreprocessors into one dataset."""

    def __init__(
        self,
        sources: List[StructureDataPreprocessor],
        output_dir: Union[str, Path],
        val_frac: float = 0.05,
        name: str = "combined",
    ):
        self.name = name
        self.sources = sources
        self.output_dir = Path(output_dir)
        self.val_frac = val_frac

    def process(self, excluded_inchis: Set[str]) -> Set[str]:
        all_inchis: Set[str] = set()
        for src in self.sources:
            raw = src.load_raw_smiles()
            for smi in tqdm(
                raw, desc=f"Filtering {src.name} for combined", leave=False
            ):
                inchi = src._smiles_to_inchi(smi, filter_atoms=True)
                if inchi:
                    all_inchis.add(inchi)

        inchis = list(all_inchis)
        random.shuffle(inchis)
        split_idx = int((1 - self.val_frac) * len(inchis))
        train = [i for i in inchis[:split_idx] if i not in excluded_inchis]
        val = inchis[split_idx:]

        self._save_split(train, self.output_dir / f"{self.name}_train.csv")
        self._save_split(val, self.output_dir / f"{self.name}_val.csv")

        return set()


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------


# Spectral datasets now live under data/neims/<folder>/, with labels.tsv and
# split_random.tsv inside a mist_inputs directory whose depth varies per dataset.
# Preprocessed CSVs are written to <that dir>/preprocessed/ so they sit next to the
# labels/split/collated-pkl files they are derived from.
SPECTRAL_LAYOUT: Tuple[Tuple[str, str], ...] = (
    # (registry name, path relative to data/neims)
    ("atmomaccs", "atmomaccs_new/mist_inputs"),
    ("atmomaccs_tms", "atmomaccs_new/mist_inputs_tms"),
    ("mixed_augment", "mixed_augment/mist_inputs/mixed_augment"),
    ("mixed_augment_tms", "mixed_augment_tms/mist_inputs/mixed_augment_tms"),
    ("gecko_new", "gecko_new/mist_inputs"),
    ("gecko_tms", "gecko_tms/mist_inputs"),
    ("gecko_new_atmomaccs_test", "gecko_new_atmomaccs_test/mist_inputs"),
    (
        "gecko_new_mixed_augment_atmomaccs_test",
        "gecko_new_mixed_augment_atmomaccs_test/mist_inputs",
    ),
    (
        "gecko_tms_mixed_augment_atmomaccs_tms_test",
        "gecko_tms_mixed_augment_atmomaccs_tms_test/mist_inputs",
    ),
    (
        "gecko_tms_mixed_augment_tms_atmomaccs_tms_test",
        "gecko_tms_mixed_augment_tms_atmomaccs_tms_test/mist_inputs",
    ),
    ("combined_atmomaccs_test", "combined_atmomaccs_test/mist_inputs"),
)

# Every molecule in these datasets is held out of every *other* dataset's train
# split, across all of their own splits rather than just test/val. These are the
# evaluation sets the models are ultimately judged on, so any occurrence of one of
# their molecules in training data is leakage. atmomaccs_new/mist_inputs covers all
# 9375 atmomaccs spectra (wang + li + ferraz-caetano + kruger-confined), and
# mist_inputs_tms the TMS-derivatized counterparts, so the two aggregates are
# sufficient -- the per-subset directories under mist_inputs/ are subsets of them.
LEAKAGE_GUARD_DATASETS: Tuple[str, ...] = ("atmomaccs", "atmomaccs_tms")

# Written by build_guarded_splits.py: split_random.tsv minus every train/val row whose
# molecule is a guard-dataset molecule. spec2mol trains from split files rather than
# from the preprocessed CSVs, so the guard has to live in the split file too; reading
# the same guarded file here keeps both training stages on one definition of "train".
GUARDED_SPLIT_FILE = "split_random_guarded.tsv"


def _build_registry(data: Path, fp2mol: Path):
    neims = data / "neims"

    spectral: List[SpectralDataPreprocessor] = [
        SpectralDataPreprocessor(
            name=name,
            data_dir=neims / rel,
            output_dir=neims / rel / "preprocessed",
            # The guard datasets are the source of the guard, so they keep the
            # original split.
            split_file=(
                "split_random.tsv"
                if name in LEAKAGE_GUARD_DATASETS
                else GUARDED_SPLIT_FILE
            ),
        )
        for name, rel in SPECTRAL_LAYOUT
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

def _verify_no_leakage(
    datasets: List[DataPreprocessor], guard_inchis: Set[str]
) -> bool:
    """Read back every train CSV that was written and assert the guard set is absent."""
    if not guard_inchis:
        return True

    print("\n[VERIFY] Checking written train splits against the leakage guard set")
    clean = True
    for ds in datasets:
        name = getattr(ds, "name", None)
        if name is None or name in LEAKAGE_GUARD_DATASETS:
            continue
        path = Path(ds.output_dir) / f"{name}_train.csv"
        if not path.exists():
            continue
        train_inchis = set(pd.read_csv(path)["inchi"].dropna())
        leaked = train_inchis & guard_inchis
        if leaked:
            clean = False
            print(f"  [FAIL] {name}: {len(leaked)} guarded molecules in {path}")
        else:
            print(f"  [OK]   {name}: {len(train_inchis)} train molecules, 0 leaked")

    if clean:
        print("[VERIFY] No atmomaccs / atmomaccs_tms leakage in any train split.")
    else:
        print("[VERIFY] LEAKAGE DETECTED -- do not train on these splits.")
    return clean


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Preprocess fp2mol datasets.")
    parser.add_argument(
        "--dataset",
        "-d",
        metavar="NAME",
        help="Run only this dataset. Omit to run all.",
    )
    parser.add_argument(
        "--exclude-from",
        "-e",
        nargs="+",
        metavar="NAME",
        dest="exclude_from",
        help="Spectral dataset names whose test/val InChIs are excluded from training "
        "(default when running a single dataset: none). The atmomaccs leakage guard "
        "is applied on top of this and is not affected by it.",
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
    parser.add_argument(
        "--no-leakage-guard",
        action="store_true",
        dest="no_leakage_guard",
        help="Disable the atmomaccs / atmomaccs_tms leakage guard. Only for debugging "
        "-- training data produced this way is not safe to evaluate on atmomaccs.",
    )
    args = parser.parse_args()

    DATA = Path(args.data_dir)
    FP2MOL = DATA / "fp2mol"

    spectral_datasets, structure_datasets, combined = _build_registry(DATA, FP2MOL)

    if args.max_heavy_atoms is not None:
        for ds in spectral_datasets + structure_datasets + [combined]:
            ds.max_heavy_atoms = args.max_heavy_atoms
        print(f"[INFO] Heavy atom cap: {args.max_heavy_atoms}")

    spectral_by_name: dict = {ds.name: ds for ds in spectral_datasets}
    all_datasets: dict = {ds.name: ds for ds in spectral_datasets + structure_datasets}
    all_datasets[combined.name] = combined

    # ---- Leakage guard: every atmomaccs / atmomaccs_tms molecule, all splits ----
    guard_inchis: Set[str] = set()
    if args.no_leakage_guard:
        print("[WARN] Leakage guard DISABLED -- atmomaccs molecules may enter training.")
    else:
        for name in LEAKAGE_GUARD_DATASETS:
            ds = spectral_by_name.get(name)
            if ds is None:
                parser.error(f"Leakage guard dataset '{name}' is not in the registry.")
            try:
                collected = ds.get_exclusions(splits=None)
            except Exception as e:
                # Silently training on leaked data is worse than failing loudly.
                parser.error(
                    f"Leakage guard dataset '{name}' could not be read ({e}). "
                    f"Expected {ds.data_dir / 'labels.tsv'} and "
                    f"{ds.data_dir / ds.split_file}. "
                    f"Pass --no-leakage-guard only if you accept the leakage risk."
                )
            print(f"[GUARD] {name}: {len(collected)} unique molecules held out")
            guard_inchis |= collected
        print(f"[GUARD] {len(guard_inchis)} unique molecules in total")

    # ---- Standard held-out hygiene: test/val of the requested spectral sources ----
    excluded_inchis: Set[str] = set()
    exclude_sources = args.exclude_from
    if exclude_sources is None and args.dataset is None:
        # Full run: collect exclusions from every spectral dataset
        exclude_sources = list(spectral_by_name.keys())

    for name in exclude_sources or []:
        if name not in spectral_by_name:
            print(
                f"[WARN] --exclude-from '{name}' is not a known spectral dataset, skipping."
            )
            continue
        try:
            excluded_inchis |= spectral_by_name[name].get_exclusions()
        except Exception as e:
            print(f"[WARN] Could not collect exclusions from '{name}': {e}")

    def _exclusions_for(ds: DataPreprocessor) -> Set[str]:
        """
        The guard datasets are the evaluation sets themselves. They are never trained
        on -- the models that are evaluated on atmomaccs are the *_atmomaccs_test
        datasets, which already carry the whole atmomaccs pool as their test split --
        so their own train/val/test split is irrelevant and is left alone. Everything
        else is additionally guarded.
        """
        if getattr(ds, "name", None) in LEAKAGE_GUARD_DATASETS:
            return excluded_inchis
        return excluded_inchis | guard_inchis

    if args.dataset:
        if args.dataset not in all_datasets:
            parser.error(
                f"Unknown dataset '{args.dataset}'. "
                f"Available: {', '.join(sorted(all_datasets))}"
            )
        target = all_datasets[args.dataset]
        _run(target, _exclusions_for(target))
        ran: List[DataPreprocessor] = [target]
    else:
        # Full pipeline: spectral process() also writes output, so run them again
        ran = []
        for ds in spectral_datasets:
            _run(ds, _exclusions_for(ds))
            ran.append(ds)
        for ds in structure_datasets:
            _run(ds, _exclusions_for(ds))
            ran.append(ds)
        _run(combined, _exclusions_for(combined))
        ran.append(combined)

    if not _verify_no_leakage(ran, guard_inchis):
        sys.exit(1)
