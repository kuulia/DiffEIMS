"""
sample_fp2mol_for_neims.py
==========================
Sample ~300 000 atmospherically-relevant, chemically-diverse molecules from
the fp2mol combined dataset for NEIMS EI-MS spectrum simulation.

Strategy
--------
1. Atmospheric pre-filter
     MW 100–1000 Da  |  atoms ∈ {C,H,N,O,S,F,Cl,P,B,Br,I,Si}  |  ≥1 oxygen  |  no fragment
     (formal charges are permitted: nitro/nitrate groups carry [N+]/[O-] in RDKit)
2. MW stratification
     9 equal-width bins: [100–200), [200–300), …, [900–1001)
     Equal target quota per bin  →  balanced MW coverage
3. Per-bin ATMOMACCS-V5 clustering (integer-count fingerprint, 205 features)
     MiniBatch k-means with k=1000 per bin
     Random sample ≈33 333 molecules from the cluster-stratified pool

Deduplication & exclusion
--------------------------
All deduplication and exclusion operations use InChI as the canonical identifier.
InChI is computed from the stereochemistry-stripped mol (isomericSmiles=False round-trip),
so stereo variants collapse to the same InChI — matching the behaviour of the previous
SMILES-based approach but using the more robust, language-independent InChI standard.

Usage
-----
python data_processing/sample_fp2mol_for_neims.py \\
    --input  ../data/fp2mol/combined/preprocessed/combined_train.csv \\
    --output ../data/fp2mol/neims_augment/sampled_300k.csv \\
    --n_jobs 8
"""

import argparse
import os
import random
import sys
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

# ── ATMOMACCS import (handles both script and module invocation) ───────────────
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.ATMOMACCS_no_binary import pyGenATMOMACCS  # V5 integer fingerprint

RDLogger.DisableLog("rdApp.*")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Element allowlist: union of ATMOMACCS benchmark datasets (Table I), excluding
# metals/salts (e.g. Na) since we require covalent, uncharged molecules.
ATMO_ATOMS = {"C", "N", "S", "O", "F", "Cl", "H", "P", "B", "Br", "I", "Si"}

# 9 equal-width MW bins covering 100–1000 Da
MW_BINS = [
    (100, 199.99),
    (200, 299.99),
    (300, 399.99),
    (400, 499.99),
    (500, 599.99),
    (600, 699.99),
    (700, 799.99),
    (800, 899.99),
    (900, 999.99),
    (1000, 1099.99),
    (1100, 1199.99),
    (1200, 1299.99),
    (1300, 1399.99),
    (1400, 1500),
]

TARGET_TOTAL = 700000
TARGET_PER_BIN = TARGET_TOTAL // len(MW_BINS)  # 33 333 per bin
N_CLUSTERS = 1_000  # k-means clusters per MW bin
RANDOM_SEED = 42

# ─────────────────────────────────────────────────────────────────────────────
# Step 1 – Atmospheric pre-filter
# ─────────────────────────────────────────────────────────────────────────────


def _filter_mol(mol) -> Optional[tuple[str, float, str]]:
    """
    Apply atmospheric relevance rules to an RDKit mol.

    Rules
    -----
    * MW:    100 ≤ MW ≤ 1500 Da
    * Atoms: only {C, H, N, O, S, F, Cl, P, B, Br, I, Si}
    * O:     at least one oxygen atom
    * C:     ≥ 1 and ≤ 40 (loose upper bound for GCxGC HR-TOF EI-MS)
    * No fragments (salts/mixtures excluded)
    * Formal charges permitted (nitro/nitrate groups carry [N+]/[O-] in RDKit)
    * Stereochemistry stripped (isomericSmiles=False round-trip before InChI)

    Returns (canonical_smiles, mw, inchi) or None.
    InChI is computed from the stereo-stripped mol and used for all
    deduplication and exclusion operations.
    """
    try:
        smi = Chem.MolToSmiles(mol, isomericSmiles=False)
        mol = Chem.MolFromSmiles(smi)
        if mol is None or "." in smi:
            return None

        mw = Descriptors.MolWt(mol)
        if not (100.0 <= mw <= 1500.0):
            return None

        num_O = 0
        num_C = 0
        for atom in mol.GetAtoms():
            sym = atom.GetSymbol()
            if sym not in ATMO_ATOMS:
                return None
            if sym == "O":
                num_O += 1
            elif sym == "C":
                num_C += 1

        if num_O < 1 or num_C < 1:
            return None

        # Loose upper bound for GCxGC HR-TOF EI-MS amenability
        if num_C > 40:
            return None

        inchi = Chem.MolToInchi(mol)
        if inchi is None:
            return None

        return smi, mw, inchi
    except Exception:
        return None


def _process_inchi(inchi: str) -> Optional[tuple]:
    """Worker: InChI → atmospheric-filtered (smiles, mw, inchi) or None."""
    RDLogger.DisableLog("rdApp.*")
    try:
        mol = Chem.MolFromInchi(inchi)
        if mol is None:
            return None
        return _filter_mol(mol)
    except Exception:
        return None


def _process_smiles(smi: str) -> Optional[tuple]:
    """Worker: SMILES → atmospheric-filtered (canonical_smiles, mw, inchi) or None."""
    RDLogger.DisableLog("rdApp.*")
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        return _filter_mol(mol)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 – ATMOMACCS-V5 descriptor
# ─────────────────────────────────────────────────────────────────────────────


def _compute_atmomaccs(smi: str) -> Optional[np.ndarray]:
    """Worker: SMILES → 205-dim ATMOMACCS-V5 integer vector or None."""
    RDLogger.DisableLog("rdApp.*")
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        return pyGenATMOMACCS(mol).astype(np.float32)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 – Per-bin clustering + sampling
# ─────────────────────────────────────────────────────────────────────────────


def _sample_bin(
    bin_pairs: list[tuple[str, str]],
    target_n: int,
    n_clusters: int,
    seed: int,
    n_jobs: int,
    bin_label: str,
) -> list[tuple[str, str]]:
    """
    Given (SMILES, InChI) pairs for one MW bin:
      1. Compute ATMOMACCS-V5 fingerprints on SMILES (parallel)
      2. MiniBatch k-means clustering
      3. Sample proportionally from each cluster

    Parameters
    ----------
    bin_pairs  : (smiles, inchi) tuples already passing the atmospheric filter
    target_n   : how many molecules to return from this bin
    n_clusters : k for k-means
    seed       : random seed
    n_jobs     : parallelism for ATMOMACCS computation
    bin_label  : string for progress messages

    Returns
    -------
    List of sampled (smiles, inchi) pairs (len ≤ target_n).
    InChI is carried through from the filter step — no recomputation needed.
    """
    rng = np.random.default_rng(seed)
    n = len(bin_pairs)

    # Short-circuit: if fewer molecules than quota, take all
    if n <= target_n:
        print(f"    {bin_label}: only {n:,} molecules — taking all")
        return list(bin_pairs)

    bin_smiles = [smi for smi, _ in bin_pairs]

    # ── Compute ATMOMACCS fingerprints ────────────────────────────────────────
    print(f"    {bin_label}: computing ATMOMACCS for {n:,} molecules …")
    fps_raw = Parallel(n_jobs=n_jobs)(
        delayed(_compute_atmomaccs)(smi)
        for smi in tqdm(bin_smiles, desc=f"      {bin_label} descriptors", leave=False)
    )

    # Drop failures — keep (smi, inchi, fp) triples
    valid_triples = [
        (smi, inchi, fp)
        for (smi, inchi), fp in zip(bin_pairs, fps_raw)
        if fp is not None
    ]
    if not valid_triples:
        warnings.warn(f"{bin_label}: no valid ATMOMACCS fingerprints — skipping bin")
        return []

    smiles_valid = [t[0] for t in valid_triples]
    inchi_valid = [t[1] for t in valid_triples]
    fps_valid = [t[2] for t in valid_triples]
    X = np.vstack(fps_valid)  # shape (n_valid, 205)
    n_valid = len(smiles_valid)

    # ── k-means ───────────────────────────────────────────────────────────────
    k = min(n_clusters, n_valid)
    print(f"    {bin_label}: k-means k={k} on {n_valid:,} molecules …")
    kmeans = MiniBatchKMeans(
        n_clusters=k,
        random_state=seed,
        batch_size=min(10_000, n_valid),
        n_init=3,
        verbose=0,
    )
    labels = kmeans.fit_predict(X)

    # ── Sample from each cluster ──────────────────────────────────────────────
    per_cluster = int(np.ceil(target_n / k))
    sampled: list[tuple[str, str]] = []
    for c in range(k):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        take = min(per_cluster, len(idx))
        chosen = rng.choice(idx, size=take, replace=False)
        sampled.extend((smiles_valid[i], inchi_valid[i]) for i in chosen)

    # Trim to exact target if we overshot
    if len(sampled) > target_n:
        chosen_idx = rng.choice(len(sampled), size=target_n, replace=False)
        sampled = [sampled[i] for i in chosen_idx]

    return sampled


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def _smiles_to_inchi_set(smiles_iterable) -> set:
    """
    Convert an iterable of SMILES to a set of stereo-stripped InChI strings.

    Stereochemistry is removed (isomericSmiles=False round-trip) before
    computing InChI so stereo variants of the same molecule map to one entry.
    Failures are silently skipped.
    """
    exclusion = set()
    for smi in smiles_iterable:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                smi_no_stereo = Chem.MolToSmiles(mol, isomericSmiles=False)
                mol_no_stereo = Chem.MolFromSmiles(smi_no_stereo)
                if mol_no_stereo is not None:
                    inchi = Chem.MolToInchi(mol_no_stereo)
                    if inchi is not None:
                        exclusion.add(inchi)
        except Exception:
            pass
    return exclusion


def _build_exclusion_set(labels_path: str) -> set:
    """
    Load a labels.tsv (must have a 'smiles' column) and return a set of
    stereo-stripped InChI strings for ALL molecules in the file.

    Used for the ATMOMACCS dataset where every molecule is part of the
    evaluation set and should be excluded from augmentation sampling.
    """
    df = pd.read_csv(labels_path, sep="\t")
    return _smiles_to_inchi_set(df["smiles"].dropna())


def _build_split_exclusion_set(
    labels_path: str,
    split_path: str,
    splits: tuple = ("val", "test"),
) -> set:
    """
    Load a labels.tsv + split.tsv pair and return stereo-stripped InChI strings
    for molecules that belong to any of the requested split names.

    Parameters
    ----------
    labels_path : path to labels.tsv  (columns must include 'spec' and 'smiles')
    split_path  : path to split.tsv   (columns must include 'name' and 'split')
    splits      : split names to exclude (default: val and test)

    Used for MassSpecGym where only the val/test molecules must be excluded
    (training molecules may be used for augmentation).
    """
    labels_df = pd.read_csv(labels_path, sep="\t")
    split_df = pd.read_csv(split_path, sep="\t")

    holdout_specs = set(split_df.loc[split_df["split"].isin(splits), "name"])
    holdout_smiles = labels_df.loc[
        labels_df["spec"].isin(holdout_specs), "smiles"
    ].dropna()
    return _smiles_to_inchi_set(holdout_smiles)


def main(args: argparse.Namespace) -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # ── Build exclusion set ───────────────────────────────────────────────────
    print(f"\n[1/5] Building exclusion set …")

    # ATMOMACCS: exclude all molecules (entire dataset is used for evaluation)
    print(f"    ATMOMACCS labels: {args.exclude}")
    exclusion_set = _build_exclusion_set(args.exclude)
    print(f"    → {len(exclusion_set):,} molecules from ATMOMACCS")

    # MassSpecGym: exclude only val/test splits
    if args.msg_labels and args.msg_split:
        print(f"    MassSpecGym labels: {args.msg_labels}")
        print(f"    MassSpecGym split : {args.msg_split}")
        msg_excl = _build_split_exclusion_set(
            args.msg_labels, args.msg_split, splits=("val", "test")
        )
        print(f"    → {len(msg_excl):,} molecules from MassSpecGym val/test")
        exclusion_set |= msg_excl

    print(f"    Total exclusion set: {len(exclusion_set):,} unique molecules")

    # ── Load input dataset ────────────────────────────────────────────────────
    print(f"\n[2/5] Loading dataset from: {args.input}")
    df = pd.read_csv(args.input)

    # Support both 'inchi' and 'smiles' columns
    if "inchi" in df.columns:
        records = df["inchi"].dropna().tolist()
        worker_fn = _process_inchi
        input_label = "InChI"
    elif "smiles" in df.columns:
        records = df["smiles"].dropna().tolist()
        worker_fn = _process_smiles
        input_label = "SMILES"
    else:
        raise ValueError("Input CSV must have an 'inchi' or 'smiles' column.")

    print(f"    {len(records):,} {input_label} records loaded")

    # ── Atmospheric pre-filter ────────────────────────────────────────────────
    print(f"\n[3/5] Applying atmospheric filter (n_jobs={args.n_jobs}) …")
    filter_results = Parallel(n_jobs=args.n_jobs)(
        delayed(worker_fn)(rec) for rec in tqdm(records, desc="  Filtering", leave=True)
    )

    passed_raw = [r for r in filter_results if r is not None]

    # Deduplicate on InChI: more robust than canonical SMILES — language-independent,
    # consistent normalisation, handles tautomers.  Stripping stereochemistry before
    # InChI generation (done in _filter_mol) collapses stereo variants to one entry.
    seen: dict[str, tuple[str, float]] = {}  # inchi → (smi, mw)
    for smi, mw, inchi in passed_raw:
        if inchi not in seen:
            seen[inchi] = (smi, mw)
    # passed carries (smi, mw, inchi) so the InChI is available downstream
    passed = [(smi, mw, inchi) for inchi, (smi, mw) in seen.items()]

    print(
        f"    Passed filter : {len(passed_raw):,} / {len(records):,} "
        f"({100 * len(passed_raw) / max(len(records), 1):.1f}%)"
    )
    print(f"    After InChI dedup (stereoisomers): {len(passed):,}")

    # ── Exclude ATMOMACCS molecules ───────────────────────────────────────────
    passed = [
        (smi, mw, inchi) for smi, mw, inchi in passed if inchi not in exclusion_set
    ]
    print(f"    After ATMOMACCS exclusion: {len(passed):,}")

    # ── MW stratification ─────────────────────────────────────────────────────
    print(f"\n[4/5] MW stratification into {len(MW_BINS)} bins …")
    # bins map index → list of (smi, inchi) pairs; SMILES needed for ATMOMACCS,
    # InChI carried through so the final output requires no recomputation.
    bins: dict[int, list[tuple[str, str]]] = {i: [] for i in range(len(MW_BINS))}
    for smi, mw, inchi in passed:
        for i, (lo, hi) in enumerate(MW_BINS):
            if lo <= mw < hi:
                bins[i].append((smi, inchi))
                break

    for i, (lo, hi) in enumerate(MW_BINS):
        hi_label = f"{hi - 1}" if hi == 1001 else str(hi)
        print(f"    [{lo}–{hi_label}] Da : {len(bins[i]):,} molecules")

    # ── Per-bin ATMOMACCS clustering + sampling ───────────────────────────────
    print(f"\n[5/5] Per-bin clustering & sampling (target {TARGET_PER_BIN:,}/bin) …")
    all_sampled: list[tuple[str, str]] = []  # (smiles, inchi) pairs

    for i, (lo, hi) in enumerate(MW_BINS):
        hi_label = f"{hi - 1}" if hi == 1001 else str(hi)
        label = f"[{lo}–{hi_label}]"
        print(f"\n  Bin {label}: {len(bins[i]):,} → target {TARGET_PER_BIN:,}")

        sampled = _sample_bin(
            bin_pairs=bins[i],
            target_n=TARGET_PER_BIN,
            n_clusters=N_CLUSTERS,
            seed=RANDOM_SEED + i,
            n_jobs=args.n_jobs,
            bin_label=label,
        )
        print(f"    → sampled {len(sampled):,}")
        all_sampled.extend(sampled)

    # ── Final trim ────────────────────────────────────────────────────────────
    # No deduplication needed: InChI dedup was applied immediately after filtering,
    # and bins are disjoint by MW — no molecule can appear in two bins.
    print(f"\nTotal sampled: {len(all_sampled):,}")

    if len(all_sampled) > TARGET_TOTAL:
        rng = np.random.default_rng(RANDOM_SEED)
        idx = rng.choice(len(all_sampled), size=TARGET_TOTAL, replace=False)
        all_sampled = [all_sampled[i] for i in sorted(idx)]
        print(f"Trimmed to {len(all_sampled):,}")

    # ── Save ──────────────────────────────────────────────────────────────────
    # InChI was computed once in _filter_mol and carried through — no second pass.
    print(f"\nSaving to: {args.output}")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    out_df = pd.DataFrame(all_sampled, columns=["smiles", "inchi"])
    out_df.to_csv(args.output, index=False)

    print(f"\n✓ Done. Saved {len(out_df):,} molecules to {args.output}")

    # ── Summary statistics ────────────────────────────────────────────────────
    if len(out_df) > 0:
        mws = [
            Descriptors.MolWt(Chem.MolFromSmiles(s))
            for s in out_df["smiles"]
            if Chem.MolFromSmiles(s) is not None
        ]
        print(f"\nSummary:")
        print(
            f"  MW  — mean: {np.mean(mws):.1f}  "
            f"median: {np.median(mws):.1f}  "
            f"min: {np.min(mws):.1f}  max: {np.max(mws):.1f}"
        )
        # Per-bin counts in output
        for i, (lo, hi) in enumerate(MW_BINS):
            hi_label = f"{hi - 1}" if hi == 1001 else str(hi)
            count = sum(1 for mw in mws if lo <= mw < hi)
            print(f"  [{lo}–{hi_label}] Da : {count:,}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sample ~300K atmospheric molecules from fp2mol for NEIMS augmentation"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="../data/fp2mol/combined/preprocessed/combined_train.csv",
        help="Path to combined fp2mol CSV (must have 'inchi' or 'smiles' column)",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        default="../data/atmomaccs/labels.tsv",
        help="Path to ATMOMACCS labels.tsv — ALL molecules here are excluded",
    )
    parser.add_argument(
        "--msg-labels",
        type=str,
        default="../data/msg/labels.tsv",
        help="Path to MassSpecGym labels.tsv — only val/test molecules are excluded",
    )
    parser.add_argument(
        "--msg-split",
        type=str,
        default="../data/msg/split.tsv",
        help="Path to MassSpecGym split.tsv (columns: name, split)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="../data/fp2mol/neims_augment/sampled_300k.csv",
        help="Output CSV path (columns: smiles, inchi)",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=12,
        help="Number of parallel workers (default: 4; -1 = all CPUs)",
    )
    args = parser.parse_args()
    main(args)
