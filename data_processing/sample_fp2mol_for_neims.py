"""
sample_fp2mol_for_neims.py
==========================
Sample ~300 000 atmospherically-relevant, chemically-diverse molecules from
the fp2mol combined dataset for NEIMS EI-MS spectrum simulation.

Strategy
--------
1. Atmospheric pre-filter
     MW --min-mw (default 16)–1500 Da  |  atoms ∈ {C,H,N,O,S,F,Cl,P,B,Br,I,Si}  |  1–40 C  |  no fragment
     ≥1 oxygen only with --require-oxygen
     (formal charges are permitted: nitro/nitrate groups carry [N+]/[O-] in RDKit)
2. ATMOMACCS-V5 clustering of the whole pool (integer-count fingerprint, 205 features)
     MiniBatch k-means, --n-clusters in total (default 15 000).  To keep k-means
     tractable on ~3M molecules the pool is split into MW bins ([--min-mw, 100) when
     --min-mw < 100, then 100-Da bins up to 1500) and each bin is clustered on its
     own, with clusters in proportion to its size.  The bins set no quotas.
3. Size-matched sample spread over the clusters
     --n molecules (default 286 252, the size of neims-augment-random), as evenly as
     possible over all clusters of all bins: clusters smaller than the common cap
     are taken whole and the rest share the remaining budget equally
     (water-filling); within a cluster the pick is random.  This maximises how many
     regions of the pool's chemical space are represented, at a fixed size.

mixed_augment was sampled with --require-oxygen --min-mw 100. That left it with
no O-free molecules (so no CH or CHN compounds) and none below 100 Da, while about
20% of the atmomaccs test set is O-free and 9% is below 100 Da. It also gave every
MW bin an equal quota, which put about a third of it above 500 Da. The defaults
now drop both filters (the 16 Da floor matches build_fp2mol_datasets.py) and the
quotas; the two flags bring back mixed_augment's filter, not its sampling.

Input
-----
A CSV with an 'inchi' or 'smiles' column, or a NEIMS new-format SDF such as the
augment-full spectra of the whole pool (3.3M molecules). From an SDF the pool is
the mol-name line of every record that has a spectrum, i.e. the SMILES NEIMS was
given. The output keeps that string in its 'input' column, so neims-pipeline can
select the sampled records from the same SDF without simulating anything again.

Deduplication & exclusion
--------------------------
Deduplication uses InChI, computed from the stereochemistry-stripped mol
(isomericSmiles=False round-trip), so stereo variants collapse to one entry.

The leakage guard is the one build_fp2mol_datasets.py applies: every atmomaccs and
atmomaccs_tms molecule (all splits) and every annotated Yee structure is excluded
by InChIKey connectivity block, which is blind to stereo, isotopes and protonation.
The written output is checked against the guard again, on both the canonical
SMILES and the input string, and the script fails if anything slipped through.

Usage (from data_processing/, where the default paths resolve)
--------------------------------------------------------------
python sample_fp2mol_for_neims.py \\
    --input  <neims-pipeline>/data/neims/augment_full/NEIMS/spectra.sdf \\
    --output <neims-pipeline>/data/neims/mixed_augment_full_sample/sampled.csv \\
    --n_jobs 8
"""

import argparse
import os
import random
import sys
import warnings
from pathlib import Path
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

from build_fp2mol_datasets import (  # the fp2mol datasets' leakage guard
    LEAKAGE_GUARD_DATASETS,
    YEE_REFERENCE_CSV,
    _build_registry,
    _connectivity_key,
    load_yee_reference_keys,
)

RDLogger.DisableLog("rdApp.*")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Element allowlist: union of ATMOMACCS benchmark datasets (Table I), excluding
# metals/salts (e.g. Na) since we require covalent, uncharged molecules.
ATMO_ATOMS = {"C", "N", "S", "O", "F", "Cl", "H", "P", "B", "Br", "I", "Si"}

# 14 MW bins covering 100–1500 Da; _mw_bins() adds one below 100 Da when --min-mw < 100.
# They only split the pool so k-means runs per bin, not on all ~3M molecules at once:
# each bin gets clusters in proportion to its size, and no bin has a quota.
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

# Default sample size: the molecules with a spectrum in neims-augment-random (286 252
# of its 287 483; the other 1 231 are net-charged), so both sets enter preprocessing
# at the same size. Every molecule sampled here has a spectrum.
DEFAULT_N = 286_252
# k-means clusters over the whole pool, about the 14 x 1000 of the old per-bin scheme
DEFAULT_N_CLUSTERS = 15_000
RANDOM_SEED = 42

# Same lower MW bound as _passes_filter in build_fp2mol_datasets.py
DEFAULT_MIN_MW = 16.0


def _mw_bins(min_mw: float) -> list[tuple[float, float]]:
    """MW_BINS, preceded by a [min_mw, 100) bin when min_mw is below 100 Da."""
    lo = MW_BINS[0][0]
    if min_mw < lo:
        return [(min_mw, lo - 0.01)] + MW_BINS
    return MW_BINS


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 – Atmospheric pre-filter
# ─────────────────────────────────────────────────────────────────────────────


def _filter_mol(
    mol, min_mw: float, require_oxygen: bool
) -> Optional[tuple[str, float, str]]:
    """
    Apply atmospheric relevance rules to an RDKit mol.

    Rules
    -----
    * MW:    min_mw ≤ MW ≤ 1500 Da
    * Atoms: only {C, H, N, O, S, F, Cl, P, B, Br, I, Si}
    * O:     at least one oxygen atom, only if require_oxygen
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
        if not (min_mw <= mw <= 1500.0):
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

        if num_C < 1 or (require_oxygen and num_O < 1):
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


def _process_inchi(inchi: str, min_mw: float, require_oxygen: bool) -> Optional[tuple]:
    """Worker: InChI → atmospheric-filtered (smiles, mw, inchi) or None."""
    RDLogger.DisableLog("rdApp.*")
    try:
        mol = Chem.MolFromInchi(inchi)
        if mol is None:
            return None
        return _filter_mol(mol, min_mw, require_oxygen)
    except Exception:
        return None


def _process_smiles(smi: str, min_mw: float, require_oxygen: bool) -> Optional[tuple]:
    """Worker: SMILES → atmospheric-filtered (canonical_smiles, mw, inchi) or None."""
    RDLogger.DisableLog("rdApp.*")
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        return _filter_mol(mol, min_mw, require_oxygen)
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
# Step 3 – Per-bin clustering, then one size-matched allocation over all clusters
# ─────────────────────────────────────────────────────────────────────────────


def _cluster_bin(
    bin_smiles: list[str],
    n_clusters: int,
    seed: int,
    n_jobs: int,
    bin_label: str,
) -> np.ndarray:
    """
    ATMOMACCS-V5 MiniBatch k-means labels for the molecules of one MW bin.

    A molecule whose fingerprint fails gets label -1 and is never sampled. When the
    bin has no more molecules than n_clusters, each molecule is its own cluster.
    """
    n = len(bin_smiles)
    labels = np.full(n, -1, dtype=np.int64)

    print(f"    {bin_label}: computing ATMOMACCS for {n:,} molecules …")
    fps_raw = Parallel(n_jobs=n_jobs)(
        delayed(_compute_atmomaccs)(smi)
        for smi in tqdm(bin_smiles, desc=f"      {bin_label} descriptors", leave=False)
    )
    ok = np.array([fp is not None for fp in fps_raw])
    if not ok.any():
        warnings.warn(f"{bin_label}: no valid ATMOMACCS fingerprints — skipping bin")
        return labels

    X = np.vstack([fp for fp in fps_raw if fp is not None])  # shape (n_valid, 205)
    n_valid = len(X)
    k = min(n_clusters, n_valid)
    if k == n_valid:
        labels[ok] = np.arange(n_valid)
        return labels

    print(f"    {bin_label}: k-means k={k:,} on {n_valid:,} molecules …")
    kmeans = MiniBatchKMeans(
        n_clusters=k,
        random_state=seed,
        batch_size=min(10_000, n_valid),
        n_init=3,
        verbose=0,
    )
    labels[ok] = kmeans.fit_predict(X)
    return labels


def _allocate(sizes: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """
    How many molecules to take from each cluster: n in total, as even as possible.

    Water-filling: every cluster gets min(size, q) for the largest common cap q that
    keeps the total at or below n, and the remainder goes one each to randomly chosen
    clusters that still have molecules left. Clusters smaller than the cap are taken
    whole, so no region of chemical space is crowded out by a large one.
    """
    if n >= sizes.sum():
        return sizes.copy()
    lo, hi = 0, int(sizes.max())
    while lo < hi:  # largest q with sum(min(sizes, q)) <= n
        mid = (lo + hi + 1) // 2
        if np.minimum(sizes, mid).sum() <= n:
            lo = mid
        else:
            hi = mid - 1
    take = np.minimum(sizes, lo)
    rest = n - int(take.sum())  # < number of clusters larger than lo, by choice of lo
    take[rng.choice(np.flatnonzero(sizes > lo), size=rest, replace=False)] += 1
    return take


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def _guard_keys(data_dir: str, yee_reference: str) -> set[str]:
    """
    Connectivity keys of every atmomaccs / atmomaccs_tms molecule (all splits) and
    every annotated Yee structure. Raises if a source cannot be read, so a missing
    file cannot silently shrink the guard.
    """
    spectral, _, _ = _build_registry(Path(data_dir), Path(data_dir) / "fp2mol")
    by_name = {ds.name: ds for ds in spectral}
    keys: set[str] = set()
    for name in LEAKAGE_GUARD_DATASETS:
        collected = by_name[name].get_exclusions(splits=None)
        ds_keys = {_connectivity_key(i) for i in collected}
        print(f"    {name}: {len(ds_keys):,} skeletons")
        keys |= ds_keys
    yee_keys = load_yee_reference_keys(yee_reference)
    print(f"    yee annotations ({yee_reference}): {len(yee_keys):,} skeletons")
    keys |= yee_keys
    keys.discard(None)
    print(f"    Total: {len(keys):,} skeletons held out")
    return keys


def _smiles_key(smi: str) -> Optional[str]:
    """Connectivity key of a SMILES, or None if it does not convert."""
    try:
        return _connectivity_key(Chem.MolToInchi(Chem.MolFromSmiles(smi)))
    except Exception:
        return None


def _read_sdf_pool(sdf_path: str) -> tuple[list[str], int]:
    """
    Mol-name line of every record in a NEIMS new-format SDF that has a spectrum.

    Streams the file, so a multi-GB SDF is fine. Records are split the way
    neims_output_to_mgf.py splits them: on $$$$, first non-empty line = SMILES.
    Returns (names, number of records without a spectrum).
    """
    names: list[str] = []
    name, has_spectrum, n_no_spectrum = None, False, 0
    with open(sdf_path, "r", errors="replace") as fh:
        for line in fh:
            if line.startswith("$$$$"):
                if name is not None:
                    if has_spectrum:
                        names.append(name)
                    else:
                        n_no_spectrum += 1
                name, has_spectrum = None, False
            elif name is None:
                name = line.strip() or None
            elif "predicted spectrum" in line.lower():
                has_spectrum = True
    if name is not None:  # last record without a trailing $$$$
        if has_spectrum:
            names.append(name)
        else:
            n_no_spectrum += 1
    return names, n_no_spectrum


def main(args: argparse.Namespace) -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # ── Leakage guard ─────────────────────────────────────────────────────────
    print(f"\n[1/5] Building leakage guard (atmomaccs, atmomaccs_tms, Yee) …")
    yee_reference = args.yee_reference or str(
        Path(args.data_dir).parent / YEE_REFERENCE_CSV
    )
    guard_keys = _guard_keys(args.data_dir, yee_reference)

    # ── Load input dataset ────────────────────────────────────────────────────
    print(f"\n[2/5] Loading dataset from: {args.input}")
    if args.input.endswith(".sdf"):
        records, n_no_spectrum = _read_sdf_pool(args.input)
        print(f"    Skipped {n_no_spectrum:,} SDF records without a spectrum")
        worker_fn = _process_smiles
        input_label = "SDF SMILES"
    else:
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
    print(
        f"\n[3/5] Applying atmospheric filter (n_jobs={args.n_jobs}, "
        f"min_mw={args.min_mw:g}, require_oxygen={args.require_oxygen}) …"
    )
    filter_results = Parallel(n_jobs=args.n_jobs)(
        delayed(worker_fn)(rec, args.min_mw, args.require_oxygen)
        for rec in tqdm(records, desc="  Filtering", leave=True)
    )

    # Keep the input record next to its filter result: for an SDF it is the
    # exact SMILES NEIMS simulated, which neims-pipeline selects records by.
    passed_raw = [(r, rec) for r, rec in zip(filter_results, records) if r is not None]

    # Deduplicate on InChI: more robust than canonical SMILES — language-independent,
    # consistent normalisation, handles tautomers.  Stripping stereochemistry before
    # InChI generation (done in _filter_mol) collapses stereo variants to one entry.
    seen: dict[str, tuple[str, float]] = {}  # inchi → (smi, mw)
    source: dict[str, str] = {}  # inchi → input record
    for (smi, mw, inchi), rec in passed_raw:
        if inchi not in seen:
            seen[inchi] = (smi, mw)
            source[inchi] = rec
    # passed carries (smi, mw, inchi) so the InChI is available downstream
    passed = [(smi, mw, inchi) for inchi, (smi, mw) in seen.items()]

    print(
        f"    Passed filter : {len(passed_raw):,} / {len(records):,} "
        f"({100 * len(passed_raw) / max(len(records), 1):.1f}%)"
    )
    print(f"    After InChI dedup (stereoisomers): {len(passed):,}")

    # ── Leakage guard ─────────────────────────────────────────────────────────
    passed = [
        (smi, mw, inchi)
        for smi, mw, inchi in passed
        if _connectivity_key(inchi) not in guard_keys
    ]
    print(f"    After leakage guard: {len(passed):,}")

    if args.n > len(passed):
        raise RuntimeError(
            f"Only {len(passed):,} molecules left after filtering, fewer than the "
            f"{args.n:,} needed for a size-matched sample."
        )

    # ── MW binning ────────────────────────────────────────────────────────────
    mw_bins = _mw_bins(args.min_mw)
    print(f"\n[4/5] Splitting into {len(mw_bins)} MW bins for clustering …")
    # bins map index → list of (smi, inchi) pairs; SMILES needed for ATMOMACCS,
    # InChI carried through so the final output requires no recomputation.
    bins: dict[int, list[tuple[str, str]]] = {i: [] for i in range(len(mw_bins))}
    for smi, mw, inchi in passed:
        for i, (lo, hi) in enumerate(mw_bins):
            if lo <= mw < hi:
                bins[i].append((smi, inchi))
                break

    for i, (lo, hi) in enumerate(mw_bins):
        hi_label = f"{hi - 1}" if hi == 1001 else str(hi)
        print(f"    [{lo}–{hi_label}] Da : {len(bins[i]):,} molecules")

    # ── Per-bin ATMOMACCS clustering ──────────────────────────────────────────
    print(
        f"\n[5/5] Clustering into ~{args.n_clusters:,} clusters, "
        f"then sampling {args.n:,} molecules over all of them …"
    )
    pairs: list[tuple[str, str]] = []  # (smiles, inchi) of every clustered molecule
    clusters: list[np.ndarray] = []  # indices into pairs, one array per cluster
    for i, (lo, hi) in enumerate(mw_bins):
        if not bins[i]:
            continue
        hi_label = f"{hi - 1}" if hi == 1001 else str(hi)
        label = f"[{lo}–{hi_label}]"
        # Clusters in proportion to bin size, so clusters are of similar size in every bin
        k = max(1, round(args.n_clusters * len(bins[i]) / len(passed)))
        labels = _cluster_bin(
            [smi for smi, _ in bins[i]], k, RANDOM_SEED + i, args.n_jobs, label
        )
        # Group the bin's indices by label (sorting, not one scan per cluster)
        order = np.argsort(labels, kind="stable")
        order = order[labels[order] >= 0]
        offset = len(pairs)
        pairs.extend(bins[i])
        for idx in np.split(order, np.flatnonzero(np.diff(labels[order])) + 1):
            clusters.append(idx + offset)

    # ── Size-matched sample, spread evenly over all clusters ──────────────────
    # No deduplication needed: InChI dedup was applied immediately after filtering,
    # and bins are disjoint by MW — no molecule can appear in two bins.
    rng = np.random.default_rng(RANDOM_SEED)
    sizes = np.array([len(c) for c in clusters])
    take = _allocate(sizes, args.n, rng)
    all_sampled: list[tuple[str, str]] = [
        pairs[j]
        for idx, t in zip(clusters, take)
        if t
        for j in rng.choice(idx, size=t, replace=False)
    ]
    print(
        f"\n    {len(clusters):,} clusters (median size {np.median(sizes):.0f}, "
        f"largest {sizes.max():,}); at most {take.max():,} taken from any cluster, "
        f"{int((take == sizes).sum()):,} clusters taken whole"
    )
    print(f"\nTotal sampled: {len(all_sampled):,}")

    # ── Verify ────────────────────────────────────────────────────────────────
    # InChI was computed once in _filter_mol and carried through — no second pass.
    out_df = pd.DataFrame(all_sampled, columns=["smiles", "inchi"])
    out_df["input"] = [source[i] for i in out_df["inchi"]]

    # Re-derive the keys from what is written, as any later leakage check will.
    # An input that is an InChI is checked through its canonical SMILES instead.
    to_check = ["smiles"] if input_label == "InChI" else ["smiles", "input"]
    for col in to_check:
        leaked = [s for s in out_df[col] if _smiles_key(s) in guard_keys]
        if leaked:
            raise RuntimeError(
                f"{len(leaked)} sampled molecules ('{col}' column) match the "
                f"leakage guard, e.g. {leaked[:3]}. Nothing written."
            )
    print(f"\n[VERIFY] No atmomaccs, atmomaccs_tms or annotated Yee molecule sampled.")

    # ── Save ──────────────────────────────────────────────────────────────────
    print(f"\nSaving to: {args.output}")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    out_df.to_csv(args.output, index=False)

    print(f"\n✓ Done. Saved {len(out_df):,} molecules to {args.output}")

    # ── Summary statistics ────────────────────────────────────────────────────
    if len(out_df) > 0:
        mols = [m for m in map(Chem.MolFromSmiles, out_df["smiles"]) if m is not None]
        mws = [Descriptors.MolWt(m) for m in mols]
        n_no_o = sum(all(a.GetSymbol() != "O" for a in m.GetAtoms()) for m in mols)
        print(f"\nSummary:")
        print(
            f"  MW  — mean: {np.mean(mws):.1f}  "
            f"median: {np.median(mws):.1f}  "
            f"min: {np.min(mws):.1f}  max: {np.max(mws):.1f}"
        )
        print(f"  O-free: {n_no_o:,} ({100 * n_no_o / len(mols):.1f}%)")
        # Per-bin counts in output
        for i, (lo, hi) in enumerate(mw_bins):
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
        help="Pool: CSV with an 'inchi' or 'smiles' column, or a NEIMS new-format .sdf",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="../data",
        help="DiffMS data/ tree holding the atmomaccs datasets the guard reads",
    )
    parser.add_argument(
        "--yee-reference",
        type=str,
        default=None,
        help=f"Annotated Yee structures (default: <data-dir>/../{YEE_REFERENCE_CSV})",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=DEFAULT_N,
        help="Sample size (default: neims-augment-random's molecules with a spectrum)",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=DEFAULT_N_CLUSTERS,
        help="k-means clusters over the whole pool, split across MW bins by bin size",
    )
    parser.add_argument(
        "--min-mw",
        type=float,
        default=DEFAULT_MIN_MW,
        help="Lower MW bound in Da (mixed_augment used 100); below 100 adds a low-MW bin",
    )
    parser.add_argument(
        "--require-oxygen",
        action="store_true",
        help="Keep only molecules with at least one O, as mixed_augment did",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="../data/fp2mol/neims_augment/sampled_with_o_free.csv",
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
