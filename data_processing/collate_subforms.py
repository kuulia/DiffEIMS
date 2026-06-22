"""
collate_subforms.py

Consolidates a folder of per-spectrum subformula JSON files (as produced by MIST /
SIRIUS subformula annotation) into a single pickle file.  This avoids the "lots of
small file I/O" bottleneck on HPC cluster file-systems.

The resulting pickle is a plain Python dict:
    { spec_name (str) : tree (dict) }
where ``tree`` is the raw parsed JSON content:
    {
        "cand_form":  <str>,
        "cand_ion":   <str>,
        "output_tbl": <dict or None>,
    }

Usage
-----
    python data_processing/collate_subforms.py \\
        --subform_folder data/paired_spectra/canopus/subformulae \\
        --output         data/paired_spectra/canopus/subforms.pkl \\
        --num_workers    8

The output file is written to ``--output``.  If ``--output`` is omitted it
defaults to ``<subform_folder>/subforms.pkl``.
"""

import argparse
import json
import logging
import pickle
from pathlib import Path
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def _load_one(json_path: Path) -> tuple:
    """Load a single subform JSON file and return (spec_name, tree_dict)."""
    try:
        with open(json_path, "r") as f:
            tree = json.load(f)
        return (json_path.stem, tree)
    except Exception as exc:
        logging.warning(f"Failed to load {json_path}: {exc}")
        return (json_path.stem, None)


def collate(subform_folder: str, output: str = None, num_workers: int = 1) -> Path:
    """Read all *.json files in *subform_folder* and pickle them as a single dict.

    Args:
        subform_folder: Directory containing ``<spec_name>.json`` files.
        output:         Destination pickle path.  Defaults to
                        ``<subform_folder>/subforms.pkl``.
        num_workers:    Number of parallel reader processes.

    Returns:
        Path to the written pickle file.
    """
    subform_folder = Path(subform_folder)
    if not subform_folder.exists():
        raise FileNotFoundError(f"subform_folder does not exist: {subform_folder}")

    json_files = sorted(subform_folder.glob("*.json"))
    if not json_files:
        raise ValueError(f"No *.json files found in {subform_folder}")

    logging.info(f"Found {len(json_files):,} JSON files in {subform_folder}")

    if num_workers > 1:
        with Pool(num_workers) as pool:
            results = list(
                tqdm(
                    pool.imap(_load_one, json_files, chunksize=256),
                    total=len(json_files),
                    desc="Loading subforms",
                )
            )
    else:
        results = [_load_one(f) for f in tqdm(json_files, desc="Loading subforms")]

    # Build dict, skipping any entries that failed to load
    subform_dict = {name: tree for name, tree in results if tree is not None}
    n_failed = len(results) - len(subform_dict)
    if n_failed:
        logging.warning(f"{n_failed} files could not be loaded and were skipped.")

    logging.info(f"Collated {len(subform_dict):,} subform entries.")

    if output is None:
        output = subform_folder / "subforms.pkl"
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "wb") as f:
        pickle.dump(subform_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    logging.info(f"Wrote pickle to {output}  ({output.stat().st_size / 1e6:.1f} MB)")
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Collate per-spectrum subformula JSON files into one pickle."
    )
    parser.add_argument(
        "--subform_folder",
        required=True,
        help="Directory containing <spec_name>.json subformula files.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output pickle path (default: <subform_folder>/subforms.pkl).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of parallel reader processes (default: 1).",
    )
    args = parser.parse_args()
    collate(args.subform_folder, args.output, args.num_workers)


if __name__ == "__main__":
    main()
