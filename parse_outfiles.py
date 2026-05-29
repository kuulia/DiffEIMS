"""
Parse SLURM .out files and extract config + test metrics into a CSV.

Usage:
    python parse_outfiles.py outfiles/17947206_0.out outfiles/17942563_0.out
    python parse_outfiles.py outfiles/*.out
    python parse_outfiles.py outfiles/*.out -o results.csv
"""

import argparse
import ast
import csv
import os
import re
import sys


# Config fields to extract (section, key) -> csv column name
CONFIG_FIELDS = [
    ("general", "name",                       "run_name"),
    ("general", "encoder",                    "encoder_path"),
    ("general", "decoder",                    "decoder_path"),
    ("general", "test_only",                  "test_only_ckpt"),
    ("general", "num_test_samples",           "num_test_samples"),
    ("dataset",  "name",                       "dataset"),
    ("dataset",  "split_file",                "split_file"),
    ("dataset",  "spec_features",             "spec_features"),
    ("dataset",  "mol_features",              "mol_features"),
    ("dataset",  "merge",                     "merge"),
    ("model",   "model",                      "model_type"),
    ("model",   "diffusion_steps",            "diffusion_steps"),
    ("model",   "n_layers",                   "n_layers"),
    ("train",   "n_epochs",                   "n_epochs"),
    ("train",   "batch_size",                 "batch_size"),
    ("train",   "eval_batch_size",            "eval_batch_size"),
    ("train",   "lr",                         "lr"),
    ("train",   "seed",                       "seed"),
]

# Test metrics to collect (as they appear after "test/")
TEST_METRICS = [
    "NLL", "X_KL", "E_KL", "E_CE", "validity",
    "acc_at_1",  "acc_at_5",  "acc_at_10",
    "tanimoto_at_1", "tanimoto_at_5", "tanimoto_at_10",
]


def _basename_without_ext(path):
    return os.path.splitext(os.path.basename(path))[0]


def _shorten_path(p):
    """Keep only the last two path components for readability."""
    if not p:
        return p
    parts = p.rstrip("/").split("/")
    return "/".join(parts[-2:]) if len(parts) >= 2 else p


def parse_outfile(path):
    row = {"job_file": _basename_without_ext(path)}

    config_found = False
    test_metrics = {}
    total_runtime = None

    with open(path, "r", errors="replace") as f:
        for line in f:
            line = line.rstrip()

            # --- Config line (line 3 pattern) ---
            if not config_found and "Read config" not in line:
                m = re.search(r"\[root\]\[INFO\] - (\{.+\})$", line)
                if m:
                    try:
                        cfg = ast.literal_eval(m.group(1))
                        for section, key, col in CONFIG_FIELDS:
                            val = cfg.get(section, {}).get(key, "")
                            if isinstance(val, str) and "/" in val:
                                val = _shorten_path(val)
                            row[col] = val
                        config_found = True
                    except (ValueError, SyntaxError):
                        pass

            # --- Test metric lines ---
            m = re.match(r"\s+test/(\S+)\s+([\d.eE+\-]+|nan)\s*$", line)
            if m:
                metric_name = m.group(1)
                value = m.group(2)
                test_metrics[metric_name] = value

            # --- Total runtime ---
            m = re.match(r"Total runtime:\s*(\d+)\s*seconds", line)
            if m:
                total_runtime = int(m.group(1))

    for metric in TEST_METRICS:
        row[f"test/{metric}"] = test_metrics.get(metric, "")

    row["total_runtime_s"] = total_runtime if total_runtime is not None else ""
    return row


def main():
    parser = argparse.ArgumentParser(description="Parse .out files into a CSV.")
    parser.add_argument("files", nargs="+", help=".out files to parse")
    parser.add_argument("-o", "--output", default="results.csv",
                        help="Output CSV path (default: results.csv)")
    args = parser.parse_args()

    rows = []
    for path in args.files:
        if not os.path.isfile(path):
            print(f"Warning: {path} not found, skipping.", file=sys.stderr)
            continue
        row = parse_outfile(path)
        rows.append(row)
        print(f"Parsed: {path}")

    if not rows:
        print("No files parsed.", file=sys.stderr)
        sys.exit(1)

    # Build column order: metadata, config fields, test metrics, runtime
    all_columns = (
        ["job_file"]
        + [col for _, _, col in CONFIG_FIELDS]
        + [f"test/{m}" for m in TEST_METRICS]
        + ["total_runtime_s"]
    )
    # Add any unexpected columns at the end
    extra = [k for k in rows[0] if k not in all_columns]
    all_columns += extra

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} row(s) to {args.output}")


if __name__ == "__main__":
    main()
