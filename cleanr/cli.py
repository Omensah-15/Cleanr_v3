#!/usr/bin/env python3
"""
CleanR v3 — Production Data Cleaning Engine
Usage: cleanr <input> [output] [options]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

from cleanr import VERSION
from cleanr.engine import CleanREngine


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="cleanr",
        description=f"CleanR v{VERSION} — Production Data Cleaning Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  cleanr data.csv
  cleanr messy.xlsx clean.csv
  cleanr data.csv --drop-na --detect-outliers --outlier-method remove
  cleanr data.csv --impute-strategy knn --drop-col-threshold 0.8
  cleanr data.csv --keep id,name,email --rename fname=first_name
  cleanr data.csv --split full_name "first_name,last_name" " "
  cleanr data.csv --output-format parquet --no-fingerprint --quiet
  cleanr large.csv --chunk 100000 --quick
""",
    )

    # Positional
    parser.add_argument("input",  help="Input file (CSV/TSV/JSON/JSONL/XLSX/Parquet)")
    parser.add_argument("output", nargs="?", help="Output path (default: <input>_clean.<ext>)")

    # ── Cleaning ──────────────────────────────────────────────────────────────
    g = parser.add_argument_group("Cleaning Pipeline")
    g.add_argument("--normalize",    dest="normalize",    action="store_true", default=True)
    g.add_argument("--no-normalize", dest="normalize",    action="store_false")
    g.add_argument("--trim",         dest="trim",         action="store_true", default=True)
    g.add_argument("--no-trim",      dest="trim",         action="store_false")
    g.add_argument("--dedup",        dest="dedup",        action="store_true", default=True)
    g.add_argument("--no-dedup",     dest="dedup",        action="store_false")
    g.add_argument("--drop-constant",    dest="drop_constant",    action="store_true", default=True)
    g.add_argument("--no-drop-constant", dest="drop_constant",    action="store_false")
    g.add_argument("--auto-types",   dest="auto_types",   action="store_true", default=True)
    g.add_argument("--no-auto-types",dest="auto_types",   action="store_false")
    g.add_argument("--validate-formats",    dest="validate_formats", action="store_true", default=True)
    g.add_argument("--no-validate-formats", dest="validate_formats", action="store_false")

    # ── Outliers ──────────────────────────────────────────────────────────────
    og = parser.add_argument_group("Outlier Detection")
    og.add_argument("--detect-outliers", action="store_true",
                    help="Enable Isolation Forest + IQR consensus outlier detection")
    og.add_argument("--outlier-method", choices=["flag", "remove"], default="flag",
                    help="Action for detected outliers (default: flag)")
    og.add_argument("--outlier-contamination", type=float, default=0.02, metavar="0.0-0.5",
                    help="Expected outlier fraction for Isolation Forest (default: 0.02)")

    # ── Missing values ────────────────────────────────────────────────────────
    mg = parser.add_argument_group("Missing Values")
    mg.add_argument("--impute-strategy",
                    choices=["auto", "knn", "median", "mean", "mode", "constant"],
                    default="auto",
                    help="Imputation strategy (default: auto — KNN where beneficial)")
    mg.add_argument("--fill", metavar="VALUE",
                    help="Constant fill value (used with --impute-strategy constant)")
    mg.add_argument("--drop-na", action="store_true",
                    help="Drop rows containing any missing value (overrides imputation)")
    mg.add_argument("--drop-col-threshold", type=float, metavar="FRAC",
                    help="Drop columns with >= FRAC missing (e.g. 0.8 = 80%%)")

    # ── Column ops ────────────────────────────────────────────────────────────
    cg = parser.add_argument_group("Column Operations")
    cg.add_argument("--keep", metavar="COL1,COL2",
                    help="Keep only these columns (comma-separated)")
    cg.add_argument("--drop", metavar="COL1,COL2",
                    help="Drop these columns (comma-separated)")
    cg.add_argument("--rename", nargs="+", metavar="OLD=NEW",
                    help="Rename columns: OLD=NEW pairs")
    cg.add_argument("--add", nargs="+", metavar="NEW=OLD",
                    help="Add columns as copies: NEW=OLD pairs")
    cg.add_argument("--split", nargs=3, action="append",
                    metavar=("COL", "NEW_COLS", "DELIM"),
                    help="Split COL on DELIM into comma-separated NEW_COLS")

    # ── Validation rules ──────────────────────────────────────────────────────
    vg = parser.add_argument_group("Row Validation Rules")
    vg.add_argument("--rules", metavar="JSON_FILE",
                    help="JSON file containing validation rules list")
    vg.add_argument("--rule-action", choices=["flag", "remove"], default="flag",
                    help="Action when rows fail validation rules (default: flag)")

    # ── I/O ───────────────────────────────────────────────────────────────────
    ig = parser.add_argument_group("Input / Output")
    ig.add_argument("--encoding",      metavar="ENC",
                    help="Force file encoding (utf-8, latin1, cp1252, ...)")
    ig.add_argument("--output-format",
                    choices=["csv","tsv","json","jsonl","xlsx","parquet","auto"],
                    default="auto",
                    help="Output format (default: same as input)")
    ig.add_argument("--chunk", type=int, metavar="N",
                    help="Chunk size for large files (rows per chunk)")
    ig.add_argument("--quick", action="store_true",
                    help="Skip schema inference and memory optimisation")

    # ── Reporting ─────────────────────────────────────────────────────────────
    rg = parser.add_argument_group("Reporting")
    rg.add_argument("--report",         metavar="PATH", help="JSON quality report path")
    rg.add_argument("--audit",          metavar="PATH", help="JSON audit log path")
    rg.add_argument("--no-fingerprint", action="store_true",
                    help="Skip SHA-256 fingerprint computation")
    rg.add_argument("-q","--quiet",     action="store_true",
                    help="Suppress all output (exit 0=success, 1=error)")

    args = parser.parse_args()

    # ── Resolve paths ─────────────────────────────────────────────────────────
    input_path = Path(args.input).resolve()
    if not input_path.exists():
        print(f"Error: file not found: {input_path}", file=sys.stderr)
        return 1

    if args.output:
        output_path = Path(args.output).resolve()
    else:
        ext_map = {"csv":".csv","tsv":".tsv","json":".json",
                   "jsonl":".jsonl","xlsx":".xlsx","parquet":".parquet"}
        out_ext = (ext_map.get(args.output_format, input_path.suffix)
                   if args.output_format != "auto"
                   else input_path.suffix)
        output_path = input_path.with_name(f"{input_path.stem}_clean{out_ext}")

    # ── Build opts ────────────────────────────────────────────────────────────
    opts: Dict = {
        "normalize":            args.normalize,
        "trim":                 args.trim,
        "dedup":                args.dedup,
        "drop_constant":        args.drop_constant,
        "auto_types":           args.auto_types,
        "validate_formats":     args.validate_formats,
        "detect_outliers":      args.detect_outliers,
        "outlier_method":       args.outlier_method,
        "outlier_contamination": args.outlier_contamination,
        "impute_strategy":      args.impute_strategy,
        "fill_value":           args.fill,
        "drop_na":              args.drop_na,
        "drop_col_threshold":   args.drop_col_threshold,
        "encoding":             args.encoding,
        "output_format":        args.output_format,
        "chunk_size":           args.chunk,
        "quick":                args.quick,
        "no_fingerprint":       args.no_fingerprint,
        "report_path":          args.report,
        "audit_path":           args.audit,
    }

    if args.keep:
        opts["keep"] = [c.strip() for c in args.keep.split(",") if c.strip()]
    if args.drop:
        opts["drop"] = [c.strip() for c in args.drop.split(",") if c.strip()]
    if args.split:
        opts["split"] = [
            {"column": col, "new_columns": nc.split(","), "delimiter": delim}
            for col, nc, delim in args.split
        ]
    if args.add:
        opts["add"] = _parse_kv(args.add, "--add")
    if args.rename:
        opts["rename"] = _parse_kv(args.rename, "--rename")
    if args.rules:
        try:
            with open(args.rules) as fh:
                opts["rules"] = json.load(fh)
            opts["rule_action"] = args.rule_action
        except Exception as exc:
            print(f"Error reading rules file: {exc}", file=sys.stderr)
            return 1

    # ── Run ───────────────────────────────────────────────────────────────────
    engine = CleanREngine(verbose=not args.quiet)
    try:
        result = engine.clean(input_path, output_path, **opts)
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        if not args.quiet:
            import traceback
            traceback.print_exc()
        return 1

    if not args.quiet:
        post   = result["post_profile"]
        rows   = post.row_count
        cols   = post.col_count
        score  = post.quality_score
        label  = post.quality_label
        elapsed = result["elapsed"]
        print(f"\n  Done: {elapsed:.2f}s  |  {rows:,} rows x {cols} cols  "
              f"|  Quality: {label} ({score:.0f}/100)")
        print(f"  Report: {result['report_path']}")
        print(f"  Audit:  {result['audit_path']}\n")

    return 0


def _parse_kv(pairs: List[str], flag: str) -> Dict[str, str]:
    out = {}
    for p in pairs:
        if "=" not in p:
            print(f"Warning: malformed {flag} arg '{p}' (expected KEY=VALUE)",
                  file=sys.stderr)
            continue
        k, _, v = p.partition("=")
        out[k.strip()] = v.strip()
    return out


if __name__ == "__main__":
    sys.exit(main())
