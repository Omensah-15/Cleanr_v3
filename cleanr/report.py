"""
report.py
---------
Generates terminal reports and structured JSON quality reports.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from cleanr.audit import AuditLog, Fingerprint
from cleanr.profiler import DatasetProfile, QualityDimension

# ANSI colours
_C = {
    "green":   "\033[92m",
    "yellow":  "\033[93m",
    "red":     "\033[91m",
    "cyan":    "\033[96m",
    "dim":     "\033[2m",
    "bold":    "\033[1m",
    "reset":   "\033[0m",
}

_SCORE_COLOUR = {
    "Excellent": _C["green"],
    "Good":      _C["green"],
    "Fair":      _C["yellow"],
    "Poor":      _C["red"],
    "Critical":  _C["red"],
}

W = 72  # terminal width


def render_terminal(
    pre:        DatasetProfile,
    post:       DatasetProfile,
    audit:      AuditLog,
    input_fp:   Optional[Fingerprint],
    output_fp:  Optional[Fingerprint],
    elapsed:    float,
    input_path: Path,
    output_path: Path,
) -> str:
    lines: List[str] = []

    def rule(c="─"): lines.append(c * W)
    def blank(): lines.append("")
    def head(t):
        blank()
        lines.append(f"  {_C['bold']}{t}{_C['reset']}")
        rule()

    # ── Banner ────────────────────────────────────────────────────────────────
    lines.append("+" + "=" * (W - 2) + "+")
    lines.append("|" + " CleanR v3  Data Quality Report ".center(W - 2) + "|")
    lines.append("+" + "=" * (W - 2) + "+")

    # ── Files ─────────────────────────────────────────────────────────────────
    head("Source Files")
    lines.append(f"  Input   {input_path}")
    lines.append(f"  Output  {output_path}")
    lines.append(f"  Elapsed {elapsed:.2f}s")

    # ── Overview table ────────────────────────────────────────────────────────
    head("Dataset Overview")
    _row_table(lines, [
        ("Metric",          "Before",                          "After"),
        ("Rows",            f"{pre.row_count:,}",              f"{post.row_count:,}"),
        ("Columns",         str(pre.col_count),                str(post.col_count)),
        ("Missing cells",   f"{pre.missing_cells:,} ({pre.missing_pct:.1f}%)",
                            f"{post.missing_cells:,} ({post.missing_pct:.1f}%)"),
        ("Duplicate rows",  f"{pre.duplicate_rows:,}",         f"{post.duplicate_rows:,}"),
        ("Memory",          f"{pre.memory_mb:.2f} MB",         f"{post.memory_mb:.2f} MB"),
    ])

    # ── Quality score ─────────────────────────────────────────────────────────
    head("Quality Score")
    colour = _SCORE_COLOUR.get(post.quality_label, "")
    bar    = _bar(post.quality_score, 100, 44)
    lines.append(f"  {colour}{post.quality_label} — {post.quality_score:.1f} / 100{_C['reset']}")
    lines.append(f"  [{bar}]")
    blank()
    # Dimension breakdown
    lines.append(f"  {'Dimension':<16} {'Score':>7}  {'Weight':>7}")
    rule("·")
    for dim in post.dimensions:
        bar_s = _mini_bar(dim.score, 20)
        lines.append(f"  {dim.name:<16} {dim.score:>6.1f}  "
                     f"  {int(dim.weight*100):>3}%  {bar_s}")

    # ── Cleaning actions ──────────────────────────────────────────────────────
    entries = audit.to_list()
    if entries:
        head("Cleaning Actions")
        for e in entries:
            sym = "!" if e["level"] == "WARNING" else "+"
            lines.append(f"  {sym}  [{e['plugin']}]  {e['action']}")

    # ── Column profiles ───────────────────────────────────────────────────────
    head("Column Profiles  (post-clean)")
    hdr = f"  {'Column':<26} {'Type':<14} {'Nulls':>7}  {'Unique':>8}  Notes"
    lines.append(hdr)
    rule("·")
    for col, cp in post.columns.items():
        null_s  = f"{cp.null_pct:.0f}%"
        uniq_s  = f"{cp.unique_count:,}"
        flags_s = "  " + "  ".join(cp.anomaly_flags) if cp.anomaly_flags else ""
        lines.append(f"  {col[:26]:<26} {cp.dtype[:14]:<14} "
                     f"{null_s:>7}  {uniq_s:>8}{flags_s}")
        if cp.mean is not None:
            lines.append(f"    mean={cp.mean}  std={cp.std}  "
                         f"min={cp.min}  p25={cp.p25}  "
                         f"median={cp.median}  p75={cp.p75}  max={cp.max}")

    # ── Issues ────────────────────────────────────────────────────────────────
    if post.issues_critical or post.issues_warning or post.issues_info:
        head("Issues")
        for iss in post.issues_critical:
            lines.append(f"  {_C['red']}[CRITICAL]{_C['reset']}  {iss}")
        for iss in post.issues_warning:
            lines.append(f"  {_C['yellow']}[WARNING] {_C['reset']}  {iss}")
        for iss in post.issues_info:
            lines.append(f"  {_C['dim']}[INFO]    {_C['reset']}  {iss}")

    # ── Warnings from audit ───────────────────────────────────────────────────
    warns = audit.warnings()
    if warns:
        head("Pipeline Warnings")
        for w in warns:
            lines.append(f"  {_C['yellow']}!{_C['reset']}  {w}")

    # ── Fingerprints ──────────────────────────────────────────────────────────
    if input_fp or output_fp:
        head("Integrity Fingerprints  (SHA-256)")
        if input_fp:
            lines.append(f"  Input  file  {input_fp.file_hash[:32]}...")
            lines.append(f"  Input  data  {input_fp.data_hash[:32]}...")
        if output_fp:
            lines.append(f"  Output file  {output_fp.file_hash[:32]}...")
            lines.append(f"  Output data  {output_fp.data_hash[:32]}...")

    blank()
    return "\n".join(lines)


def build_json_report(
    pre:         DatasetProfile,
    post:        DatasetProfile,
    schema:      Dict,
    audit:       AuditLog,
    input_fp:    Optional[Fingerprint],
    output_fp:   Optional[Fingerprint],
    elapsed:     float,
    input_path:  Path,
    output_path: Path,
) -> Dict:
    def _profile_dict(p: DatasetProfile) -> Dict:
        return {
            "rows":          p.row_count,
            "columns":       p.col_count,
            "missing_cells": p.missing_cells,
            "missing_pct":   p.missing_pct,
            "duplicate_rows": p.duplicate_rows,
            "memory_mb":     p.memory_mb,
        }

    def _col_profile_dict(cp) -> Dict:
        d = {
            "dtype":         cp.dtype,
            "null_count":    cp.null_count,
            "null_pct":      cp.null_pct,
            "unique_count":  cp.unique_count,
            "unique_pct":    cp.unique_pct,
            "entropy":       cp.entropy,
            "anomaly_flags": cp.anomaly_flags,
            "top_values":    cp.top_values,
        }
        for attr in ("mean","std","min","p05","p25","median","p75","p95","max",
                     "iqr","skewness","kurtosis","normality_p","zero_count","neg_count",
                     "min_len","max_len","avg_len","empty_count"):
            v = getattr(cp, attr, None)
            if v is not None:
                d[attr] = v
        return d

    return {
        "cleanr_version":  "3.0.0",
        "generated_at":    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input":           str(input_path),
        "output":          str(output_path),
        "elapsed_seconds": elapsed,
        "pre_clean":       _profile_dict(pre),
        "post_clean":      _profile_dict(post),
        "quality": {
            "score":      post.quality_score,
            "label":      post.quality_label,
            "dimensions": [
                {"name": d.name, "score": d.score,
                 "weight": d.weight, "issues": d.issues}
                for d in post.dimensions
            ],
        },
        "issues": {
            "critical": post.issues_critical,
            "warning":  post.issues_warning,
            "info":     post.issues_info,
        },
        "column_profiles": {
            col: _col_profile_dict(cp) for col, cp in post.columns.items()
        },
        "schema": {
            col: {
                "inferred_dtype":        s.inferred_dtype,
                "semantic_type":         s.semantic_type,
                "cardinality":           s.cardinality,
                "cardinality_ratio":     s.cardinality_ratio,
                "entropy":               s.entropy,
                "distribution":          s.distribution,
                "skewness":              s.skewness,
                "kurtosis":              s.kurtosis,
                "is_identifier":         s.is_identifier,
                "is_constant":           s.is_constant,
                "coercion_success_rate": s.coercion_success_rate,
                "datetime_format":       s.datetime_format,
                "anomalies":             s.anomalies,
                "notes":                 s.notes,
                "confidence":            s.confidence,
            }
            for col, s in schema.items()
        },
        "audit_trail":  audit.to_list(),
        "fingerprints": {
            "input":  asdict(input_fp)  if input_fp  else None,
            "output": asdict(output_fp) if output_fp else None,
        },
    }


def save_json_report(data: Dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _bar(value: float, total: float, width: int = 44) -> str:
    filled = int(round(max(value, 0) / max(total, 1) * width))
    return "#" * filled + "-" * (width - filled)


def _mini_bar(value: float, width: int = 20) -> str:
    filled = int(round(max(value, 0) / 100 * width))
    return "[" + "#" * filled + " " * (width - filled) + "]"


def _row_table(lines: List[str], rows: List[Tuple]):
    col_w = [max(len(str(row[i])) for row in rows) for i in range(len(rows[0]))]
    for j, row in enumerate(rows):
        line = "  " + "  ".join(str(cell).ljust(col_w[i]) for i, cell in enumerate(row))
        lines.append(line)
        if j == 0:
            lines.append("  " + "  ".join("-" * w for w in col_w))
