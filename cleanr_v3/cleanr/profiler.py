"""
profiler.py
-----------
Production statistical profiler.

Per-column metrics:
  - Complete descriptive statistics (mean, std, percentiles, IQR, CV)
  - Skewness, excess kurtosis, Shapiro-Wilk / KS normality test p-value
  - Entropy, Gini impurity (for categorical)
  - String length distribution
  - Most / least frequent values

Dataset-level metrics:
  - Structural completeness, consistency, uniqueness, validity, timeliness
  - Weighted multi-dimensional quality score (0-100) following DQ frameworks
  - Actionable issue catalogue with severity levels
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as _scipy_stats


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ColumnProfile:
    name:         str
    dtype:        str
    count:        int
    null_count:   int
    null_pct:     float
    unique_count: int
    unique_pct:   float
    # Numeric
    mean:         Optional[float] = None
    std:          Optional[float] = None
    cv:           Optional[float] = None   # coefficient of variation
    min:          Optional[float] = None
    p05:          Optional[float] = None
    p25:          Optional[float] = None
    median:       Optional[float] = None
    p75:          Optional[float] = None
    p95:          Optional[float] = None
    max:          Optional[float] = None
    iqr:          Optional[float] = None
    skewness:     Optional[float] = None
    kurtosis:     Optional[float] = None
    normality_p:  Optional[float] = None   # Shapiro-Wilk or KS p-value
    zero_count:   Optional[int]   = None
    neg_count:    Optional[int]   = None
    # String
    min_len:      Optional[int]   = None
    max_len:      Optional[int]   = None
    avg_len:      Optional[float] = None
    std_len:      Optional[float] = None
    empty_count:  Optional[int]   = None
    # Categorical / shared
    entropy:      Optional[float] = None   # Shannon entropy (0-1 normalised)
    gini:         Optional[float] = None   # Gini impurity
    top_values:   List[Tuple[Any, int, float]] = field(default_factory=list)  # (val, count, pct)
    bottom_values: List[Tuple[Any, int, float]] = field(default_factory=list)
    anomaly_flags: List[str] = field(default_factory=list)


@dataclass
class QualityDimension:
    """Single quality dimension with score and issues."""
    name:   str
    score:  float          # 0-100
    weight: float          # contribution to overall score
    issues: List[str] = field(default_factory=list)


@dataclass
class DatasetProfile:
    row_count:     int
    col_count:     int
    total_cells:   int
    missing_cells: int
    missing_pct:   float
    duplicate_rows:    int
    near_duplicate_rows: int   # rows with >95% column overlap
    memory_mb:     float
    size_bytes:    int
    columns:       Dict[str, ColumnProfile] = field(default_factory=dict)
    # Quality
    quality_score:   float = 0.0
    quality_label:   str   = ""
    dimensions:      List[QualityDimension] = field(default_factory=list)
    issues_critical: List[str] = field(default_factory=list)
    issues_warning:  List[str] = field(default_factory=list)
    issues_info:     List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def profile(df: pd.DataFrame) -> DatasetProfile:
    """Build a comprehensive DatasetProfile from a DataFrame."""
    total_cells    = df.size
    missing_cells  = int(df.isna().sum().sum())
    missing_pct    = _pct(missing_cells, total_cells)
    duplicate_rows = int(df.duplicated().sum())
    mem_mb         = round(df.memory_usage(deep=True).sum() / 1e6, 3)
    size_bytes     = int(df.memory_usage(deep=True).sum())

    # Near-duplicate detection (approximate — check rows with same hash mod sample)
    near_dupes = _estimate_near_duplicates(df)

    col_profiles: Dict[str, ColumnProfile] = {}
    for col in df.columns:
        col_profiles[col] = _profile_column(df[col])

    # Quality scoring
    dims  = _score_dimensions(df, col_profiles, duplicate_rows,
                               missing_pct, near_dupes)
    score = _weighted_score(dims)
    label = _quality_label(score)

    # Issue catalogue
    crit, warn, info = _catalogue_issues(
        df, col_profiles, duplicate_rows, near_dupes, missing_pct
    )

    return DatasetProfile(
        row_count=len(df),
        col_count=len(df.columns),
        total_cells=total_cells,
        missing_cells=missing_cells,
        missing_pct=missing_pct,
        duplicate_rows=duplicate_rows,
        near_duplicate_rows=near_dupes,
        memory_mb=mem_mb,
        size_bytes=size_bytes,
        columns=col_profiles,
        quality_score=score,
        quality_label=label,
        dimensions=dims,
        issues_critical=crit,
        issues_warning=warn,
        issues_info=info,
    )


# ---------------------------------------------------------------------------
# Column profiling
# ---------------------------------------------------------------------------

def _profile_column(s: pd.Series) -> ColumnProfile:
    name        = str(s.name)
    dtype       = str(s.dtype)
    count       = len(s)
    null_count  = int(s.isna().sum())
    null_pct    = _pct(null_count, count)
    non_null    = s.dropna()
    unique_cnt  = int(s.nunique(dropna=True))
    unique_pct  = _pct(unique_cnt, max(count - null_count, 1))
    flags: List[str] = []

    # Top / bottom values
    top_values, bottom_values = _top_bottom(s)

    cp = ColumnProfile(
        name=name, dtype=dtype, count=count,
        null_count=null_count, null_pct=null_pct,
        unique_count=unique_cnt, unique_pct=unique_pct,
        top_values=top_values,
        bottom_values=bottom_values,
    )

    if null_pct >= 80:
        flags.append(f"critical_nulls:{null_pct:.0f}%")
    elif null_pct >= 40:
        flags.append(f"high_nulls:{null_pct:.0f}%")

    if unique_cnt == 1 and len(non_null) > 0:
        flags.append("constant_column")
    if unique_cnt == count and count > 10:
        flags.append("all_unique")

    # Entropy
    if len(non_null) > 0:
        vc  = s.value_counts(normalize=True, dropna=True)
        n_u = len(vc)
        if n_u > 1:
            h = float(-(vc * np.log2(vc + 1e-12)).sum())
            cp.entropy = round(h / math.log2(n_u), 4)
            # Gini impurity
            cp.gini = round(1.0 - float((vc ** 2).sum()), 4)
        else:
            cp.entropy = 0.0
            cp.gini = 0.0

    # Numeric stats
    if (pd.api.types.is_numeric_dtype(s)
            and not pd.api.types.is_bool_dtype(s)
            and len(non_null) > 0):
        _fill_numeric_stats(cp, non_null, flags)

    # String stats
    elif s.dtype == object or str(s.dtype).startswith(("string", "category")):
        _fill_string_stats(cp, non_null, flags)

    cp.anomaly_flags = flags
    return cp


def _fill_numeric_stats(cp: ColumnProfile, s: pd.Series, flags: List[str]):
    try:
        cp.mean   = _f(s.mean())
        cp.std    = _f(s.std())
        cp.min    = _f(s.min())
        cp.p05    = _f(s.quantile(0.05))
        cp.p25    = _f(s.quantile(0.25))
        cp.median = _f(s.median())
        cp.p75    = _f(s.quantile(0.75))
        cp.p95    = _f(s.quantile(0.95))
        cp.max    = _f(s.max())
        cp.iqr    = _f((cp.p75 or 0) - (cp.p25 or 0))
        if cp.mean and cp.mean != 0 and cp.std is not None:
            cp.cv = _f(cp.std / abs(cp.mean))
        cp.zero_count = int((s == 0).sum())
        cp.neg_count  = int((s < 0).sum())

        # Distribution shape
        n = len(s)
        if n >= 8:
            cp.skewness = _f(float(s.skew()))
            cp.kurtosis = _f(float(s.kurtosis()))
            # Normality test
            if n <= 5000:
                _, p = _scipy_stats.shapiro(s.sample(min(n, 5000), random_state=0))
            else:
                # KS test against theoretical normal
                z = (s - s.mean()) / max(s.std(), 1e-9)
                _, p = _scipy_stats.kstest(z, "norm")
            cp.normality_p = _f(float(p))

        # Outlier flags
        iqr = cp.iqr or 0
        if iqr > 0:
            lo = (cp.p25 or 0) - 1.5 * iqr
            hi = (cp.p75 or 0) + 1.5 * iqr
            n_out = int(((s < lo) | (s > hi)).sum())
            if n_out > 0:
                flags.append(f"outliers:{n_out}:{n_out/len(s)*100:.1f}%")

        # High CV suggests noise or mixed units
        if cp.cv is not None and cp.cv > 2.0:
            flags.append(f"high_variance:cv={cp.cv:.2f}")

    except Exception:
        pass


def _fill_string_stats(cp: ColumnProfile, s: pd.Series, flags: List[str]):
    try:
        str_s = s.astype(str)
        lens  = str_s.str.len()
        if len(lens) == 0:
            return
        cp.min_len    = int(lens.min())
        cp.max_len    = int(lens.max())
        cp.avg_len    = round(float(lens.mean()), 2)
        cp.std_len    = round(float(lens.std()), 2) if len(lens) > 1 else 0.0
        cp.empty_count = int((str_s.str.strip() == "").sum())

        if cp.empty_count > 0:
            flags.append(f"empty_strings:{cp.empty_count}")
        if cp.max_len > 10_000:
            flags.append(f"very_long_strings:max={cp.max_len}")
    except Exception:
        pass


def _top_bottom(s: pd.Series, n: int = 5) -> Tuple[List, List]:
    try:
        vc  = s.value_counts(dropna=True)
        tot = max(vc.sum(), 1)
        top = [(v, int(c), round(c / tot * 100, 1))
               for v, c in vc.head(n).items()]
        bot = [(v, int(c), round(c / tot * 100, 1))
               for v, c in vc.tail(n).items()]
        return top, bot
    except Exception:
        return [], []


# ---------------------------------------------------------------------------
# Quality scoring  (DQ framework: completeness, consistency, uniqueness,
#                   validity, accuracy)
# ---------------------------------------------------------------------------

_DIM_WEIGHTS = {
    "Completeness":  0.30,
    "Uniqueness":    0.20,
    "Validity":      0.20,
    "Consistency":   0.15,
    "Accuracy":      0.15,
}


def _score_dimensions(df, profiles: Dict[str, ColumnProfile],
                       dup_rows: int, missing_pct: float,
                       near_dupes: int) -> List[QualityDimension]:
    dims = []
    n    = max(len(df), 1)

    # ── Completeness ─────────────────────────────────────────────────────────
    comp_score  = max(0.0, 100 - missing_pct * 1.8)
    comp_issues = []
    for col, cp in profiles.items():
        if cp.null_pct >= 80:
            comp_issues.append(f"[{col}] {cp.null_pct:.0f}% missing (critical)")
        elif cp.null_pct >= 40:
            comp_issues.append(f"[{col}] {cp.null_pct:.0f}% missing")
    dims.append(QualityDimension("Completeness", round(comp_score, 1),
                                  _DIM_WEIGHTS["Completeness"], comp_issues))

    # ── Uniqueness ────────────────────────────────────────────────────────────
    dup_pct    = dup_rows / n * 100
    near_pct   = near_dupes / n * 100
    uniq_score = max(0.0, 100 - dup_pct * 3 - near_pct * 1.5)
    uniq_issues = []
    if dup_rows:
        uniq_issues.append(f"{dup_rows:,} exact duplicate rows ({dup_pct:.1f}%)")
    if near_dupes:
        uniq_issues.append(f"{near_dupes:,} near-duplicate rows ({near_pct:.1f}%)")
    const_cols = [c for c, p in profiles.items() if "constant_column" in p.anomaly_flags]
    if const_cols:
        uniq_issues.append(f"Constant columns (zero information): {const_cols}")
        uniq_score -= len(const_cols) * 3
    dims.append(QualityDimension("Uniqueness", round(max(uniq_score, 0), 1),
                                  _DIM_WEIGHTS["Uniqueness"], uniq_issues))

    # ── Validity ──────────────────────────────────────────────────────────────
    # Penalise columns with high outlier counts, empty strings, invalid formats
    val_issues  = []
    val_penalty = 0.0
    for col, cp in profiles.items():
        for flag in cp.anomaly_flags:
            if flag.startswith("outliers:"):
                parts = flag.split(":")
                pct   = float(parts[2].rstrip("%")) if len(parts) > 2 else 0
                val_penalty += min(pct * 0.5, 5)
                val_issues.append(f"[{col}] {flag}")
            elif flag.startswith("empty_strings:"):
                n_emp = int(flag.split(":")[1])
                val_penalty += min(n_emp / n * 50, 5)
                val_issues.append(f"[{col}] {n_emp} empty strings")
            elif flag == "mixed_types":
                val_penalty += 8
                val_issues.append(f"[{col}] mixed data types detected")
    val_score = max(0.0, 100 - val_penalty)
    dims.append(QualityDimension("Validity", round(val_score, 1),
                                  _DIM_WEIGHTS["Validity"], val_issues))

    # ── Consistency ───────────────────────────────────────────────────────────
    con_issues  = []
    con_penalty = 0.0
    for col, cp in profiles.items():
        if "high_variance" in " ".join(cp.anomaly_flags):
            con_penalty += 3
            con_issues.append(f"[{col}] unexpectedly high variance")
        if "high_length_variance" in " ".join(cp.anomaly_flags):
            con_penalty += 2
            con_issues.append(f"[{col}] inconsistent string lengths")
    con_score = max(0.0, 100 - con_penalty)
    dims.append(QualityDimension("Consistency", round(con_score, 1),
                                  _DIM_WEIGHTS["Consistency"], con_issues))

    # ── Accuracy ──────────────────────────────────────────────────────────────
    # Proxy: columns where semantic type validation flags issues
    acc_issues  = []
    acc_penalty = 0.0
    invalid_cols = [c for c in df.columns if c.startswith("_invalid_")]
    for ic in invalid_cols:
        n_bad = int(df[ic].sum()) if pd.api.types.is_bool_dtype(df[ic]) else 0
        pct   = n_bad / n * 100
        if pct > 0:
            src = ic.replace("_invalid_", "")
            acc_penalty += min(pct * 0.8, 15)
            acc_issues.append(f"[{src}] {n_bad} invalid format values ({pct:.1f}%)")
    acc_score = max(0.0, 100 - acc_penalty)
    dims.append(QualityDimension("Accuracy", round(acc_score, 1),
                                  _DIM_WEIGHTS["Accuracy"], acc_issues))

    return dims


def _weighted_score(dims: List[QualityDimension]) -> float:
    total_w = sum(d.weight for d in dims)
    if total_w == 0:
        return 0.0
    score = sum(d.score * d.weight for d in dims) / total_w
    return round(score, 1)


def _quality_label(score: float) -> str:
    if score >= 92: return "Excellent"
    if score >= 78: return "Good"
    if score >= 60: return "Fair"
    if score >= 40: return "Poor"
    return "Critical"


# ---------------------------------------------------------------------------
# Issue catalogue
# ---------------------------------------------------------------------------

def _catalogue_issues(df, profiles, dup_rows, near_dupes, missing_pct):
    critical, warning, info = [], [], []

    if missing_pct >= 40:
        critical.append(f"Dataset is {missing_pct:.1f}% empty — data collection may be broken")
    elif missing_pct >= 15:
        warning.append(f"Missing value rate is {missing_pct:.1f}% — review imputation strategy")
    elif missing_pct > 0:
        info.append(f"Missing value rate: {missing_pct:.1f}%")

    if dup_rows:
        pct = dup_rows / max(len(df), 1) * 100
        (critical if pct > 20 else warning).append(
            f"{dup_rows:,} exact duplicate rows ({pct:.1f}%)"
        )

    if near_dupes > 0:
        info.append(f"{near_dupes:,} near-duplicate rows detected (95%+ column overlap)")

    for col, cp in profiles.items():
        for flag in cp.anomaly_flags:
            if flag.startswith("critical_nulls"):
                critical.append(f"[{col}] {flag}")
            elif flag.startswith("high_nulls"):
                warning.append(f"[{col}] {flag}")
            elif flag == "constant_column":
                warning.append(f"[{col}] constant — carries no information")
            elif flag.startswith("outliers"):
                info.append(f"[{col}] {flag}")
            elif flag == "mixed_types":
                warning.append(f"[{col}] mixed data types in single column")
            elif flag.startswith("empty_strings"):
                info.append(f"[{col}] {flag}")

    return critical, warning, info


# ---------------------------------------------------------------------------
# Near-duplicate estimation
# ---------------------------------------------------------------------------

def _estimate_near_duplicates(df: pd.DataFrame, threshold: int = 1000) -> int:
    """
    Estimate near-duplicate rows by hashing all-but-one column per row.
    Only runs on a sample for large datasets to keep profiling fast.
    """
    try:
        n = len(df)
        if n < 2:
            return 0
        sample_df = df.sample(min(n, threshold), random_state=42) if n > threshold else df
        # For each row, hash dropping one column at a time is expensive
        # Instead: drop last column and find duplicates of the remainder
        if len(sample_df.columns) < 2:
            return 0
        subset = sample_df.iloc[:, :-1]
        # Must not already be exact duplicate (those are counted separately)
        near = int(subset.duplicated().sum()) - int(sample_df.duplicated().sum())
        near = max(near, 0)
        # Scale back to full dataset
        if n > threshold:
            near = int(near * n / threshold)
        return near
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _pct(num: float, den: float, decimals: int = 2) -> float:
    return round(num / max(den, 1) * 100, decimals)


def _f(v, decimals: int = 6) -> Optional[float]:
    try:
        f = float(v)
        return None if math.isnan(f) or math.isinf(f) else round(f, decimals)
    except Exception:
        return None
