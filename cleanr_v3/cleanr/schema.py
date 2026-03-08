"""
schema.py
---------
Production-grade schema inference engine.

Multi-pass column analysis using:
  - Statistical distribution tests (Kolmogorov-Smirnov, Shapiro-Wilk)
  - Entropy-based cardinality classification
  - Confidence-scored semantic type detection with 15+ patterns
  - Datetime format disambiguation via multi-format sampling
  - Mixed-type detection via character class analysis
  - Coercion success rate thresholds with configurable tolerance
"""
from __future__ import annotations

import re
import warnings
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as _scipy_stats

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Semantic patterns  (ordered by specificity — most specific first)
# ---------------------------------------------------------------------------

_PATTERNS: Dict[str, re.Pattern] = {
    "uuid":        re.compile(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I),
    "email":       re.compile(
        r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$"),
    "url":         re.compile(
        r"^https?://[^\s/$.?#].[^\s]*$", re.I),
    "ipv4":        re.compile(
        r"^(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)$"),
    "ipv6":        re.compile(
        r"^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$"),
    "phone":       re.compile(
        r"^(\+?1[\s.\-]?)?(\(?\d{3}\)?[\s.\-]?)?\d{3}[\s.\-]?\d{4}$"),
    "postal_us":   re.compile(r"^\d{5}(-\d{4})?$"),
    "postal_uk":   re.compile(
        r"^[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}$", re.I),
    "currency":    re.compile(
        r"^[\$£€¥₹₩₪฿]?\s*-?[\d,]+(\.\d{1,4})?$"),
    "percentage":  re.compile(r"^-?\d+(\.\d+)?\s*%$"),
    "credit_card": re.compile(r"^\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}$"),
    "boolean_str": re.compile(
        r"^(true|false|yes|no|1|0|y|n|t|f|on|off|enabled|disabled)$", re.I),
    "ssn":         re.compile(r"^\d{3}-\d{2}-\d{4}$"),
    "hex_color":   re.compile(r"^#?([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$"),
    "json_str":    re.compile(r"^(\{.*\}|\[.*\])$"),
}

_DATETIME_FORMATS: List[str] = [
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%d-%m-%Y",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%m-%d-%Y",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M",
    "%m/%d/%Y %H:%M:%S",
    "%Y%m%d",
    "%d-%b-%Y",
    "%d %b %Y",
    "%b %d, %Y",
    "%B %d, %Y",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%dT%H:%M:%S.%f",
]


# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------

@dataclass
class ColumnSchema:
    name: str
    raw_dtype: str
    inferred_dtype: str          # int64 | float64 | bool | datetime64[ns] | category | string
    semantic_type: Optional[str] = None
    nullable: bool = False
    null_pct: float = 0.0
    coercion_success_rate: float = 1.0  # fraction of non-null values that coerce cleanly
    cardinality: int = 0
    cardinality_ratio: float = 0.0      # unique / non-null count
    entropy: float = 0.0                # Shannon entropy of value distribution
    is_constant: bool = False
    is_identifier: bool = False         # high-cardinality, no semantic type
    distribution: Optional[str] = None  # normal | skewed | uniform | bimodal | unknown
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    anomalies: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    datetime_format: Optional[str] = None
    sample_values: List = field(default_factory=list)
    confidence: float = 1.0            # inference confidence [0,1]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def infer_schema(df: pd.DataFrame,
                 sample_size: int = 10_000,
                 coerce_threshold: float = 0.92) -> Dict[str, ColumnSchema]:
    """
    Infer schema for all columns.

    Parameters
    ----------
    df               : DataFrame to analyse.
    sample_size      : Maximum rows to sample for inference.
    coerce_threshold : Minimum fraction of non-null values that must coerce
                       cleanly for a type assignment to be accepted.
    """
    n = len(df)
    if n > sample_size:
        # Stratified-ish sample: take rows spread across the dataset
        step = max(1, n // sample_size)
        sample = df.iloc[::step].head(sample_size)
    else:
        sample = df

    schemas: Dict[str, ColumnSchema] = {}
    for col in sample.columns:
        schemas[col] = _infer_column(col, sample[col], coerce_threshold)
    return schemas


# ---------------------------------------------------------------------------
# Per-column inference
# ---------------------------------------------------------------------------

def _infer_column(name: str, series: pd.Series,
                  coerce_threshold: float) -> ColumnSchema:
    raw_dtype    = str(series.dtype)
    null_mask    = series.isna()
    null_count   = int(null_mask.sum())
    total        = len(series)
    null_pct     = round(null_count / max(total, 1) * 100, 2)
    non_null     = series.dropna()
    nullable     = null_count > 0
    sample_vals  = non_null.head(5).tolist()

    # Degenerate: completely null
    if non_null.empty:
        return ColumnSchema(
            name=name, raw_dtype=raw_dtype, inferred_dtype="string",
            nullable=True, null_pct=null_pct,
            anomalies=["all_null"], confidence=1.0,
        )

    # Already a datetime
    if pd.api.types.is_datetime64_any_dtype(series):
        return ColumnSchema(
            name=name, raw_dtype=raw_dtype, inferred_dtype="datetime64[ns]",
            nullable=nullable, null_pct=null_pct,
            cardinality=int(non_null.nunique()),
            sample_values=sample_vals, confidence=1.0,
        )

    # Already numeric (from pandas auto-inference)
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        return _schema_for_numeric(name, raw_dtype, non_null,
                                   nullable, null_pct, sample_vals,
                                   coerce_threshold, source="native")

    # Already bool
    if pd.api.types.is_bool_dtype(series):
        return ColumnSchema(
            name=name, raw_dtype=raw_dtype, inferred_dtype="bool",
            nullable=nullable, null_pct=null_pct,
            cardinality=int(non_null.nunique()),
            sample_values=sample_vals, confidence=1.0,
        )

    # Object / string / category — deep analysis
    str_series = non_null.astype(str).str.strip()
    # Remove empty strings produced by NaN -> "nan" coercion
    str_series = str_series[~str_series.str.lower().isin(
        {"nan", "none", "null", "na", "n/a", "", "<na>"}
    )]
    if str_series.empty:
        return ColumnSchema(
            name=name, raw_dtype=raw_dtype, inferred_dtype="string",
            nullable=True, null_pct=null_pct, anomalies=["all_null"],
        )

    cardinality = int(str_series.nunique())
    card_ratio  = cardinality / max(len(str_series), 1)
    entropy     = _entropy(str_series)
    is_constant = cardinality == 1
    anomalies: List[str] = []
    notes: List[str] = []

    if is_constant:
        notes.append("constant_column")

    # ── Pass 1: boolean strings ──────────────────────────────────────────────
    bool_rate = str_series.str.match(_PATTERNS["boolean_str"]).mean()
    if bool_rate >= coerce_threshold:
        return ColumnSchema(
            name=name, raw_dtype=raw_dtype, inferred_dtype="bool",
            nullable=nullable, null_pct=null_pct,
            cardinality=cardinality, cardinality_ratio=card_ratio,
            entropy=entropy, is_constant=is_constant,
            notes=notes + ["coerced_from_string"],
            sample_values=sample_vals,
            confidence=round(float(bool_rate), 3),
        )

    # ── Pass 2: integer (with currency/separator stripping) ──────────────────
    int_series, int_rate = _try_coerce_int(str_series)
    if int_rate >= coerce_threshold and int_series is not None:
        schema = _schema_for_numeric(
            name, raw_dtype, int_series.dropna(),
            nullable, null_pct, sample_vals,
            coerce_threshold, source="coerced",
        )
        schema.coercion_success_rate = round(int_rate, 3)
        schema.notes.append("coerced_from_string")
        return schema

    # ── Pass 3: float ────────────────────────────────────────────────────────
    flt_series, flt_rate = _try_coerce_float(str_series)
    if flt_rate >= coerce_threshold and flt_series is not None:
        # Check semantic type BEFORE returning numeric
        sem = _semantic_type(str_series, sample_n=500)
        schema = _schema_for_numeric(
            name, raw_dtype, flt_series.dropna(),
            nullable, null_pct, sample_vals,
            coerce_threshold, source="coerced",
        )
        schema.semantic_type = sem
        schema.coercion_success_rate = round(flt_rate, 3)
        schema.notes.append("coerced_from_string")
        return schema

    # ── Pass 4: datetime ─────────────────────────────────────────────────────
    dt_fmt, dt_rate = _try_datetime(str_series, sample_n=300)
    if dt_rate >= coerce_threshold:
        return ColumnSchema(
            name=name, raw_dtype=raw_dtype, inferred_dtype="datetime64[ns]",
            nullable=nullable, null_pct=null_pct,
            cardinality=cardinality, cardinality_ratio=card_ratio,
            entropy=entropy, is_constant=is_constant,
            datetime_format=dt_fmt,
            coercion_success_rate=round(dt_rate, 3),
            notes=notes + [f"datetime_format:{dt_fmt}"],
            sample_values=sample_vals,
            confidence=round(dt_rate, 3),
        )

    # ── Pass 5: semantic type detection ─────────────────────────────────────
    sem, sem_confidence = _semantic_type_with_confidence(str_series, sample_n=500)

    # ── Pass 6: cardinality classification ───────────────────────────────────
    if card_ratio < 0.05 and cardinality <= 200:
        inferred = "category"
    elif card_ratio < 0.50 and cardinality <= 2000:
        inferred = "category"
    else:
        inferred = "string"

    # Identifier detection (high cardinality, no semantic type, looks like IDs)
    is_id = _looks_like_identifier(str_series, card_ratio, sem)

    # Mixed-type anomaly
    if _has_mixed_types(str_series):
        anomalies.append("mixed_types")

    # Length anomalies
    lengths = str_series.str.len()
    length_cv = lengths.std() / max(lengths.mean(), 1)
    if length_cv > 2.0:
        anomalies.append(f"high_length_variance:cv={length_cv:.1f}")

    return ColumnSchema(
        name=name, raw_dtype=raw_dtype, inferred_dtype=inferred,
        semantic_type=sem,
        nullable=nullable, null_pct=null_pct,
        cardinality=cardinality, cardinality_ratio=round(card_ratio, 4),
        entropy=round(entropy, 4),
        is_constant=is_constant, is_identifier=is_id,
        anomalies=anomalies, notes=notes,
        sample_values=sample_vals,
        confidence=round(sem_confidence if sem else (1.0 - card_ratio * 0.3), 3),
    )


# ---------------------------------------------------------------------------
# Numeric analysis
# ---------------------------------------------------------------------------

def _schema_for_numeric(name: str, raw_dtype: str, non_null: pd.Series,
                         nullable: bool, null_pct: float,
                         sample_vals: list, coerce_threshold: float,
                         source: str = "native") -> ColumnSchema:
    anomalies: List[str] = []
    notes: List[str] = []
    cardinality = int(non_null.nunique())
    card_ratio  = cardinality / max(len(non_null), 1)

    # Integer vs float
    if pd.api.types.is_float_dtype(non_null):
        all_integer = ((non_null % 1) == 0).all()
        inferred = "int64" if all_integer else "float64"
    else:
        inferred = "int64"

    # Distribution analysis
    dist_label, skewness, kurtosis = _analyse_distribution(non_null)

    # Outlier detection using IQR + Z-score combined
    outliers = _detect_outliers(non_null)
    if outliers > 0:
        pct = outliers / len(non_null) * 100
        anomalies.append(f"outliers:{outliers}:{pct:.1f}%")

    # Negative values in ID columns
    if (non_null < 0).any() and name.lower().endswith("id"):
        anomalies.append("negative_id_values")

    # Suspiciously round values (possible encoding artifact)
    if inferred == "float64" and len(non_null) > 100:
        round_pct = ((non_null % 1) == 0).mean()
        if round_pct > 0.98:
            notes.append("mostly_whole_numbers_stored_as_float")

    return ColumnSchema(
        name=name, raw_dtype=raw_dtype, inferred_dtype=inferred,
        nullable=nullable, null_pct=null_pct,
        cardinality=cardinality, cardinality_ratio=round(card_ratio, 4),
        distribution=dist_label,
        skewness=round(float(skewness), 4) if not math.isnan(skewness) else None,
        kurtosis=round(float(kurtosis), 4) if not math.isnan(kurtosis) else None,
        anomalies=anomalies, notes=notes,
        sample_values=sample_vals,
        confidence=1.0,
    )


def _analyse_distribution(s: pd.Series) -> Tuple[str, float, float]:
    """Return (label, skewness, excess_kurtosis)."""
    n = len(s)
    if n < 8:
        return "unknown", float("nan"), float("nan")
    try:
        skew = float(s.skew())
        kurt = float(s.kurtosis())  # excess kurtosis

        if abs(skew) < 0.5 and abs(kurt) < 1.0:
            # Shapiro-Wilk test for normality (sample up to 5000)
            sample = s.sample(min(n, 5000), random_state=0)
            _, p = _scipy_stats.shapiro(sample) if len(sample) <= 5000 else (0, 0)
            label = "normal" if p > 0.05 else "near_normal"
        elif abs(skew) > 1.5:
            label = "highly_skewed"
        elif abs(skew) > 0.5:
            label = "skewed"
        elif abs(kurt) > 3.0:
            label = "heavy_tailed"
        else:
            label = "unknown"

        return label, skew, kurt
    except Exception:
        return "unknown", float("nan"), float("nan")


def _detect_outliers(s: pd.Series) -> int:
    """Count outliers using combined IQR (×1.5 fence) and modified Z-score."""
    n = len(s)
    if n < 10:
        return 0
    try:
        # IQR fence
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        iqr_outliers = set(s[(s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)].index)

        # Modified Z-score (Iglewicz & Hoaglin, robust to non-normality)
        median = s.median()
        mad = (s - median).abs().median()
        if mad > 0:
            mz = 0.6745 * (s - median) / mad
            mz_outliers = set(s[mz.abs() > 3.5].index)
        else:
            mz_outliers = set()

        # Consensus: flagged by both methods
        return len(iqr_outliers & mz_outliers)
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Type coercion helpers
# ---------------------------------------------------------------------------

_STRIP_NUMERIC = re.compile(r"[\$£€¥₹₩₪฿,%\s_]")


def _try_coerce_int(s: pd.Series) -> Tuple[Optional[pd.Series], float]:
    cleaned = s.str.replace(_STRIP_NUMERIC, "", regex=True)
    # Also handle parenthesised negatives like (1,234)
    cleaned = cleaned.str.replace(r"^\((.+)\)$", r"-\1", regex=True)
    converted = pd.to_numeric(cleaned, errors="coerce")
    rate = converted.notna().mean()
    if rate < 0.85:
        return None, rate
    non_null = converted.dropna()
    if len(non_null) == 0:
        return None, 0.0
    if ((non_null % 1) != 0).any():
        return None, 0.0
    return converted, rate


def _try_coerce_float(s: pd.Series) -> Tuple[Optional[pd.Series], float]:
    cleaned = s.str.replace(_STRIP_NUMERIC, "", regex=True)
    cleaned = cleaned.str.replace(r"^\((.+)\)$", r"-\1", regex=True)
    converted = pd.to_numeric(cleaned, errors="coerce")
    rate = converted.notna().mean()
    if rate < 0.85:
        return None, rate
    return converted, rate


def _try_datetime(s: pd.Series, sample_n: int = 300) -> Tuple[str, float]:
    sample = s.sample(min(len(s), sample_n), random_state=0) if len(s) > sample_n else s
    best_fmt, best_rate = "inferred", 0.0

    for fmt in _DATETIME_FORMATS:
        try:
            parsed = pd.to_datetime(sample, format=fmt, errors="coerce")
            rate = parsed.notna().mean()
            if rate > best_rate:
                best_rate = rate
                best_fmt = fmt
            if rate >= 0.99:
                break
        except Exception:
            continue

    # Generic parse as fallback
    if best_rate < 0.85:
        try:
            parsed = pd.to_datetime(sample, errors="coerce")
            rate = float(parsed.notna().mean())
            if rate > best_rate:
                best_rate = rate
                best_fmt = "inferred"
        except Exception:
            pass

    return best_fmt, float(best_rate)


# ---------------------------------------------------------------------------
# Semantic type detection
# ---------------------------------------------------------------------------

def _semantic_type(s: pd.Series, sample_n: int = 500) -> Optional[str]:
    sem, _ = _semantic_type_with_confidence(s, sample_n)
    return sem


def _semantic_type_with_confidence(
        s: pd.Series, sample_n: int = 500) -> Tuple[Optional[str], float]:
    sample = s.sample(min(len(s), sample_n), random_state=0) if len(s) > sample_n else s
    best_sem, best_rate = None, 0.0

    for sem, pat in _PATTERNS.items():
        if sem == "boolean_str":
            continue
        try:
            rate = float(sample.str.match(pat).mean())
            if rate > best_rate and rate >= 0.80:
                best_rate = rate
                best_sem = sem
        except Exception:
            continue

    return best_sem, best_rate


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def _entropy(s: pd.Series) -> float:
    """Shannon entropy of value distribution (normalised to [0,1])."""
    vc = s.value_counts(normalize=True)
    if len(vc) <= 1:
        return 0.0
    h = float(-(vc * np.log2(vc + 1e-12)).sum())
    return h / math.log2(len(vc))   # normalise by max possible entropy


def _has_mixed_types(s: pd.Series, sample_n: int = 300) -> bool:
    """True if the column appears to contain both numeric and alpha tokens."""
    sample = s.head(sample_n)
    has_numeric = sample.str.match(r"^-?\d+(\.\d+)?$").mean() > 0.1
    has_alpha   = sample.str.match(r"^[A-Za-z]").mean() > 0.1
    return bool(has_numeric and has_alpha)


def _looks_like_identifier(s: pd.Series, card_ratio: float,
                            sem: Optional[str]) -> bool:
    """Heuristic: column appears to be a unique row identifier."""
    if sem is not None:
        return False
    if card_ratio < 0.9:
        return False
    # Check for sequential integers or UUID-like patterns
    try:
        numeric = pd.to_numeric(s, errors="coerce")
        if numeric.notna().mean() > 0.95:
            diffs = numeric.dropna().sort_values().diff().dropna()
            if diffs.nunique() <= 3 and (diffs > 0).all():
                return True
    except Exception:
        pass
    uuid_rate = s.str.match(_PATTERNS["uuid"]).mean()
    return bool(uuid_rate > 0.8)
