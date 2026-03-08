"""
imputer.py
----------
Production-grade missing value imputation engine.

Strategies (selected per-column based on dtype and missingness pattern):
  - Numeric:   KNN (default) | iterative MICE-style | median | mean | constant
  - Categorical: mode | constant | KNN (label-encoded)
  - Datetime:  forward-fill, then backward-fill (time-series friendly)
  - String:    mode | constant | "Unknown"

The strategy selection follows these rules:
  - < 5% missing   → simple (median/mode) — KNN overhead not justified
  - 5–40% missing  → KNN with k=5
  - > 40% missing  → column flagged as unreliable; constant fill with warning
  - Time-ordered datetime → forward/backward fill
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from sklearn.impute import KNNImputer, SimpleImputer
    from sklearn.preprocessing import OrdinalEncoder
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class ImputationLog:
    column:      str
    strategy:    str
    n_imputed:   int
    null_pct_before: float
    fill_value:  Optional[str] = None
    warning:     Optional[str] = None


def impute(df: pd.DataFrame,
           strategy: str = "auto",
           fill_value: Optional[str] = None,
           drop_na: bool = False,
           drop_col_threshold: Optional[float] = None,
           knn_k: int = 5) -> tuple[pd.DataFrame, List[ImputationLog]]:
    """
    Impute missing values in df.

    Parameters
    ----------
    strategy            : "auto" | "knn" | "median" | "mean" | "mode" | "constant"
    fill_value          : used when strategy=="constant" or as fallback
    drop_na             : drop rows with any remaining null
    drop_col_threshold  : drop columns with null fraction >= threshold
    knn_k               : neighbours for KNN imputation

    Returns
    -------
    (df, logs)
    """
    df    = df.copy()
    logs: List[ImputationLog] = []

    # ── 1. Drop high-null columns ─────────────────────────────────────────────
    if drop_col_threshold is not None:
        null_rates = df.isna().mean()
        cols_to_drop = null_rates[null_rates >= drop_col_threshold].index.tolist()
        for col in cols_to_drop:
            pct = null_rates[col] * 100
            logs.append(ImputationLog(
                column=col, strategy="drop_column",
                n_imputed=0, null_pct_before=round(pct, 1),
                warning=f"Dropped: {pct:.0f}% missing exceeds threshold {drop_col_threshold*100:.0f}%",
            ))
        df = df.drop(columns=cols_to_drop)

    # ── 2. Drop rows with any null ────────────────────────────────────────────
    if drop_na:
        before = len(df)
        df     = df.dropna()
        logs.append(ImputationLog(
            column="<all>", strategy="drop_rows",
            n_imputed=0, null_pct_before=0,
            fill_value=None,
            warning=f"Dropped {before - len(df):,} rows containing any null",
        ))
        return df, logs

    # ── 3. Nothing to impute ──────────────────────────────────────────────────
    if df.isna().sum().sum() == 0:
        return df, logs

    # ── 4. Impute per-column ──────────────────────────────────────────────────
    df, logs = _impute_columns(df, strategy, fill_value, knn_k, logs)

    return df, logs


# ---------------------------------------------------------------------------
# Column-level dispatch
# ---------------------------------------------------------------------------

def _impute_columns(df: pd.DataFrame, strategy: str,
                    fill_value: Optional[str], knn_k: int,
                    logs: List[ImputationLog]) -> tuple[pd.DataFrame, List[ImputationLog]]:

    null_counts = df.isna().sum()
    cols_with_nulls = null_counts[null_counts > 0].index.tolist()

    # Separate columns by type
    numeric_cols  = [c for c in cols_with_nulls
                     if pd.api.types.is_numeric_dtype(df[c])
                     and not pd.api.types.is_bool_dtype(df[c])]
    datetime_cols = [c for c in cols_with_nulls
                     if pd.api.types.is_datetime64_any_dtype(df[c])]
    other_cols    = [c for c in cols_with_nulls
                     if c not in numeric_cols and c not in datetime_cols]

    n = max(len(df), 1)

    # ── Datetime: ffill + bfill ───────────────────────────────────────────────
    for col in datetime_cols:
        before = int(df[col].isna().sum())
        df[col] = df[col].ffill().bfill()
        after   = int(df[col].isna().sum())
        logs.append(ImputationLog(
            column=col, strategy="ffill_bfill",
            n_imputed=before - after,
            null_pct_before=round(before / n * 100, 1),
        ))

    # ── Numeric ───────────────────────────────────────────────────────────────
    if numeric_cols:
        df, logs = _impute_numeric(df, numeric_cols, strategy,
                                    fill_value, knn_k, n, logs)

    # ── Categorical / string ──────────────────────────────────────────────────
    for col in other_cols:
        before = int(df[col].isna().sum())
        null_pct = before / n * 100
        strat = _select_cat_strategy(strategy, null_pct)

        if strat == "mode":
            mode = df[col].mode()
            val  = mode.iloc[0] if len(mode) > 0 else ("Unknown" if fill_value is None else fill_value)
            df[col] = df[col].fillna(val)
            logs.append(ImputationLog(
                column=col, strategy="mode",
                n_imputed=before, null_pct_before=round(null_pct, 1),
                fill_value=str(val),
            ))
        elif strat == "knn" and _HAS_SKLEARN:
            df, log = _knn_cat(df, col, knn_k, before, null_pct)
            logs.append(log)
        else:
            fv = fill_value if fill_value is not None else "Unknown"
            df[col] = df[col].fillna(fv)
            warn = f"High missingness ({null_pct:.0f}%): constant fill used" if null_pct > 40 else None
            logs.append(ImputationLog(
                column=col, strategy="constant",
                n_imputed=before, null_pct_before=round(null_pct, 1),
                fill_value=str(fv), warning=warn,
            ))

    return df, logs


def _impute_numeric(df: pd.DataFrame, numeric_cols: List[str],
                     strategy: str, fill_value: Optional[str],
                     knn_k: int, n: int,
                     logs: List[ImputationLog]) -> tuple[pd.DataFrame, List[ImputationLog]]:

    null_rates = {c: df[c].isna().mean() for c in numeric_cols}

    # Categorise by missing rate
    low_miss  = [c for c in numeric_cols if null_rates[c] < 0.05]
    mid_miss  = [c for c in numeric_cols if 0.05 <= null_rates[c] <= 0.40]
    high_miss = [c for c in numeric_cols if null_rates[c] > 0.40]

    # Low missing: median fill (fast, good enough)
    for col in low_miss:
        before = int(df[col].isna().sum())
        med    = df[col].median()
        df[col] = df[col].fillna(med)
        logs.append(ImputationLog(
            column=col, strategy="median",
            n_imputed=before, null_pct_before=round(null_rates[col] * 100, 1),
            fill_value=str(round(float(med), 4)),
        ))

    # Mid missing: KNN imputation
    if mid_miss and _HAS_SKLEARN and strategy in ("auto", "knn"):
        # Need at least 2 numeric columns for KNN
        all_num = df.select_dtypes(include="number").columns.tolist()
        if len(all_num) >= 2:
            try:
                df, knn_logs = _knn_numeric(df, mid_miss, all_num, knn_k, n, null_rates)
                logs.extend(knn_logs)
            except Exception as exc:
                # Fallback to median
                for col in mid_miss:
                    before  = int(df[col].isna().sum())
                    med     = df[col].median()
                    df[col] = df[col].fillna(med)
                    logs.append(ImputationLog(
                        column=col, strategy="median_fallback",
                        n_imputed=before,
                        null_pct_before=round(null_rates[col] * 100, 1),
                        fill_value=str(round(float(med), 4)),
                        warning=f"KNN failed ({exc}); used median",
                    ))
        else:
            for col in mid_miss:
                before  = int(df[col].isna().sum())
                med     = df[col].median()
                df[col] = df[col].fillna(med)
                logs.append(ImputationLog(
                    column=col, strategy="median",
                    n_imputed=before,
                    null_pct_before=round(null_rates[col] * 100, 1),
                    fill_value=str(round(float(med), 4)),
                ))
    elif mid_miss:
        for col in mid_miss:
            before  = int(df[col].isna().sum())
            med     = df[col].median()
            df[col] = df[col].fillna(med)
            logs.append(ImputationLog(
                column=col, strategy="median",
                n_imputed=before,
                null_pct_before=round(null_rates[col] * 100, 1),
                fill_value=str(round(float(med), 4)),
            ))

    # High missing: constant or user-supplied fill with warning
    for col in high_miss:
        before = int(df[col].isna().sum())
        if fill_value is not None:
            try:
                fv = float(fill_value)
            except (ValueError, TypeError):
                fv = df[col].median()
        else:
            fv = df[col].median()
        df[col] = df[col].fillna(fv)
        logs.append(ImputationLog(
            column=col, strategy="median_high_miss",
            n_imputed=before,
            null_pct_before=round(null_rates[col] * 100, 1),
            fill_value=str(round(float(fv), 4)),
            warning=f"Column has {null_rates[col]*100:.0f}% missing — imputed values unreliable",
        ))

    return df, logs


def _knn_numeric(df: pd.DataFrame, target_cols: List[str],
                  all_num_cols: List[str], k: int, n: int,
                  null_rates: Dict) -> tuple[pd.DataFrame, List[ImputationLog]]:
    from sklearn.impute import KNNImputer

    logs: List[ImputationLog] = []
    sub   = df[all_num_cols].copy()

    # Convert nullable int to float for sklearn
    for col in sub.columns:
        if str(sub[col].dtype).startswith(("Int", "UInt")):
            sub[col] = sub[col].astype("float64")

    before_counts = {c: int(sub[c].isna().sum()) for c in target_cols}

    imputer = KNNImputer(n_neighbors=min(k, max(1, n - 1)),
                          weights="distance")
    arr = imputer.fit_transform(sub.values)
    sub_imp = pd.DataFrame(arr, columns=sub.columns, index=sub.index)

    for col in target_cols:
        df[col] = sub_imp[col]
        logs.append(ImputationLog(
            column=col, strategy=f"knn_k{k}",
            n_imputed=before_counts[col],
            null_pct_before=round(null_rates[col] * 100, 1),
        ))

    return df, logs


def _knn_cat(df: pd.DataFrame, col: str, k: int, before: int,
              null_pct: float) -> tuple[pd.DataFrame, ImputationLog]:
    """KNN imputation for a single categorical column via ordinal encoding."""
    try:
        from sklearn.impute import KNNImputer
        from sklearn.preprocessing import OrdinalEncoder

        num_cols = df.select_dtypes(include="number").columns.tolist()
        if not num_cols:
            raise ValueError("no numeric context columns")

        enc   = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        cat_e = enc.fit_transform(df[[col]].astype(str))
        cat_e[cat_e == -1] = np.nan

        X = np.hstack([df[num_cols].values.astype(float), cat_e])
        imp = KNNImputer(n_neighbors=min(k, len(df) - 1))
        X_imp = imp.fit_transform(X)

        cat_imp = np.round(X_imp[:, -1]).astype(int)
        cat_imp = np.clip(cat_imp, 0, len(enc.categories_[0]) - 1)
        df[col] = enc.inverse_transform(cat_imp.reshape(-1, 1))[:, 0]

        return df, ImputationLog(
            column=col, strategy=f"knn_cat_k{k}",
            n_imputed=before, null_pct_before=round(null_pct, 1),
        )
    except Exception as exc:
        mode = df[col].mode()
        fv   = mode.iloc[0] if len(mode) > 0 else "Unknown"
        df[col] = df[col].fillna(fv)
        return df, ImputationLog(
            column=col, strategy="mode_fallback",
            n_imputed=before, null_pct_before=round(null_pct, 1),
            fill_value=str(fv), warning=f"KNN cat failed: {exc}",
        )


def _select_cat_strategy(strategy: str, null_pct: float) -> str:
    if strategy == "constant":
        return "constant"
    if strategy == "knn" and _HAS_SKLEARN:
        return "knn"
    if null_pct < 5:
        return "mode"
    if null_pct <= 40 and _HAS_SKLEARN:
        return "knn"
    return "constant"
