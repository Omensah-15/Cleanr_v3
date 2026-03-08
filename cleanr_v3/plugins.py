"""
plugins.py
----------
Production-grade plugin system.

Base class + 14 built-in plugins:
  NormalizeColumns, TrimWhitespace, RemoveDuplicates, HandleMissing,
  TypeCoercion, FormatValidator, MemoryOptimize, SelectColumns,
  SplitColumn, RenameColumns, AddColumns,
  OutlierDetector, ConstantColumnDropper, RowValidator
"""
from __future__ import annotations

import re
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import RobustScaler
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class CleanrPlugin(ABC):
    name:        str = "base"
    description: str = ""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config  = config or {}
        self.actions: List[str] = []

    @abstractmethod
    def run(self, df: pd.DataFrame) -> pd.DataFrame: ...

    def log(self, msg: str):
        self.actions.append(msg)

    def report(self) -> List[str]:
        return list(self.actions)


# ---------------------------------------------------------------------------
# NormalizeColumns
# ---------------------------------------------------------------------------

class NormalizeColumnsPlugin(CleanrPlugin):
    name        = "normalize_columns"
    description = "Normalizes column names to lowercase snake_case."

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        seen:     set       = set()
        new_cols: List[str] = []
        changed               = 0

        for i, col in enumerate(df.columns, start=1):
            original = str(col).strip() if col is not None else ""
            if not original or original.lower() in ("nan", "none", ""):
                cleaned = f"column_{i}"
            else:
                cleaned = original.lower()
                cleaned = re.sub(r"[\s\-\.\/\\:;,!?()[\]{}'\"]+", "_", cleaned)
                cleaned = re.sub(r"[^\w]",                        "",  cleaned)
                cleaned = re.sub(r"_+",                           "_", cleaned).strip("_")
                if not cleaned or cleaned.isdigit():
                    cleaned = f"column_{i}"

            base, count = cleaned, 1
            while cleaned in seen:
                cleaned = f"{base}_{count}"
                count  += 1
            seen.add(cleaned)
            new_cols.append(cleaned)
            if cleaned != original:
                changed += 1

        df.columns = new_cols
        if changed:
            self.log(f"Normalized {changed}/{len(new_cols)} column names to snake_case")
        return df


# ---------------------------------------------------------------------------
# TrimWhitespace
# ---------------------------------------------------------------------------

class TrimWhitespacePlugin(CleanrPlugin):
    name        = "trim_whitespace"
    description = "Strips leading/trailing whitespace; normalizes internal whitespace."

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        normalize_internal = self.config.get("normalize_internal", True)
        obj_cols           = df.select_dtypes(include="object").columns.tolist()
        changed            = 0

        for col in obj_cols:
            before = df[col].copy()
            s      = df[col].where(df[col].isna(), df[col].astype(str).str.strip())
            if normalize_internal:
                s = s.where(s.isna(), s.str.replace(r"\s{2,}", " ", regex=True))
            # Replace empty strings with NaN
            s = s.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA,
                            "none": pd.NA, "null": pd.NA, "NULL": pd.NA,
                            "N/A": pd.NA, "n/a": pd.NA, "NA": pd.NA, "NaN": pd.NA})
            df[col] = s
            if not before.equals(df[col]):
                changed += 1

        if changed:
            self.log(f"Trimmed and normalised whitespace in {changed} string columns")
        return df


# ---------------------------------------------------------------------------
# RemoveDuplicates
# ---------------------------------------------------------------------------

class RemoveDuplicatesPlugin(CleanrPlugin):
    name        = "remove_duplicates"
    description = "Drops exact and near-duplicate rows."

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        before       = len(df)
        keep         = self.config.get("keep", "first")
        subset       = self.config.get("subset")  # list of columns or None

        df           = df.drop_duplicates(keep=keep, subset=subset)
        removed      = before - len(df)
        if removed:
            self.log(f"Removed {removed:,} duplicate rows ({before:,} -> {len(df):,})")
        else:
            self.log("No duplicate rows found")
        return df


# ---------------------------------------------------------------------------
# HandleMissing  (delegates to imputer.py)
# ---------------------------------------------------------------------------

class HandleMissingPlugin(CleanrPlugin):
    name        = "handle_missing"
    description = "Imputes or removes missing values using per-column strategies."

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        from cleanr.imputer import impute

        df, logs = impute(
            df,
            strategy           = self.config.get("strategy", "auto"),
            fill_value         = self.config.get("fill_value"),
            drop_na            = self.config.get("drop_na", False),
            drop_col_threshold = self.config.get("drop_col_threshold"),
            knn_k              = self.config.get("knn_k", 5),
        )
        for log in logs:
            if log.warning:
                self.log(f"WARNING [{log.column}] {log.warning}")
            elif log.n_imputed > 0:
                detail = f" (fill={log.fill_value})" if log.fill_value else ""
                self.log(f"Imputed {log.n_imputed:,} nulls in '{log.column}' "
                         f"via {log.strategy}{detail}")
        return df


# ---------------------------------------------------------------------------
# TypeCoercion
# ---------------------------------------------------------------------------

class TypeCoercionPlugin(CleanrPlugin):
    name        = "type_coercion"
    description = "Coerces columns to inferred dtypes from schema analysis."

    def __init__(self, schema=None, config=None):
        super().__init__(config)
        self.schema = schema or {}

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        for col, col_schema in self.schema.items():
            if col not in df.columns:
                continue
            target = col_schema.inferred_dtype
            if target == str(df[col].dtype):
                continue  # already correct
            try:
                df = self._coerce(df, col, target)
            except Exception as exc:
                self.log(f"WARNING [{col}] cannot coerce to {target}: {exc}")
        return df

    def _coerce(self, df: pd.DataFrame, col: str, target: str) -> pd.DataFrame:
        if target in ("int64", "Int64"):
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(r"[\$£€¥,\s%]", "", regex=True),
                errors="coerce",
            ).astype("Int64")
            self.log(f"Coerced '{col}' -> Int64")

        elif target == "float64":
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(r"[\$£€¥,\s%]", "", regex=True),
                errors="coerce",
            ).astype("float64")
            self.log(f"Coerced '{col}' -> float64")

        elif target == "bool":
            df[col] = _coerce_bool(df[col])
            self.log(f"Coerced '{col}' -> bool")

        elif target == "datetime64[ns]":
            fmt = getattr(self.schema.get(col), "datetime_format", None)
            if fmt and fmt != "inferred":
                df[col] = pd.to_datetime(df[col], format=fmt, errors="coerce")
            else:
                df[col] = pd.to_datetime(df[col], errors="coerce")
            self.log(f"Coerced '{col}' -> datetime64[ns]")

        elif target == "category":
            df[col] = df[col].astype("category")
            self.log(f"Coerced '{col}' -> category")

        return df


# ---------------------------------------------------------------------------
# FormatValidator
# ---------------------------------------------------------------------------

class FormatValidatorPlugin(CleanrPlugin):
    name        = "format_validator"
    description = "Validates semantic types and flags invalid values."

    def __init__(self, schema=None, config=None):
        super().__init__(config)
        self.schema = schema or {}

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        from cleanr.schema import _PATTERNS

        for col, col_schema in self.schema.items():
            sem = col_schema.semantic_type
            if not sem or sem not in _PATTERNS or col not in df.columns:
                continue
            pat = _PATTERNS[sem]
            try:
                str_vals = df[col].dropna().astype(str)
                mask     = df[col].notna() & ~df[col].astype(str).str.match(pat)
                n_bad    = int(mask.sum())
                if n_bad:
                    flag_col        = f"_invalid_{col}"
                    df[flag_col]    = mask
                    pct             = n_bad / max(len(df), 1) * 100
                    self.log(f"Flagged {n_bad:,} ({pct:.1f}%) invalid {sem} values "
                             f"in '{col}' -> '{flag_col}'")
            except Exception as exc:
                self.log(f"WARNING [{col}] format validation failed: {exc}")
        return df


# ---------------------------------------------------------------------------
# OutlierDetector  (Isolation Forest + IQR consensus)
# ---------------------------------------------------------------------------

class OutlierDetectorPlugin(CleanrPlugin):
    name        = "outlier_detector"
    description = "Flags outlier rows using Isolation Forest + IQR consensus."

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        method      = self.config.get("method", "flag")   # flag | remove
        contamination = self.config.get("contamination", 0.02)
        min_rows    = self.config.get("min_rows", 200)

        num_cols = [c for c in df.select_dtypes(include="number").columns
                    if not c.startswith("_invalid_")]

        if not num_cols or len(df) < min_rows:
            self.log("Outlier detection skipped: insufficient numeric data or rows")
            return df

        # IQR flags per column
        iqr_mask = pd.Series(False, index=df.index)
        for col in num_cols:
            s  = df[col].dropna()
            q1 = s.quantile(0.25)
            q3 = s.quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                lo = q1 - 3.0 * iqr
                hi = q3 + 3.0 * iqr
                iqr_mask |= (df[col].notna() & ((df[col] < lo) | (df[col] > hi)))

        # Isolation Forest
        if_mask = pd.Series(False, index=df.index)
        if _HAS_SKLEARN and len(df) >= min_rows:
            try:
                X = df[num_cols].copy()
                for c in X.columns:
                    X[c] = X[c].fillna(X[c].median())
                scaler = RobustScaler()
                X_s    = scaler.fit_transform(X)
                iso    = IsolationForest(
                    contamination=contamination,
                    random_state=42, n_jobs=-1,
                )
                preds  = iso.fit_predict(X_s)
                if_mask = pd.Series(preds == -1, index=df.index)
            except Exception as exc:
                self.log(f"WARNING Isolation Forest failed: {exc}")

        # Consensus: flagged by BOTH methods
        consensus = iqr_mask & if_mask
        n_flagged = int(consensus.sum())

        if n_flagged == 0:
            self.log("No consensus outliers detected")
            return df

        pct = n_flagged / len(df) * 100
        if method == "remove":
            df = df[~consensus].reset_index(drop=True)
            self.log(f"Removed {n_flagged:,} ({pct:.1f}%) outlier rows "
                     "(IQR + Isolation Forest consensus)")
        else:
            df["_is_outlier"] = consensus
            self.log(f"Flagged {n_flagged:,} ({pct:.1f}%) outlier rows -> '_is_outlier' column")

        return df


# ---------------------------------------------------------------------------
# ConstantColumnDropper
# ---------------------------------------------------------------------------

class ConstantColumnDropperPlugin(CleanrPlugin):
    name        = "constant_column_dropper"
    description = "Drops columns that carry no information (single unique value)."

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        dropped = []
        for col in df.columns:
            if df[col].nunique(dropna=True) <= 1:
                dropped.append(col)
        if dropped:
            df = df.drop(columns=dropped)
            self.log(f"Dropped {len(dropped)} constant/empty columns: {dropped}")
        return df


# ---------------------------------------------------------------------------
# MemoryOptimize
# ---------------------------------------------------------------------------

class MemoryOptimizePlugin(CleanrPlugin):
    name        = "memory_optimize"
    description = "Downcasts numerics, promotes low-cardinality strings to category."

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        before_mb = df.memory_usage(deep=True).sum() / 1e6

        # Object -> category if cardinality ratio < 0.5
        for col in df.select_dtypes(include="object").columns:
            ratio = df[col].nunique() / max(len(df), 1)
            if ratio < 0.50:
                df[col] = df[col].astype("category")

        # Downcast integers
        for col in df.select_dtypes(include=["int64", "int32", "int16"]).columns:
            try:
                df[col] = pd.to_numeric(df[col], downcast="integer", errors="ignore")
            except Exception:
                pass

        # Downcast floats
        for col in df.select_dtypes(include=["float64"]).columns:
            try:
                df[col] = pd.to_numeric(df[col], downcast="float", errors="ignore")
            except Exception:
                pass

        after_mb = df.memory_usage(deep=True).sum() / 1e6
        saved    = max(before_mb - after_mb, 0.0)
        self.log(f"Memory: {before_mb:.2f} MB -> {after_mb:.2f} MB "
                 f"(saved {saved:.2f} MB, {saved/max(before_mb,0.001)*100:.0f}%)")
        return df


# ---------------------------------------------------------------------------
# SelectColumns
# ---------------------------------------------------------------------------

class SelectColumnsPlugin(CleanrPlugin):
    name        = "select_columns"
    description = "Keeps or drops named columns."

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        keep = self.config.get("keep")
        drop = self.config.get("drop")

        if keep:
            missing   = [c for c in keep if c not in df.columns]
            available = [c for c in keep if c in df.columns]
            if missing:
                self.log(f"WARNING Columns not found (skipped): {missing}")
            df = df[available]
            self.log(f"Kept {len(available)} columns")

        elif drop:
            to_drop = [c for c in drop if c in df.columns]
            not_found = [c for c in drop if c not in df.columns]
            if not_found:
                self.log(f"WARNING Columns to drop not found: {not_found}")
            if to_drop:
                df = df.drop(columns=to_drop)
                self.log(f"Dropped {len(to_drop)} columns: {to_drop}")
        return df


# ---------------------------------------------------------------------------
# SplitColumn
# ---------------------------------------------------------------------------

class SplitColumnPlugin(CleanrPlugin):
    name        = "split_column"
    description = "Splits a column by delimiter into multiple named columns."

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        for spec in self.config.get("splits", []):
            col      = spec["column"]
            new_cols = spec["new_columns"]
            delim    = spec["delimiter"]

            if col not in df.columns:
                self.log(f"WARNING Column '{col}' not found for split")
                continue

            n      = len(new_cols)
            splits = df[col].astype(str).str.split(re.escape(delim), n=n - 1, expand=True)
            for i, nc in enumerate(new_cols):
                df[nc] = splits[i] if i in splits.columns else pd.NA

            self.log(f"Split '{col}' on '{delim}' -> {new_cols}")
        return df


# ---------------------------------------------------------------------------
# RenameColumns
# ---------------------------------------------------------------------------

class RenameColumnsPlugin(CleanrPlugin):
    name        = "rename_columns"
    description = "Renames columns using an old->new mapping."

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        rename_map = self.config.get("rename_map", {})
        missing    = [k for k in rename_map if k not in df.columns]
        if missing:
            self.log(f"WARNING Columns not found for rename: {missing}")
        valid = {k: v for k, v in rename_map.items() if k in df.columns}
        if valid:
            df = df.rename(columns=valid)
            self.log(f"Renamed {len(valid)} columns: "
                     + ", ".join(f"'{k}'->'{v}'" for k, v in valid.items()))
        return df


# ---------------------------------------------------------------------------
# AddColumns
# ---------------------------------------------------------------------------

class AddColumnsPlugin(CleanrPlugin):
    name        = "add_columns"
    description = "Adds new columns as copies of existing columns."

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        add_map = self.config.get("add_map", {})
        for new_col, src in add_map.items():
            if src not in df.columns:
                self.log(f"WARNING Source column '{src}' not found")
                continue
            df[new_col] = df[src].copy()
            self.log(f"Added '{new_col}' as copy of '{src}'")
        return df


# ---------------------------------------------------------------------------
# RowValidator
# ---------------------------------------------------------------------------

class RowValidatorPlugin(CleanrPlugin):
    name        = "row_validator"
    description = "Validates rows against configurable rules and flags violations."

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        rules = self.config.get("rules", [])
        if not rules:
            return df

        violation_mask = pd.Series(False, index=df.index)

        for rule in rules:
            col   = rule.get("column")
            rtype = rule.get("type")

            if not col or col not in df.columns:
                self.log(f"WARNING Rule references unknown column: {col}")
                continue

            try:
                if rtype == "min":
                    v    = rule["value"]
                    mask = df[col].notna() & (df[col] < v)
                    violation_mask |= mask
                    self.log(f"Rule min({col}>={v}): {int(mask.sum())} violations")

                elif rtype == "max":
                    v    = rule["value"]
                    mask = df[col].notna() & (df[col] > v)
                    violation_mask |= mask
                    self.log(f"Rule max({col}<={v}): {int(mask.sum())} violations")

                elif rtype == "not_null":
                    mask = df[col].isna()
                    violation_mask |= mask
                    self.log(f"Rule not_null({col}): {int(mask.sum())} violations")

                elif rtype == "regex":
                    pat  = re.compile(rule["pattern"])
                    mask = df[col].notna() & ~df[col].astype(str).str.match(pat)
                    violation_mask |= mask
                    self.log(f"Rule regex({col}): {int(mask.sum())} violations")

                elif rtype == "allowed_values":
                    allowed = set(rule["values"])
                    mask    = df[col].notna() & ~df[col].isin(allowed)
                    violation_mask |= mask
                    self.log(f"Rule allowed_values({col}): {int(mask.sum())} violations")

            except Exception as exc:
                self.log(f"WARNING Rule on '{col}' failed: {exc}")

        action = self.config.get("action", "flag")
        n_viol = int(violation_mask.sum())

        if n_viol > 0:
            if action == "remove":
                df = df[~violation_mask].reset_index(drop=True)
                self.log(f"Removed {n_viol:,} rows violating validation rules")
            else:
                df["_rule_violation"] = violation_mask
                self.log(f"Flagged {n_viol:,} rule violations -> '_rule_violation'")
        else:
            self.log("All rows passed validation rules")

        return df


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

BUILTIN_PLUGINS: Dict[str, type] = {
    p.name: p for p in [
        NormalizeColumnsPlugin, TrimWhitespacePlugin, RemoveDuplicatesPlugin,
        HandleMissingPlugin, TypeCoercionPlugin, FormatValidatorPlugin,
        MemoryOptimizePlugin, SelectColumnsPlugin, SplitColumnPlugin,
        RenameColumnsPlugin, AddColumnsPlugin,
        OutlierDetectorPlugin, ConstantColumnDropperPlugin, RowValidatorPlugin,
    ]
}


def get_plugin(name: str) -> type:
    if name not in BUILTIN_PLUGINS:
        raise ValueError(f"Unknown plugin '{name}'. Available: {sorted(BUILTIN_PLUGINS)}")
    return BUILTIN_PLUGINS[name]


# ---------------------------------------------------------------------------
# Util
# ---------------------------------------------------------------------------

def _coerce_bool(series: pd.Series) -> pd.Series:
    TRUE_VALS  = {"true", "yes", "1", "y", "t", "on", "enabled"}
    FALSE_VALS = {"false", "no", "0", "n", "f", "off", "disabled"}
    s      = series.astype(str).str.strip().str.lower()
    result = pd.Series(pd.NA, index=series.index, dtype="boolean")
    result[s.isin(TRUE_VALS)]  = True
    result[s.isin(FALSE_VALS)] = False
    return result
