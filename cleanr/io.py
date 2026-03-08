"""
io.py
-----
Production I/O layer.

Supports: CSV, TSV, TXT, JSON, JSONL, XLSX, XLS, Parquet (optional).
Handles: compressed inputs (gz, bz2, xz, zip), chunked CSV reading,
         multi-sheet Excel, encoding fallbacks.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import pandas as pd

from cleanr.detector import FormatInfo

warnings.filterwarnings("ignore")

try:
    import pyarrow  # noqa
    _HAS_PARQUET = True
except ImportError:
    try:
        import fastparquet  # noqa
        _HAS_PARQUET = True
    except ImportError:
        _HAS_PARQUET = False

_FALLBACK_ENCODINGS = ["utf-8-sig", "utf-8", "latin1", "cp1252", "iso-8859-1"]


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load(filepath: Path, fmt_info: FormatInfo,
         chunk_size: Optional[int] = None,
         quick: bool = False) -> pd.DataFrame:
    """Load a data file into a DataFrame."""
    fmt = fmt_info.fmt

    if fmt in ("csv", "tsv", "txt"):
        return _load_delimited(filepath, fmt_info, chunk_size, quick)
    if fmt in ("xlsx", "xls"):
        return _load_excel(filepath, fmt_info)
    if fmt == "json":
        return _load_json(filepath, fmt_info)
    if fmt == "jsonl":
        return _load_jsonl(filepath, fmt_info)
    if fmt == "parquet":
        return _load_parquet(filepath)
    # Fallback: try delimited
    return _load_delimited(filepath, fmt_info, chunk_size, quick)


def save(df: pd.DataFrame, output_path: Path,
         fmt: str = "csv", encoding: str = "utf-8",
         sheet_name: str = "CleanR"):
    """Save a DataFrame to file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out = _prepare_for_export(df)

    if fmt in ("csv", "txt"):
        out.to_csv(output_path, index=False, encoding=encoding)
    elif fmt == "tsv":
        out.to_csv(output_path, index=False, sep="\t", encoding=encoding)
    elif fmt == "xlsx":
        out.to_excel(output_path, index=False, sheet_name=sheet_name,
                     engine="openpyxl")
    elif fmt == "xls":
        out.to_excel(output_path, index=False, sheet_name=sheet_name)
    elif fmt == "json":
        out.to_json(output_path, orient="records", indent=2, force_ascii=False)
    elif fmt == "jsonl":
        out.to_json(output_path, orient="records", lines=True, force_ascii=False)
    elif fmt == "parquet":
        if not _HAS_PARQUET:
            raise RuntimeError(
                "Parquet output requires pyarrow: pip install pyarrow"
            )
        out.to_parquet(output_path, index=False)
    else:
        out.to_csv(output_path, index=False, encoding=encoding)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_delimited(filepath: Path, fmt_info: FormatInfo,
                     chunk_size: Optional[int], quick: bool) -> pd.DataFrame:
    compression = _compression_arg(fmt_info.compression)
    encodings   = ([fmt_info.encoding] if fmt_info.encoding
                   else _FALLBACK_ENCODINGS)
    last_err    = None

    for enc in encodings:
        try:
            kwargs = dict(
                sep          = fmt_info.delimiter,
                encoding     = enc,
                quotechar    = fmt_info.quotechar,
                dtype        = str if quick else None,
                low_memory   = False,
                on_bad_lines = "skip",
                header       = 0 if fmt_info.has_header else None,
                compression  = compression,
            )
            if chunk_size and chunk_size > 0:
                chunks = list(pd.read_csv(filepath, chunksize=chunk_size, **kwargs))
                return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
            else:
                return pd.read_csv(filepath, **kwargs)
        except Exception as exc:
            last_err = exc
            continue

    raise ValueError(f"Cannot read '{filepath}'. Last error: {last_err}") from last_err


def _load_excel(filepath: Path, fmt_info: FormatInfo) -> pd.DataFrame:
    try:
        engine     = "openpyxl" if filepath.suffix.lower() == ".xlsx" else None
        sheet_name = fmt_info.sheet_name or 0
        df         = pd.read_excel(filepath, sheet_name=sheet_name, engine=engine)
        if isinstance(df, dict):
            # Multiple sheets — concatenate with a sheet_name column
            parts = []
            for sname, sdf in df.items():
                sdf["_sheet"] = str(sname)
                parts.append(sdf)
            df = pd.concat(parts, ignore_index=True)
        return df
    except Exception as exc:
        raise ValueError(f"Cannot read Excel '{filepath}': {exc}") from exc


def _load_json(filepath: Path, fmt_info: FormatInfo) -> pd.DataFrame:
    enc = fmt_info.encoding or "utf-8"
    for orient in ("records", "split", "index", None):
        try:
            return pd.read_json(filepath, orient=orient, encoding=enc)
        except Exception:
            continue
    raise ValueError(f"Cannot parse JSON file '{filepath}'")


def _load_jsonl(filepath: Path, fmt_info: FormatInfo) -> pd.DataFrame:
    try:
        return pd.read_json(filepath, lines=True,
                            encoding=fmt_info.encoding or "utf-8")
    except Exception as exc:
        raise ValueError(f"Cannot read JSONL '{filepath}': {exc}") from exc


def _load_parquet(filepath: Path) -> pd.DataFrame:
    if not _HAS_PARQUET:
        raise RuntimeError(
            "Parquet input requires pyarrow: pip install pyarrow"
        )
    return pd.read_parquet(filepath)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compression_arg(comp: Optional[str]):
    if comp in ("gz", "bz2", "xz", "zip"):
        return comp
    return None


def _prepare_for_export(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten pandas-specific types for broad compatibility."""
    out = df.copy()
    for col in out.columns:
        dtype_str = str(out[col].dtype)
        if dtype_str == "category":
            out[col] = out[col].astype(object)
        elif dtype_str == "boolean":
            out[col] = out[col].astype(object)
        elif dtype_str.startswith(("Int", "UInt")):
            out[col] = out[col].astype(object)
    return out
