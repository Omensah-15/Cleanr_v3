"""
detector.py
-----------
Production-grade file format and encoding detection.

Combines magic-byte inspection, extension hints, CSV dialect sniffing,
and multi-pass encoding detection with chardet confidence scoring.
"""
from __future__ import annotations

import csv
import io
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

try:
    import chardet
    _HAS_CHARDET = True
except ImportError:
    _HAS_CHARDET = False

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

SUPPORTED_FORMATS = frozenset(
    {"csv", "tsv", "txt", "json", "jsonl", "xlsx", "xls", "parquet"}
)

# Raw magic bytes -> format
_MAGIC: list[tuple[bytes, str]] = [
    (b"PK\x03\x04",       "xlsx"),   # ZIP (xlsx, also docx etc — disambiguate by ext)
    (b"\xd0\xcf\x11\xe0", "xls"),    # OLE2
    (b"PAR1",             "parquet"),
    (b"\x89PNG",          "_image"), # guard: skip image files
    (b"\xff\xd8\xff",     "_image"),
]

_ENCODINGS_ORDERED = [
    "utf-8-sig", "utf-8", "utf-16", "utf-16-le", "utf-16-be",
    "latin1", "cp1252", "iso-8859-1", "cp850",
]


@dataclass
class FormatInfo:
    fmt: str                         # csv | tsv | json | jsonl | xlsx | xls | parquet | txt
    encoding: str = "utf-8"
    delimiter: str = ","
    has_header: bool = True
    quotechar: str = '"'
    sheet_name: Optional[str] = None
    compression: Optional[str] = None   # gz | bz2 | xz | zip | None
    issues: List[str] = field(default_factory=list)


def detect(filepath: Path, force_encoding: Optional[str] = None) -> FormatInfo:
    """Auto-detect format, encoding, compression, and CSV dialect."""
    info = FormatInfo(fmt="csv")

    # ── compression ──────────────────────────────────────────────────────────
    name = filepath.name.lower()
    for ext, comp in ((".gz", "gz"), (".bz2", "bz2"), (".xz", "xz"), (".zip", "zip")):
        if name.endswith(ext):
            info.compression = comp
            name = name[: -len(ext)]
            break

    # ── extension hint ────────────────────────────────────────────────────────
    suffix = Path(name).suffix.lstrip(".").lower()
    if suffix in SUPPORTED_FORMATS:
        info.fmt = suffix
    elif suffix in ("txt", "dat", "log"):
        info.fmt = "txt"

    # ── magic-byte override for binary formats ────────────────────────────────
    if info.compression is None:
        raw_head = _read_bytes(filepath, 8)
        if raw_head:
            for magic, fmt in _MAGIC:
                if raw_head.startswith(magic):
                    if fmt == "_image":
                        info.issues.append("file_appears_to_be_binary_image")
                    elif fmt == "xlsx" and suffix == "xls":
                        pass  # trust extension for OLE vs ZIP ambiguity
                    else:
                        info.fmt = fmt
                    break

    # ── encoding detection ────────────────────────────────────────────────────
    if force_encoding:
        info.encoding = force_encoding
    elif info.fmt not in ("xlsx", "xls", "parquet"):
        info.encoding = _detect_encoding(filepath, info.compression)

    # ── CSV/TSV dialect sniffing ──────────────────────────────────────────────
    if info.fmt in ("csv", "tsv", "txt"):
        delim, has_header, quotechar = _sniff_dialect(
            filepath, info.encoding, info.compression
        )
        info.delimiter  = "\t" if info.fmt == "tsv" else delim
        info.has_header = has_header
        info.quotechar  = quotechar

    return info


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_bytes(path: Path, n: int) -> bytes:
    try:
        with open(path, "rb") as fh:
            return fh.read(n)
    except OSError:
        return b""


def _detect_encoding(path: Path, compression: Optional[str],
                     sample_bytes: int = 131_072) -> str:
    """Return the most probable encoding, validated by a decode attempt."""
    raw = _sample_bytes(path, compression, sample_bytes)
    if not raw:
        return "utf-8"

    # BOM detection first — definitive
    if raw[:3] == b"\xef\xbb\xbf":
        return "utf-8-sig"
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return "utf-16"

    # chardet heuristic
    if _HAS_CHARDET:
        result = chardet.detect(raw)
        enc = (result.get("encoding") or "utf-8").lower()
        confidence = result.get("confidence", 0)
        if confidence >= 0.80:
            enc = _normalise_encoding(enc)
            if _validate_encoding(raw, enc):
                return enc

    # Ordered fallback
    for enc in _ENCODINGS_ORDERED:
        if _validate_encoding(raw, enc):
            return enc

    return "utf-8"


def _sample_bytes(path: Path, compression: Optional[str], n: int) -> bytes:
    try:
        if compression == "gz":
            import gzip
            with gzip.open(path, "rb") as fh:
                return fh.read(n)
        elif compression == "bz2":
            import bz2
            with bz2.open(path, "rb") as fh:
                return fh.read(n)
        elif compression == "xz":
            import lzma
            with lzma.open(path, "rb") as fh:
                return fh.read(n)
        else:
            with open(path, "rb") as fh:
                return fh.read(n)
    except Exception:
        return b""


def _validate_encoding(raw: bytes, enc: str) -> bool:
    try:
        raw.decode(enc)
        return True
    except (UnicodeDecodeError, LookupError):
        return False


def _normalise_encoding(enc: str) -> str:
    return (enc
            .replace("windows-1252", "cp1252")
            .replace("iso-8859-1",   "latin1")
            .replace("ascii",        "utf-8"))


def _sniff_dialect(path: Path, encoding: str, compression: Optional[str],
                   sample_lines: int = 30) -> tuple[str, bool, str]:
    """Return (delimiter, has_header, quotechar) for a text file."""
    candidates = [",", "\t", ";", "|", "^", "~"]
    try:
        lines = _read_text_lines(path, encoding, compression, sample_lines)
        if not lines:
            return ",", True, '"'
        sample = "\n".join(lines)

        sniffer = csv.Sniffer()
        try:
            dialect = sniffer.sniff(sample, delimiters="".join(candidates))
            has_header = True
            try:
                has_header = sniffer.has_header(sample)
            except Exception:
                pass
            return dialect.delimiter, has_header, dialect.quotechar or '"'
        except csv.Error:
            pass

        # Fallback: most frequent candidate across first lines
        counts = {d: sample.count(d) for d in candidates}
        best = max(counts, key=counts.get)
        return (best if counts[best] > 0 else ","), True, '"'

    except Exception:
        return ",", True, '"'


def _read_text_lines(path: Path, encoding: str, compression: Optional[str],
                     n: int) -> List[str]:
    import io as _io
    try:
        if compression == "gz":
            import gzip
            fh = gzip.open(path, "rt", encoding=encoding, errors="replace")
        elif compression == "bz2":
            import bz2
            fh = bz2.open(path, "rt", encoding=encoding, errors="replace")
        elif compression == "xz":
            import lzma
            fh = lzma.open(path, "rt", encoding=encoding, errors="replace")
        else:
            fh = open(path, encoding=encoding, errors="replace")
        with fh:
            return [fh.readline() for _ in range(n)]
    except Exception:
        return []
