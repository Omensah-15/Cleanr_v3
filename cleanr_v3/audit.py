"""
audit.py
--------
Append-only timestamped audit log + SHA-256 dataset fingerprinting.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class AuditEntry:
    elapsed_s: float
    plugin:    str
    action:    str
    level:     str = "INFO"    # INFO | WARNING | ERROR


@dataclass
class Fingerprint:
    algorithm:  str
    file_hash:  str
    data_hash:  str
    row_count:  int
    col_count:  int
    created_at: float


class AuditLog:
    def __init__(self):
        self._start   = time.time()
        self._entries: List[AuditEntry] = []

    def record(self, plugin: str, actions: List[str]):
        elapsed = round(time.time() - self._start, 4)
        for action in actions:
            level = "WARNING" if action.startswith("WARNING") else "INFO"
            self._entries.append(AuditEntry(elapsed, plugin, action, level))

    def entries(self) -> List[AuditEntry]:
        return list(self._entries)

    def to_list(self) -> List[Dict]:
        return [asdict(e) for e in self._entries]

    def warnings(self) -> List[str]:
        return [e.action for e in self._entries if e.level == "WARNING"]

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({
                "total_elapsed_s": round(time.time() - self._start, 3),
                "entry_count": len(self._entries),
                "entries": self.to_list(),
            }, fh, indent=2)


def fingerprint_file(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as fh:
        for chunk in iter(lambda: fh.read(65_536), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint_dataframe(df) -> str:
    """Deterministic SHA-256 of DataFrame content."""
    import pandas as pd
    h = hashlib.sha256()
    h.update("|".join(str(c) for c in df.columns).encode("utf-8"))
    h.update(f"|rows={len(df)}|cols={len(df.columns)}".encode())
    for start in range(0, len(df), 10_000):
        chunk = df.iloc[start: start + 10_000]
        h.update(chunk.to_csv(index=False).encode("utf-8", errors="replace"))
    return h.hexdigest()


def make_fingerprint(filepath: Path, df) -> Fingerprint:
    return Fingerprint(
        algorithm  = "sha256",
        file_hash  = fingerprint_file(filepath),
        data_hash  = fingerprint_dataframe(df),
        row_count  = len(df),
        col_count  = len(df.columns),
        created_at = time.time(),
    )
