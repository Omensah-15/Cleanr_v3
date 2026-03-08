"""
engine.py
---------
CleanR v3 pipeline orchestrator.

Pipeline:
  detect -> load -> pre-profile -> infer schema -> normalize -> trim ->
  dedup -> handle missing (KNN/smart) -> type coercion -> format validation ->
  outlier detection -> constant column drop -> column ops -> memory optimize ->
  custom plugins -> post-profile -> save -> report
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

warnings.filterwarnings("ignore")

from cleanr import VERSION
from cleanr.audit import AuditLog, make_fingerprint
from cleanr.detector import detect
from cleanr.io import load, save
from cleanr import profiler as prof
from cleanr import schema as sch
from cleanr import report as rep
from cleanr.plugins import (
    NormalizeColumnsPlugin, TrimWhitespacePlugin, RemoveDuplicatesPlugin,
    HandleMissingPlugin, TypeCoercionPlugin, FormatValidatorPlugin,
    MemoryOptimizePlugin, SelectColumnsPlugin, SplitColumnPlugin,
    RenameColumnsPlugin, AddColumnsPlugin,
    OutlierDetectorPlugin, ConstantColumnDropperPlugin, RowValidatorPlugin,
    CleanrPlugin,
)

DEFAULT_CHUNK_SIZE = 100_000


class CleanREngine:
    """Production data cleaning engine."""

    def __init__(self, verbose: bool = True):
        self.verbose         = verbose
        self._custom_plugins: List[CleanrPlugin] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def register_plugin(self, plugin: CleanrPlugin):
        """Append a custom plugin to the end of the pipeline."""
        self._custom_plugins.append(plugin)

    def clean(self, input_path: Path, output_path: Path, **opts) -> Dict[str, Any]:
        """
        Run the full cleaning pipeline.

        Core opts
        ---------
        encoding            str   Force encoding (e.g. 'utf-8', 'latin1')
        chunk_size          int   Rows per chunk for large files
        quick               bool  Skip schema inference & memory optimisation

        Cleaning
        ---------
        normalize           bool  Normalise column names (default True)
        trim                bool  Trim whitespace (default True)
        dedup               bool  Remove exact duplicates (default True)
        drop_constant       bool  Drop constant columns (default True)
        auto_types          bool  Coerce types from schema (default True)
        validate_formats    bool  Flag invalid semantic values (default True)
        detect_outliers     bool  Isolation Forest + IQR outlier detection
        outlier_method      str   'flag' | 'remove'
        impute_strategy     str   'auto' | 'knn' | 'median' | 'mode' | 'constant'
        fill_value          str   Constant fill value
        drop_na             bool  Drop rows with any null
        drop_col_threshold  float Drop columns >= this null fraction

        Column ops
        ----------
        keep        list[str]
        drop        list[str]
        rename      dict str->str
        add         dict str->str  (new_col -> source_col)
        split       list[dict]     [{column, new_columns, delimiter}]

        Rules
        -----
        rules       list[dict]  [{column, type, value/pattern/values, action}]

        Output
        ------
        output_format   str  csv|tsv|json|jsonl|xlsx|parquet|auto
        report_path     str
        audit_path      str
        no_fingerprint  bool
        """
        start = time.time()
        audit = AuditLog()

        self._log(f"\n  CleanR v{VERSION}")
        self._log(f"  Input   {input_path}")
        self._log(f"  Output  {output_path}\n")

        # ── 1. Detect ─────────────────────────────────────────────────────────
        fmt_info = detect(input_path, force_encoding=opts.get("encoding"))
        self._log(f"  Format:   {fmt_info.fmt.upper()}"
                  f"  Encoding: {fmt_info.encoding}"
                  f"  Delimiter: {repr(fmt_info.delimiter)}"
                  + (f"  Compression: {fmt_info.compression}" if fmt_info.compression else ""))

        # ── 2. Load ───────────────────────────────────────────────────────────
        t  = time.time()
        df = load(input_path, fmt_info,
                  chunk_size=opts.get("chunk_size", DEFAULT_CHUNK_SIZE),
                  quick=opts.get("quick", False))
        self._log(f"  Loaded:   {len(df):,} rows x {len(df.columns)} cols  "
                  f"({time.time()-t:.2f}s)")

        if df.empty:
            raise ValueError("Input file contains no data.")

        # ── 3. Pre-clean profile ──────────────────────────────────────────────
        t          = time.time()
        pre_profile = prof.profile(df)
        self._log(f"  Profile:  quality {pre_profile.quality_label} "
                  f"{pre_profile.quality_score:.0f}/100  ({time.time()-t:.2f}s)")

        # ── 4. Input fingerprint ──────────────────────────────────────────────
        input_fp = None
        if not opts.get("no_fingerprint"):
            t        = time.time()
            input_fp = make_fingerprint(input_path, df)
            self._log(f"  Fingerprint input: {input_fp.data_hash[:24]}...  "
                      f"({time.time()-t:.2f}s)")

        # ── 5. Schema inference ───────────────────────────────────────────────
        if not opts.get("quick"):
            t      = time.time()
            schema = sch.infer_schema(df)
            self._log(f"  Schema:   {len(schema)} columns inferred  ({time.time()-t:.2f}s)")
        else:
            schema = {}

        # ── 6. Pipeline ───────────────────────────────────────────────────────
        self._log("\n  Pipeline:")
        df = self._run_pipeline(df, schema, audit, opts)

        # ── 7. Post-clean profile ─────────────────────────────────────────────
        t            = time.time()
        post_profile = prof.profile(df)
        self._log(f"\n  Post-clean: quality {post_profile.quality_label} "
                  f"{post_profile.quality_score:.0f}/100  ({time.time()-t:.2f}s)")

        # ── 8. Save ───────────────────────────────────────────────────────────
        out_fmt = opts.get("output_format", "auto")
        if out_fmt == "auto":
            out_fmt = fmt_info.fmt
        t = time.time()
        save(df, output_path, fmt=out_fmt,
             encoding=opts.get("encoding", "utf-8"))
        self._log(f"  Saved:    {out_fmt.upper()} -> {output_path}  "
                  f"({time.time()-t:.2f}s)")

        # ── 9. Output fingerprint ─────────────────────────────────────────────
        output_fp = None
        if not opts.get("no_fingerprint") and output_path.exists():
            output_fp = make_fingerprint(output_path, df)
            self._log(f"  Fingerprint output: {output_fp.data_hash[:24]}...")

        # ── 10. Reports ───────────────────────────────────────────────────────
        elapsed = round(time.time() - start, 3)

        terminal_rpt = rep.render_terminal(
            pre_profile, post_profile, audit,
            input_fp, output_fp, elapsed,
            input_path, output_path,
        )

        json_rpt = rep.build_json_report(
            pre_profile, post_profile, schema, audit,
            input_fp, output_fp, elapsed,
            input_path, output_path,
        )

        report_path = Path(opts.get("report_path") or
                           str(output_path.with_suffix("")) + ".report.json")
        audit_path  = Path(opts.get("audit_path")  or
                           str(output_path.with_suffix("")) + ".audit.json")

        rep.save_json_report(json_rpt, report_path)
        audit.save(audit_path)

        if self.verbose:
            print(terminal_rpt)

        return {
            "success":             True,
            "elapsed":             elapsed,
            "pre_profile":         pre_profile,
            "post_profile":        post_profile,
            "schema":              schema,
            "input_fingerprint":   input_fp,
            "output_fingerprint":  output_fp,
            "report_path":         str(report_path),
            "audit_path":          str(audit_path),
            "output_path":         str(output_path),
            "json_report":         json_rpt,
        }

    # ── Pipeline ──────────────────────────────────────────────────────────────

    def _run_pipeline(self, df: pd.DataFrame, schema: Dict,
                       audit: AuditLog, opts: Dict) -> pd.DataFrame:

        def run(plugin: CleanrPlugin) -> pd.DataFrame:
            nonlocal df
            t  = time.time()
            df = plugin.run(df)
            elapsed = time.time() - t
            actions = plugin.report()
            audit.record(plugin.name, actions)
            if self.verbose and actions:
                for a in actions:
                    sym = "!" if a.startswith("WARNING") else "+"
                    print(f"    {sym} [{plugin.name}] {a}")
            elif self.verbose and elapsed > 0.3:
                print(f"    . [{plugin.name}] completed ({elapsed:.2f}s)")
            return df

        # Step 1 — Normalise column names
        if opts.get("normalize", True):
            df = run(NormalizeColumnsPlugin())
            # Re-infer schema with new names
            if schema:
                schema = sch.infer_schema(df)

        # Step 2 — Trim whitespace
        if opts.get("trim", True):
            df = run(TrimWhitespacePlugin())

        # Step 3 — Remove duplicates
        if opts.get("dedup", True):
            df = run(RemoveDuplicatesPlugin())

        # Step 4 — Drop constant columns (before type work)
        if opts.get("drop_constant", True) and not opts.get("quick"):
            df = run(ConstantColumnDropperPlugin())

        # Step 5 — Column selection
        if opts.get("keep") or opts.get("drop"):
            df = run(SelectColumnsPlugin(config={
                "keep": opts.get("keep"),
                "drop": opts.get("drop"),
            }))

        # Step 6 — Smart missing value imputation
        df = run(HandleMissingPlugin(config={
            "strategy":           opts.get("impute_strategy", "auto"),
            "fill_value":         opts.get("fill_value"),
            "drop_na":            opts.get("drop_na", False),
            "drop_col_threshold": opts.get("drop_col_threshold"),
            "knn_k":              opts.get("knn_k", 5),
        }))

        # Step 7 — Type coercion
        if opts.get("auto_types", True) and schema:
            df = run(TypeCoercionPlugin(schema=schema))

        # Step 8 — Format validation
        if opts.get("validate_formats", True) and schema:
            df = run(FormatValidatorPlugin(schema=schema))

        # Step 9 — Outlier detection
        if opts.get("detect_outliers", False):
            df = run(OutlierDetectorPlugin(config={
                "method":        opts.get("outlier_method", "flag"),
                "contamination": opts.get("outlier_contamination", 0.02),
            }))

        # Step 10 — Row validation rules
        if opts.get("rules"):
            df = run(RowValidatorPlugin(config={
                "rules":  opts["rules"],
                "action": opts.get("rule_action", "flag"),
            }))

        # Step 11 — Split columns
        if opts.get("split"):
            df = run(SplitColumnPlugin(config={"splits": opts["split"]}))

        # Step 12 — Add columns
        if opts.get("add"):
            df = run(AddColumnsPlugin(config={"add_map": opts["add"]}))

        # Step 13 — Rename columns
        if opts.get("rename"):
            df = run(RenameColumnsPlugin(config={"rename_map": opts["rename"]}))

        # Step 14 — Memory optimisation
        if not opts.get("quick"):
            df = run(MemoryOptimizePlugin())

        # Step 15 — Custom plugins
        for plugin in self._custom_plugins:
            df = run(plugin)

        return df

    def _log(self, msg: str):
        if self.verbose:
            print(msg)
