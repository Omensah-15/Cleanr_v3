# CleanR v3 — Intelligent Data Cleaning Engine

CleanR is a fast, production-grade data cleaning engine for the command line. Built for data analysts, data engineers, and anyone who works with messy real-world files — it handles CSV, Excel, JSON, Parquet, and more. One command auto-detects your format and encoding, removes duplicates, imputes missing values using KNN, infers and corrects column types, flags invalid emails and phone numbers, detects outliers with Isolation Forest, scores your data quality across five dimensions, and produces a full audit trail with SHA-256 integrity fingerprints. Stop hunting through spreadsheet menus or writing one-off cleaning scripts — CleanR does it all in seconds.

For full installation instructions, feature reference, and working examples, visit the guide:
**[Full Usage Guide](https://omensah-15.github.io/CleanR-v3/)**

---

## Demo

<div align="center">
  <img src="https://github.com/Omensah-15/CleanR-v3/blob/2e38d9a7fe8e1defb8abf56168a4c59671d38ac2/docs/demo2.gif" alt="Demo" width="800">
</div>

- Before: [messy_data.csv](https://github.com/Omensah-15/CleanR-v3/blob/60202d590931d30e0fa7504f0283ac953a6f5510/docs/messy_data.csv)
- After: [clean_data.csv](https://github.com/Omensah-15/CleanR-v3/blob/60202d590931d30e0fa7504f0283ac953a6f5510/docs/clean_data.csv)

---

## Quick Start

```bash
pip install -e .
cleanr data.csv
```

---

## Features

| Capability | Details |
|---|---|
| **Auto format detection** | CSV, TSV, TXT, JSON, JSONL, Excel (.xlsx/.xls), Parquet |
| **Encoding detection** | UTF-8, UTF-8-BOM, Latin-1, CP1252, ISO-8859-1, UTF-16 |
| **Schema inference** | Detects int, float, bool, datetime, category, string |
| **Semantic types** | Email, phone, URL, UUID, IP, currency, percentage |
| **Intelligent pipeline** | Normalize → Trim → Dedup → Handle nulls → Type coerce → Validate → Optimize |
| **Smart imputation** | KNN for mid-missing numerics, mode for categoricals, ffill for datetimes |
| **Outlier detection** | Isolation Forest + IQR consensus — flag or remove |
| **Format validation** | Flags invalid emails, phones, URLs, UUIDs in dedicated columns |
| **Memory optimization** | Auto-downcasts numerics, promotes categories |
| **Quality scoring** | 0–100 weighted score across Completeness, Uniqueness, Validity, Consistency, Accuracy |
| **JSON quality report** | Full pre/post profiles, schema, issues, performance metrics |
| **Audit trail** | Timestamped JSON log of every action taken |
| **Fingerprinting** | SHA-256 of raw file bytes + DataFrame content for integrity |
| **Plugin architecture** | Extend with custom `CleanrPlugin` subclasses |
| **Multiple output formats** | Write to CSV, TSV, JSON, JSONL, XLSX, Parquet |
| **Chunked streaming** | Handles files larger than RAM with `--chunk N` |

---

## Author

**Mensah Obed**
[Email](mailto:heavenzlebron7@gmail.com) · [LinkedIn](https://www.linkedin.com/in/obed-mensah-87001237b)
