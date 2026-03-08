# CleanR v3 — Intelligent Data Cleaning Engine
A command-line data processing engine that automatically cleans, profiles, validates, and optimizes datasets for analytics and machine learning workflows.
---

## Demo
<div align="center">
  <img src="https://github.com/Omensah-15/CleanR-v3/blob/2e38d9a7fe8e1defb8abf56168a4c59671d38ac2/docs/demo2.gif" alt="Demo" width="800">
</div>

- Before: [messy_data.csv](https://github.com/Omensah-15/CleanR-v3/blob/60202d590931d30e0fa7504f0283ac953a6f5510/docs/messy_data.csv)
- After: [clean_data.csv](https://github.com/Omensah-15/CleanR-v3/blob/60202d590931d30e0fa7504f0283ac953a6f5510/docs/clean_data.csv)

---

## Features

| Capability | Details |
|---|---|
| **Auto format detection** | CSV, TSV, TXT, JSON, JSONL, Excel (.xlsx/.xls), Parquet |
| **Encoding detection** | UTF-8, UTF-8-BOM, Latin-1, CP1252, ISO-8859-1, UTF-16 |
| **Schema inference** | Detects int, float, bool, datetime, category, string |
| **Semantic types** | Email, phone, URL, UUID, IP, currency, percentage |
| **Intelligent pipeline** | Normalize → Trim → Dedup → Handle nulls → Type coerce → Validate → Optimize |
| **Format validation** | Flags invalid emails, phones, URLs, UUIDs in dedicated columns |
| **Memory optimization** | Auto-downcasts numerics, promotes categories |
| **Quality scoring** | 0–100 score with label (Excellent / Good / Fair / Poor / Critical) |
| **JSON quality report** | Full pre/post profiles, schema, issues, performance metrics |
| **Audit trail** | Timestamped JSON log of every action taken |
| **Fingerprinting** | SHA-256 of raw file bytes + DataFrame content for integrity |
| **Plugin architecture** | Extend with custom `CleanrPlugin` subclasses |
| **Multiple output formats** | Write to CSV, TSV, JSON, JSONL, XLSX, Parquet |
| **Chunked streaming** | Handles files larger than RAM with `--chunk N` |

---

## 👨‍💻 Author

**Mensah Obed**
[Email](mailto:heavenzlebron7@gmail.com) 
[LinkedIn](https://www.linkedin.com/in/obed-mensah-87001237b)
