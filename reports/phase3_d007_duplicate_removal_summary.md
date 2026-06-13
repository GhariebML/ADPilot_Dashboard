# D007 — Exact-text duplicate removal

- Input analytical dataset: **39,712 × 53**
- Exact-text duplicate components: **642**
- Canonical listings retained: **642**
- Duplicate extras excluded: **938**
- Output analytical dataset: **38,774 × 53**
- Raw dataset: unchanged at **39,713 × 53**

Canonical selection order: latest `listed_date`, then greater field completeness, higher `images_count`, later `scraped_at`, and stable source order. All excluded records and their canonical listing IDs are preserved in the audit files.
