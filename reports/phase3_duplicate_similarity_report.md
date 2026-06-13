# Phase 3 — Stricter Duplicate Similarity Analysis

No records were removed. Analytical dataset remains **39,712 × 53**.

## Candidate blocking rule

Listings were compared only when these fields matched exactly: `category, property_type, location_full, bedrooms, bathrooms, area_value, price_egp, agent_id, furnished, payment_method`.

This reproduced the earlier broad candidate set: **1,597 groups**, **3,918 rows**, and **2,321 potential extra rows**.

## Stricter results

| Rule | Components | Rows involved | Potential extras | Dataset share |
|---|---:|---:|---:|---:|
| Exact normalized title + description | 642 | 1,580 | 938 | 2.362% |
| Strong text similarity | 929 | 2,249 | 1,320 | 3.324% |
| Balanced text similarity | 1,036 | 2,505 | 1,469 | 3.699% |
| Broad core-signature rule | 1,597 | 3,918 | 2,321 | 5.845% |

## Rule definitions

- **Exact text:** normalized titles and descriptions are identical.
- **Strong:** exact title with description similarity ≥90, exact description with title similarity ≥90, both similarities ≥95, or a specific matching reference with both similarities ≥80.
- **Balanced:** both similarities ≥90, one exact text field with the other ≥80, or a specific matching reference with one text similarity ≥70.

## Interpretation

Exact-text matches are the safest repost candidates, but identical units in the same project can still share templated text. Strong and balanced rules capture more likely reposts but increase the risk of merging genuinely separate units. The broad rule is unsuitable for automatic removal without further evidence.
