# Phase 1 — Environment and File Inspection

Inspection was non-destructive. No cleaning, filtering, imputation, standardization, feature engineering, or modeling was performed.

## Files

| File | Format | Size | Role |
|---|---:|---:|---|
| Egypt real estate 2026.zip | ZIP | 15,873,606 bytes | Uploaded source archive; preserved unchanged |
| propertyfinder.csv | CSV | 72,440,854 bytes | Sole dataset inside archive; extracted to data/raw |
| Pasted markdown.md | Markdown | about 36 KB | Project instructions; not an AdPilot source-code repository |

Archive relationship: the ZIP contains exactly one file, `propertyfinder.csv`; there are no workbook sheets or relational companion tables.

## CSV structure

- Shape: **39,713 rows × 53 columns**
- Encoding: UTF-8 with BOM (`utf-8-sig`)
- Delimiter: comma
- Malformed CSV records: **0**
- Listed-date range: 2022-06-08T22:00:31Z to 2026-03-04T23:42:53Z
- Scrape window: 2026-03-04T14:20:33.281007 to 2026-03-05T00:43:04.172513

## Column inventory

### Identifiers

| Column | Inferred storage type | Missing | Unique |
|---|---:|---:|---:|
| `listing_id` | string | 0.003% | 39,712 |
| `internal_id` | float | 0.003% | 39,712 |
| `reference` | string | 0.005% | 39,644 |
| `agent_id` | float | 0.003% | 3,400 |
| `broker_id` | float | 0.003% | 1,074 |

### Listing classification

| Column | Inferred storage type | Missing | Unique |
|---|---:|---:|---:|
| `category` | string | 0.000% | 2 |
| `listing_type` | string | 0.000% | 2 |
| `property_type` | string | 0.003% | 19 |
| `offering_type` | string | 0.003% | 2 |
| `completion_status` | string | 50.235% | 4 |
| `listing_level` | string | 0.003% | 3 |

### Price

| Column | Inferred storage type | Missing | Unique |
|---|---:|---:|---:|
| `price_egp` | float | 0.003% | 4,957 |
| `price_period` | string | 0.003% | 2 |
| `price_currency` | string | 0.000% | 1 |
| `payment_method` | string | 49.792% | 3 |

### Location

| Column | Inferred storage type | Missing | Unique |
|---|---:|---:|---:|
| `location_full` | string | 0.003% | 1,959 |
| `city` | string | 0.003% | 12 |
| `town` | string | 0.003% | 67 |
| `district` | string | 0.874% | 519 |
| `subdistrict` | string | 17.020% | 1,134 |
| `lat` | float | 0.003% | 1,921 |
| `lon` | float | 0.003% | 1,941 |

### Property attributes

| Column | Inferred storage type | Missing | Unique |
|---|---:|---:|---:|
| `bedrooms` | string | 0.088% | 9 |
| `bathrooms` | string | 0.096% | 8 |
| `area_value` | float | 0.003% | 734 |
| `area_unit` | string | 0.000% | 1 |
| `furnished` | string | 21.114% | 3 |

### Flags/media

| Column | Inferred storage type | Missing | Unique |
|---|---:|---:|---:|
| `is_premium` | boolean | 0.000% | 2 |
| `is_verified` | boolean | 0.000% | 1 |
| `is_featured` | boolean | 0.000% | 2 |
| `is_new_construction` | boolean | 0.000% | 1 |
| `is_direct_from_dev` | boolean | 0.000% | 1 |
| `is_exclusive` | boolean | 0.000% | 1 |
| `images_count` | integer | 0.000% | 31 |
| `has_view_360` | boolean | 0.000% | 2 |
| `video_url` | string | 99.491% | 154 |
| `rera` | string | 100.000% | 0 |

### Text

| Column | Inferred storage type | Missing | Unique |
|---|---:|---:|---:|
| `title` | string | 0.003% | 32,700 |
| `description` | string | 0.003% | 35,312 |
| `amenities` | string | 0.083% | 17,995 |

### Agent/broker/contact

| Column | Inferred storage type | Missing | Unique |
|---|---:|---:|---:|
| `agent_name` | string | 0.020% | 2,652 |
| `agent_email` | string | 0.003% | 2,732 |
| `agent_is_super` | boolean | 0.000% | 1 |
| `agent_languages` | string | 76.904% | 30 |
| `broker_name` | string | 0.003% | 776 |
| `broker_email` | string | 0.511% | 770 |
| `broker_phone` | float | 0.511% | 1,063 |
| `contact_phone` | float | 0.003% | 3,061 |
| `contact_whatsapp` | float | 0.003% | 540 |
| `contact_email` | string | 0.003% | 2,732 |

### Time/URL

| Column | Inferred storage type | Missing | Unique |
|---|---:|---:|---:|
| `detail_url` | url | 0.003% | 39,712 |
| `listed_date` | string | 0.003% | 36,255 |
| `scraped_at` | string | 0.000% | 4,736 |

## Important preliminary observations

- `price_currency` is EGP for all 39,713 records and `area_unit` is sqm for all records.
- The dataset is split almost evenly between rent (19,910) and buy (19,803).
- Explicit developer, project, and compound columns are absent. Project/compound names may be embedded in `location_full`, `district`, `subdistrict`, `title`, or `description`, but this has not been inferred or standardized.
- `rera` is completely empty. Several boolean fields are constant false, including `is_verified`, `is_new_construction`, `is_direct_from_dev`, `is_exclusive`, and `agent_is_super`.
- Contact and email columns contain sensitive personal/business contact information and should be excluded from normal analytical exports and modeling.
- Raw ranges warrant later quality review: price EGP 1,300 to 1,000,000,000; area 1 to 96,600 sqm.
- `listed_date` reaches back to June 2022 while scraping occurred March 4–5, 2026, so listing staleness needs later analysis.

## Raw-file integrity

- ZIP SHA-256: `e5ac2d37344d63fe0b4f8e5ea1b10112c3f898874c1bffd5a15c82aa792dd17e`
- CSV SHA-256: `974ef8b9d03308b4ea74b6719079ab782720948d6b20340df5b1ef0cdbfe3292`