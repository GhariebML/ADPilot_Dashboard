# Phase 2 — Standalone Data Dictionary Proposal

No AdPilot schema integration is included. Recommended names are proposals only; the raw CSV remains unchanged.

Columns documented: **53**

| Original | Proposed name | Role | Sensitive | Leakage | Main quality concern |
|---|---|---|---|---|---|
| `listing_id` | `listing_id` | identifier | No | High | One value is missing; verify uniqueness and stability across reposts |
| `internal_id` | `internal_listing_id` | identifier | No | High | Stored as numeric-like; should likely be string to avoid precision/format issues |
| `category` | `transaction_category` | filter/dimension | No | Low | Meaning overlaps with offering_type and must be reconciled |
| `listing_type` | `listing_record_type` | filter/dimension | No | Medium | May mix true property listings with recommendation records |
| `detail_url` | `listing_url` | identifier/audit | No | High | One missing value; URL may expose duplicates and source patterns |
| `property_type` | `property_type` | feature/dimension | No | Low | One missing value; rare and inconsistent categories may exist |
| `offering_type` | `offering_type` | filter/dimension | No | Low | One missing value; overlaps with category |
| `completion_status` | `completion_status` | feature/dimension | No | Low | About half missing; category meaning requires validation |
| `title` | `listing_title` | text/audit | No | High | May include target price, payment amounts, project names, and promotional language |
| `price_egp` | `price_egp` | target/measure | No | Target | Mixes sale totals and rental periodic prices until scope is filtered; extreme values require review |
| `price_period` | `price_period` | filter/dimension | No | Medium | Must be consistent with transaction category |
| `price_currency` | `price_currency` | filter/dimension | No | Low | Constant EGP in current file; still useful for validation |
| `location_full` | `location_full` | feature/dimension | No | Medium | May combine district, compound, project, and free-form labels |
| `city` | `market_region` | feature/dimension | No | Low | Mixed geographic levels; requires an approved mapping before use as formal administrative geography |
| `town` | `town` | feature/dimension | No | Low | Geographic hierarchy and relationship to district need confirmation |
| `district` | `district` | feature/dimension | No | Low | Missing values and spelling variants require review |
| `subdistrict` | `submarket_or_compound` | feature/dimension | No | Medium | Mixed entity types; later classification is required before compound/project aggregation |
| `lat` | `latitude` | feature/geography | No | Low | Coordinate accuracy and whether it represents listing or area centroid are unknown |
| `lon` | `longitude` | feature/geography | No | Low | Coordinate accuracy and whether it represents listing or area centroid are unknown |
| `bedrooms` | `bedrooms` | feature/measure | No | Low | Mixed numeric and textual values require parsing rules |
| `bathrooms` | `bathrooms` | feature/measure | No | Low | Mixed numeric and textual values require parsing rules |
| `area_value` | `area_sqm` | feature/measure | No | Low | Requires validation for zero, impossible, bulk, and land values |
| `area_unit` | `area_unit` | validation/dimension | No | Low | Constant sqm in current file; retain for audit |
| `furnished` | `furnished_status` | feature/dimension | No | Low | Substantial missingness; values may be boolean or category-like |
| `listing_level` | `listing_level` | feature/audit | No | Medium | Business meaning is source-specific and may change over time |
| `is_premium` | `is_premium` | feature/audit | No | Medium | May reflect advertiser spend rather than property value |
| `is_verified` | `is_verified` | feature/audit | No | Medium | Constant false, possibly a scraper limitation |
| `is_featured` | `is_featured` | feature/audit | No | Medium | May reflect paid placement rather than intrinsic property value |
| `is_new_construction` | `is_new_construction` | feature/audit | No | Medium | Constant false, likely unreliable or extraction-limited |
| `is_direct_from_dev` | `is_direct_from_developer` | feature/audit | No | Medium | Constant false, likely unreliable or extraction-limited |
| `is_exclusive` | `is_exclusive` | feature/audit | No | Medium | Constant false, likely unreliable or extraction-limited |
| `listed_date` | `listed_at` | feature/time | No | Medium | Meaning may be first publication or latest refresh; stale/repost behavior possible |
| `images_count` | `image_count` | feature/quality | No | Medium | May proxy listing effort rather than property value |
| `has_view_360` | `has_360_view` | feature/quality | No | Medium | Rare value and possibly source-specific |
| `video_url` | `video_url` | feature/audit | No | Medium | Almost entirely missing; URL content not inspected |
| `reference` | `advertiser_reference` | identifier/audit | No | High | May identify duplicate or related listings; format is inconsistent |
| `rera` | `rera_reference` | identifier/audit | Potentially | High | Completely empty in current file |
| `description` | `listing_description` | text/feature | Potentially | High | May expose target price, installment terms, contacts, and duplicates |
| `amenities` | `amenities_text` | text/feature | No | Medium | Likely semi-structured; parsing and category normalization needed |
| `payment_method` | `payment_method` | feature/dimension | No | Low | About half missing and likely linked to sale/rent scope |
| `agent_id` | `agent_id` | identifier/audit | Yes | High | May cause memorization and broker-specific bias |
| `agent_name` | `agent_name` | audit/contact | Yes | High | Personal information; spelling and entity resolution issues possible |
| `agent_email` | `agent_email` | audit/contact | Yes | High | Personal contact information; should not be used for modeling |
| `agent_is_super` | `agent_is_super` | audit/feature | No | Medium | Constant false and business meaning unclear |
| `agent_languages` | `agent_languages` | audit/feature | Potentially | Medium | High missingness; unrelated to intrinsic property value |
| `broker_id` | `broker_id` | identifier/audit | Yes | High | May cause entity memorization and market-channel bias |
| `broker_name` | `broker_name` | audit/entity | Potentially | High | May be useful for duplicate analysis but risky as predictive feature |
| `broker_email` | `broker_email` | audit/contact | Yes | High | Sensitive business contact information; should not be used for modeling |
| `broker_phone` | `broker_phone` | audit/contact | Yes | High | Sensitive and stored numeric-like; formatting may be corrupted |
| `contact_phone` | `contact_phone` | audit/contact | Yes | High | Sensitive and stored numeric-like; useful only for duplicate audit |
| `contact_whatsapp` | `contact_whatsapp` | audit/contact | Yes | High | Sensitive and stored numeric-like; useful only for duplicate audit |
| `contact_email` | `contact_email` | audit/contact | Yes | High | Sensitive contact information; useful only for duplicate audit |
| `scraped_at` | `scraped_at` | audit/time | No | Low | Represents collection time, not market event time |