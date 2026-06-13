# Egyptian Real Estate Market Intelligence — Final EDA and Model Report

## Executive summary

The final analytical dataset contains **18,053** unique, in-scope sale listings across the six approved residential property types. The median asking price is **EGP 9,290,000**, while the median price per square metre is **EGP 60,505**.

Among towns with at least 50 usable listings, **Sidi Abdel Rahman** has the highest observed median price per square metre at approximately **EGP 118,846**. By property type, **Villa** has the highest median price per square metre.

The selected CatBoost fair-price model achieved an untouched test-set **MAE of EGP 5,798,372**, **RMSE of EGP 16,613,957**, and **R² of 0.446**. Predictions estimate the typical market asking price, not a guaranteed transaction value.

## Data preparation decisions

- Kept repeated `reference` values but excluded the field from uniqueness logic and modeling.
- Restricted the analytical scope to sale listings, EGP, and Apartment, Villa, Townhouse, Duplex, Chalet, and Twin House.
- Recovered suspicious 1–20 sqm areas from explicit units in title/description where reliable: **1,204 records**.
- Excluded only unrecoverable implausible areas, probable deposit/installment amounts, and extreme values outside conservative plausibility bounds. Every exclusion is preserved in `data/audit/model_scope_exclusions.parquet`.
- Missing categorical values are represented as `Unknown`; numeric missingness is handled inside model pipelines or natively by CatBoost.
- No developer ranking was produced because the source has no explicit developer field.

## Market tiers

Tiers use a hybrid score: **60% absolute price-per-sqm percentile and 40% percentile within the same town/property type**. This prevents all properties in expensive locations from automatically becoming luxury while preserving absolute market positioning.

| market_tier   |   listings |   median_price |   median_price_per_sqm |   median_area |    share |
|:--------------|-----------:|---------------:|-----------------------:|--------------:|---------:|
| Affordable    |       4073 |      4.134e+06 |                25477.7 |           170 | 0.225613 |
| Mid-market    |       4960 |      7.5e+06   |                48566.3 |           167 | 0.274747 |
| Upper-mid     |       4881 |      1.238e+07 |                73414.3 |           175 | 0.270371 |
| Luxury        |       4139 |      1.9e+07   |               115556   |           174 | 0.229269 |

## Model design

- Primary target: `log(price_egp)` for stable errors across a highly skewed market.
- Leakage controls: no price-derived feature, title, description, ID, reference, URL, contact, agent, or broker field is used as a predictor.
- Split: group-aware train/validation/test separation by named compound, falling back to town and district.
- Model selection used validation results only. The untouched test set was evaluated once after final refitting.

| Model           |   Validation MAE |   Validation RMSE |   Validation R2 |   Median absolute error |   Validation MAPE |   Training seconds |   Prediction seconds | Explainability   | Deployment complexity   | Main strengths                                                              | Main weaknesses                                             |
|:----------------|-----------------:|------------------:|----------------:|------------------------:|------------------:|-------------------:|---------------------:|:-----------------|:------------------------|:----------------------------------------------------------------------------|:------------------------------------------------------------|
| Random Forest   |      7.84104e+06 |       2.32576e+07 |        0.504511 |             2.71106e+06 |          0.482299 |         1.06851    |           0.0573503  | Medium           | High                    | Captures nonlinearities and interactions                                    | Large model and weaker extrapolation                        |
| Ridge           |      8.07017e+06 |       2.47288e+07 |        0.439844 |             2.89639e+06 |          0.557971 |         6.92659    |           0.104447   | High             | Medium                  | Stable, fast, interpretable linear benchmark                                | Limited nonlinear interaction capture                       |
| CatBoost        |      8.33179e+06 |       2.53287e+07 |        0.412334 |             2.5449e+06  |          0.422416 |         6.32648    |           0.0478501  | Medium           | Medium                  | Strong native handling of high-cardinality categories and nonlinear effects | More complex deployment and explanation than linear models  |
| Median baseline |      1.06518e+07 |       2.92219e+07 |        0.217793 |             3.83734e+06 |          0.804563 |         0.00501635 |           0.00141231 | High             | Low                     | Transparent and robust location/property-type benchmark                     | Cannot model nonlinear interactions or unit characteristics |

## Explainability and business use

SHAP values identify predictive associations, not causal price drivers. The dashboard presents global importance and local explanations, fair-price intervals, comparable listings, and out-of-distribution warnings.

## Limitations

- Listing prices are asking prices, not completed transaction prices.
- Some listings advertise deposits or installments despite a sale-price field; conservative rules reduce but cannot eliminate this contamination.
- Developer and formal project identifiers are absent.
- Geographic labels mix formal and marketing geographies.
- Market conditions may shift after the March 2026 scrape.
- Confidence intervals are empirical validation residual ranges, not legal or appraisal guarantees.
