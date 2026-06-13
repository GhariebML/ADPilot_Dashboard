# Final Quality Assurance

| check                                                                               | status   | detail                                                         |
|:------------------------------------------------------------------------------------|:---------|:---------------------------------------------------------------|
| File exists: notebooks/adpilot_egypt_real_estate_market_intelligence.ipynb          | PASS     | 15750                                                          |
| File exists: notebooks/adpilot_egypt_real_estate_market_intelligence.executed.ipynb | PASS     | 783789                                                         |
| File exists: dashboard/app.py                                                       | PASS     | 13599                                                          |
| File exists: requirements.txt                                                       | PASS     | 189                                                            |
| File exists: README.md                                                              | PASS     | 1912                                                           |
| File exists: src/pipeline.py                                                        | PASS     | 51619                                                          |
| File exists: data/raw/propertyfinder.csv                                            | PASS     | 72440854                                                       |
| File exists: data/processed/cleaned_sale_residential_listings.parquet               | PASS     | 12193357                                                       |
| File exists: data/dashboard/market_listings.parquet                                 | PASS     | 3306609                                                        |
| File exists: models/final_model/catboost_fair_price.cbm                             | PASS     | 1430108                                                        |
| File exists: models/metadata/model_metadata.json                                    | PASS     | 50570                                                          |
| File exists: reports/final_eda_model_report.md                                      | PASS     | 5828                                                           |
| File exists: reports/decision_log.csv                                               | PASS     | 9646                                                           |
| Analytical dataset non-empty                                                        | PASS     | 18,053 rows × 43 columns                                       |
| Raw dataset preserved                                                               | PASS     | Raw CSV readable; original checksum recorded in Phase 1        |
| Decision log complete                                                               | PASS     | 19 decisions                                                   |
| Model prediction smoke test                                                         | PASS     | Generated 3 log-price predictions                              |
| Notebook execution                                                                  | PASS     | nbconvert execution completed with bundled artifacts           |
| Streamlit application test                                                          | PASS     | AppTest completed with 8 tabs, 14 metrics, and zero exceptions |

## Result

All required smoke tests passed. Full retraining remains available through `python -m src.pipeline --root .`; the executed notebook defaults to reusing the bundled model for a fast reproducible run.
