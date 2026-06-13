# AdPilot Egyptian Real Estate Market Intelligence

Standalone EDA, fair-price modeling, market positioning, and Streamlit dashboard for the uploaded Egypt Real Estate Data 2026 dataset. It is **not integrated with AdPilot application schemas**.

## Main outputs

- `notebooks/adpilot_egypt_real_estate_market_intelligence.ipynb`
- `dashboard/app.py`
- `src/pipeline.py`
- `data/processed/cleaned_sale_residential_listings.parquet`
- `data/dashboard/market_listings.parquet`
- `models/final_model/catboost_fair_price.cbm`
- `reports/final_eda_model_report.md`
- `reports/decision_log.csv`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run the reproducible pipeline

The raw files are preserved in `data/raw/`.

```bash
python -m src.pipeline --root .
```

Reuse existing model artifacts without retraining:

```bash
python -m src.pipeline --root . --no-retrain
```

## Run the dashboard

```bash
streamlit run dashboard/app.py
```

## Model summary

- Scope: sale listings in EGP for Apartment, Villa, Townhouse, Duplex, Chalet, and Twin House.
- Target: `log(price_egp)`.
- Split: group-aware train/validation/test split by named submarket/compound, otherwise town + district.
- Deployable model: CatBoost.
- Important trade-off: Random Forest obtained the lowest validation MAE; CatBoost was retained because it had lower validation median absolute error and MAPE, while handling high-cardinality categorical fields natively.
- Test metrics and limitations are stored in `models/metadata/model_metadata.json` and shown in the dashboard.

## Cautions

The model predicts asking-price positioning, not transaction value or a certified appraisal. Listings may be stale or advertise payment-plan amounts. The source does not provide a reliable developer field, so developer rankings were intentionally omitted.
