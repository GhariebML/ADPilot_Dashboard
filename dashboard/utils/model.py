from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from catboost import CatBoostRegressor, Pool

ROOT = Path(__file__).resolve().parents[2]

@st.cache_resource(show_spinner=False)
def load_model() -> CatBoostRegressor:
    model = CatBoostRegressor()
    model.load_model(str(ROOT / "models" / "final_model" / "catboost_fair_price.cbm"))
    return model


def build_prediction_row(inputs: dict, metadata: dict, market: pd.DataFrame) -> pd.DataFrame:
    numeric = metadata["numeric_features"]
    categorical = metadata["categorical_features"]
    row = {}
    for feature in numeric:
        if feature in inputs:
            row[feature] = inputs[feature]
        elif feature in market.columns:
            row[feature] = float(pd.to_numeric(market[feature], errors="coerce").median())
        else:
            row[feature] = 0.0
    for feature in categorical:
        row[feature] = str(inputs.get(feature, "Unknown"))
    area = max(float(row.get("area_sqm", 1)), 1)
    beds = row.get("bedrooms_num", np.nan)
    baths = row.get("bathrooms_num", np.nan)
    row["bedrooms_per_100_sqm"] = float(beds) / area * 100 if pd.notna(beds) else np.nan
    row["bathrooms_per_100_sqm"] = float(baths) / area * 100 if pd.notna(baths) else np.nan
    return pd.DataFrame([row])[metadata["features"]]


def predict_price(row: pd.DataFrame, metadata: dict) -> dict:
    model = load_model()
    pred_log = float(model.predict(row)[0])
    pred = float(np.exp(pred_log))
    residual = metadata["residual_calibration"]
    lower = float(np.exp(pred_log + residual["lower_log_residual_q05"]))
    upper = float(np.exp(pred_log + residual["upper_log_residual_q95"]))
    return {"predicted": pred, "lower": lower, "upper": upper, "pred_log": pred_log}


def local_explanation(row: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    model = load_model()
    cat_idx = [metadata["features"].index(c) for c in metadata["categorical_features"]]
    shap = model.get_feature_importance(Pool(row, cat_features=cat_idx), type="ShapValues")[0, :-1]
    result = pd.DataFrame({
        "feature": metadata["features"],
        "value": [str(row.iloc[0][c]) for c in metadata["features"]],
        "impact_log_price": shap,
    })
    result["direction"] = np.where(result["impact_log_price"] >= 0, "Raises estimate", "Lowers estimate")
    result["magnitude"] = result["impact_log_price"].abs()
    return result.sort_values("magnitude", ascending=False)


def estimate_tier(pred_ppsqm: float, town: str, property_type: str, market: pd.DataFrame) -> tuple[str, float]:
    global_pct = float((market["price_per_sqm"] <= pred_ppsqm).mean())
    segment = market[(market["town"] == town) & (market["property_type"] == property_type)]
    if len(segment) < 30:
        segment = market[(market["market_region"] == market.loc[market["town"].eq(town), "market_region"].mode().iloc[0]) & (market["property_type"] == property_type)] if market["town"].eq(town).any() else market[market["property_type"] == property_type]
    segment_pct = float((segment["price_per_sqm"] <= pred_ppsqm).mean()) if len(segment) else global_pct
    score = 0.60 * global_pct + 0.40 * segment_pct
    tier = "Affordable" if score <= .25 else "Mid-market" if score <= .50 else "Upper-mid" if score <= .75 else "Luxury"
    return tier, score


def confidence_label(row: pd.DataFrame, metadata: dict, market: pd.DataFrame) -> tuple[str, list[str]]:
    flags=[]
    for feature, bounds in metadata.get("training_ranges", {}).items():
        value = pd.to_numeric(row.iloc[0].get(feature), errors="coerce")
        if pd.isna(value) or value < bounds["q01"] or value > bounds["q99"]:
            flags.append(f"{feature} is outside the central training range")
    town=str(row.iloc[0]["town"]); ptype=str(row.iloc[0]["property_type"])
    support=int(((market["town"]==town)&(market["property_type"]==ptype)).sum())
    if support < 30: flags.append(f"Only {support} comparable town/type records")
    label = "High" if not flags and support >= 100 else "Medium" if len(flags) <= 1 and support >= 30 else "Low"
    return label, flags


def comparable_listings(row: pd.DataFrame, market: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    r=row.iloc[0]
    pool=market[market["property_type"].eq(str(r["property_type"]))].copy()
    same_town=pool[pool["town"].eq(str(r["town"]))]
    if len(same_town)>=n: pool=same_town
    elif market["town"].eq(str(r["town"])).any():
        region=market.loc[market["town"].eq(str(r["town"])), "market_region"].mode().iloc[0]
        pool=pool[pool["market_region"].eq(region)]
    for c in ["area_sqm","bedrooms_num","bathrooms_num"]:
        med=float(pd.to_numeric(market[c], errors="coerce").median())
        pool[c]=pd.to_numeric(pool[c], errors="coerce").fillna(med)
    area=max(float(r["area_sqm"]),1)
    dist=((pool["area_sqm"]-area).abs()/area)
    dist += .25*(pool["bedrooms_num"]-float(r["bedrooms_num"])).abs()
    dist += .20*(pool["bathrooms_num"]-float(r["bathrooms_num"])).abs()
    dist += .15*(pool["furnished"].astype(str).ne(str(r["furnished"]))).astype(float)
    dist += .15*(pool["payment_method"].astype(str).ne(str(r["payment_method"]))).astype(float)
    pool["similarity_score"]=(1/(1+dist)).clip(0,1)
    cols=["title","town","district","submarket_or_compound","property_type","area_sqm","bedrooms_num","bathrooms_num","price_egp","price_per_sqm","market_tier","similarity_score","detail_url"]
    return pool.nlargest(n,"similarity_score")[cols]
