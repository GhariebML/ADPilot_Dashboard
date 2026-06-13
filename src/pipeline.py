from __future__ import annotations

import json
import logging
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler

RANDOM_SEED = 42
SELECTED_PROPERTY_TYPES = ["Apartment", "Villa", "Townhouse", "Duplex", "Chalet", "Twin House"]
TIER_LABELS = ["Affordable", "Mid-market", "Upper-mid", "Luxury"]

NUMERIC_FEATURES = [
    "area_sqm",
    "bedrooms_num",
    "bathrooms_num",
    "lat",
    "lon",
    "images_count",
    "listing_age_days",
    "amenities_count",
    "bedrooms_per_100_sqm",
    "bathrooms_per_100_sqm",
    "is_premium",
    "is_featured",
    "has_view_360",
]
CATEGORICAL_FEATURES = [
    "market_region",
    "town",
    "district",
    "submarket_or_compound",
    "property_type",
    "completion_status",
    "payment_method",
    "furnished",
    "listing_level",
    "new_city_indicator",
    "compound_status",
]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

NEW_CITY_NAMES = {
    "New Cairo City",
    "Sheikh Zayed City",
    "6 October City",
    "Mostakbal City - Future City",
    "New Capital City",
    "Shorouk City",
    "New Heliopolis",
    "New Obour City",
    "Noor City",
    "Madinaty",
}
REGION_MAP = {
    "South Sainai": "South Sinai",
    "Al Daqahlya": "Dakahlia",
    "Demyat": "Damietta",
}

AREA_PATTERN = re.compile(
    r"(?<!\d)(\d{2,4}(?:\.\d+)?)\s*(?:m²|m2|sqm|sq\.?\s*m|meter(?:s)?|metre(?:s)?|متر(?:\s*مربع)?)\b",
    flags=re.IGNORECASE,
)
AREA_M_PATTERN = re.compile(r"(?<!\d)(\d{2,4}(?:\.\d+)?)\s*m\b", flags=re.IGNORECASE)
PAYMENT_TEXT_PATTERN = re.compile(
    r"quarter|monthly|per\s+month|down\s*payment|downpayment|\bdp\b|deposit|مقدم|قسط|شهري|ربع سنوي",
    flags=re.IGNORECASE,
)


@dataclass
class SplitData:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    group_column: str


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def ensure_dirs(root: Path) -> dict[str, Path]:
    paths = {
        "raw": root / "data" / "raw",
        "processed": root / "data" / "processed",
        "dashboard": root / "data" / "dashboard",
        "audit": root / "data" / "audit",
        "models": root / "models" / "final_model",
        "metadata": root / "models" / "metadata",
        "reports": root / "reports",
        "tables": root / "reports" / "tables",
        "figures": root / "reports" / "figures",
        "config": root / "config",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def _clean_string(series: pd.Series, unknown: str = "Unknown") -> pd.Series:
    return series.astype("string").str.strip().replace({"": pd.NA}).fillna(unknown).astype(str)


def _parse_count(value: Any, studio_zero: bool = False) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    if studio_zero and text == "studio":
        return 0.0
    if text == "none":
        return 0.0
    text = text.replace("+", "")
    try:
        return float(text)
    except ValueError:
        return np.nan


def _extract_area_from_text(title: Any, description: Any) -> tuple[float | None, str]:
    for source, value in (("title", title), ("description", description)):
        if pd.isna(value):
            continue
        text = str(value).replace(",", "")
        candidates = [float(x) for x in AREA_PATTERN.findall(text)]
        if not candidates:
            candidates = [float(x) for x in AREA_M_PATTERN.findall(text)]
        candidates = [x for x in candidates if 21 <= x <= 2000]
        if candidates:
            return candidates[0], source
    return None, "not_recovered"


def load_and_prepare(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    paths = ensure_dirs(root)
    source = paths["processed"] / "phase3_after_d007.csv.gz"
    if not source.exists():
        source = paths["raw"] / "propertyfinder.csv"
    raw = pd.read_csv(source, low_memory=False)
    start_rows = len(raw)

    # D008: retain reference values but never treat them as unique IDs.
    audit_rows: list[pd.DataFrame] = []

    def exclude(mask: pd.Series, reason: str, detail: str = "") -> None:
        nonlocal raw
        affected = raw.loc[mask].copy()
        if affected.empty:
            return
        affected["exclusion_reason"] = reason
        affected["exclusion_detail"] = detail
        audit_rows.append(affected)
        raw = raw.loc[~mask].copy()

    exclude(raw["category"].astype(str).str.lower().ne("buy"), "not_sale_listing")
    exclude(~raw["property_type"].isin(SELECTED_PROPERTY_TYPES), "property_type_out_of_scope")
    exclude(raw["price_currency"].astype(str).str.upper().ne("EGP"), "non_egp_currency")
    exclude(raw["price_egp"].isna() | raw["price_egp"].le(0), "invalid_nonpositive_price")
    exclude(raw["area_value"].isna() | raw["area_value"].le(0), "invalid_nonpositive_area")

    df = raw.copy()
    df["market_region_original"] = df["city"]
    df["market_region"] = _clean_string(df["city"]).replace(REGION_MAP)
    df["town"] = _clean_string(df["town"])
    df["district"] = _clean_string(df["district"])
    df["submarket_or_compound"] = _clean_string(df["subdistrict"])
    for c in ["property_type", "completion_status", "payment_method", "furnished", "listing_level"]:
        df[c] = _clean_string(df[c])

    df["area_original"] = df["area_value"].astype(float)
    df["area_sqm"] = df["area_original"]
    df["area_recovery_source"] = "original"
    suspicious_area = df["area_sqm"].le(20)
    recovered_count = 0
    for idx, row in df.loc[suspicious_area, ["title", "description"]].iterrows():
        recovered, source_name = _extract_area_from_text(row["title"], row["description"])
        if recovered is not None:
            df.at[idx, "area_sqm"] = recovered
            df.at[idx, "area_recovery_source"] = source_name
            recovered_count += 1
        else:
            df.at[idx, "area_recovery_source"] = "not_recovered"

    # Conservative validity rules. Audit all removals.
    def exclude_df(mask: pd.Series, reason: str, detail: str = "") -> None:
        nonlocal df
        affected = df.loc[mask].copy()
        if affected.empty:
            return
        affected["exclusion_reason"] = reason
        affected["exclusion_detail"] = detail
        audit_rows.append(affected)
        df = df.loc[~mask].copy()

    exclude_df(df["area_sqm"].le(20), "unrecovered_implausible_area", "Area <=20 sqm and no reliable larger area in text")
    exclude_df(df["area_sqm"].gt(2000), "implausibly_large_residential_area", "Area >2000 sqm")

    df["price_per_sqm"] = df["price_egp"] / df["area_sqm"]
    combined_text = (df["title"].fillna("") + " " + df["description"].fillna("")).astype(str)
    payment_like = combined_text.str.contains(PAYMENT_TEXT_PATTERN, na=False)
    exclude_df(
        (df["price_egp"].lt(300_000) & payment_like) | (df["price_per_sqm"].lt(5_000)),
        "probable_deposit_or_installment_amount",
        "Very low total or price/sqm, often paired with payment-period wording",
    )
    exclude_df(df["price_egp"].gt(500_000_000), "implausibly_high_total_price", "Total asking price > EGP 500m")
    exclude_df(df["price_per_sqm"].gt(1_000_000), "implausibly_high_price_per_sqm", "Price/sqm > EGP 1m")

    df["bedrooms_num"] = df["bedrooms"].map(lambda x: _parse_count(x, studio_zero=True))
    df["bathrooms_num"] = df["bathrooms"].map(_parse_count)
    df["listed_date_parsed"] = pd.to_datetime(df["listed_date"], errors="coerce", utc=True)
    df["scraped_at_parsed"] = pd.to_datetime(df["scraped_at"], errors="coerce", utc=True)
    df["listing_age_days"] = (
        df["scraped_at_parsed"] - df["listed_date_parsed"]
    ).dt.total_seconds().div(86400).clip(lower=0)
    df["amenities_count"] = (
        df["amenities"].fillna("").astype(str).map(lambda x: 0 if not x.strip() else len([v for v in x.split("|") if v.strip()]))
    )
    df["bedrooms_per_100_sqm"] = df["bedrooms_num"] / df["area_sqm"] * 100
    df["bathrooms_per_100_sqm"] = df["bathrooms_num"] / df["area_sqm"] * 100
    df["new_city_indicator"] = np.where(df["town"].isin(NEW_CITY_NAMES), "New city", "Established/other")
    df["compound_status"] = np.where(
        df["submarket_or_compound"].eq("Unknown")
        | df["submarket_or_compound"].str.contains("Compounds", case=False, na=False),
        "Broad/standalone",
        "Named submarket/compound",
    )
    df["log_price"] = np.log(df["price_egp"])
    df["log_price_per_sqm"] = np.log(df["price_per_sqm"])

    # Model split group: named compounds where available; otherwise town + district.
    named = ~df["compound_status"].eq("Broad/standalone")
    df["split_group"] = np.where(
        named,
        "compound::" + df["submarket_or_compound"],
        "area::" + df["town"] + "::" + df["district"],
    )

    # Hybrid tiers: 60% absolute price/sqm percentile, 40% within-market/property type percentile.
    df["global_ppsqm_percentile"] = df["price_per_sqm"].rank(pct=True, method="average")
    segment_cols = ["town", "property_type"]
    seg_sizes = df.groupby(segment_cols, observed=True)["listing_id"].transform("size")
    within_segment = df.groupby(segment_cols, observed=True)["price_per_sqm"].rank(pct=True, method="average")
    fallback = df.groupby(["market_region", "property_type"], observed=True)["price_per_sqm"].rank(pct=True, method="average")
    df["segment_ppsqm_percentile"] = np.where(seg_sizes.ge(30), within_segment, fallback)
    df["segment_ppsqm_percentile"] = pd.Series(df["segment_ppsqm_percentile"], index=df.index).fillna(df["global_ppsqm_percentile"])
    df["tier_score"] = 0.60 * df["global_ppsqm_percentile"] + 0.40 * df["segment_ppsqm_percentile"]
    df["market_tier"] = pd.cut(
        df["tier_score"], bins=[-np.inf, 0.25, 0.50, 0.75, np.inf], labels=TIER_LABELS, ordered=True
    ).astype(str)
    top_towns = (
        df.groupby("town", observed=True)["price_per_sqm"].median().rank(pct=True).ge(0.75)
    )
    luxury_towns = set(top_towns[top_towns].index)
    df["luxury_location_indicator"] = np.where(df["town"].isin(luxury_towns), "Top-quartile town", "Other town")

    # Ensure model columns are finite/clean. CatBoost requires categorical strings.
    for c in CATEGORICAL_FEATURES:
        df[c] = _clean_string(df[c])
    for c in NUMERIC_FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["is_premium", "is_featured", "has_view_360"]:
        df[c] = df[c].astype(int)

    # Audit export.
    audit = pd.concat(audit_rows, ignore_index=True, sort=False) if audit_rows else pd.DataFrame()
    if not audit.empty:
        keep_audit_cols = [
            c for c in [
                "listing_id", "property_type", "category", "price_egp", "area_value", "area_sqm",
                "city", "town", "district", "subdistrict", "title", "detail_url",
                "exclusion_reason", "exclusion_detail"
            ] if c in audit.columns
        ]
        audit[keep_audit_cols].to_parquet(paths["audit"] / "model_scope_exclusions.parquet", index=False)
        audit["exclusion_reason"].value_counts().rename_axis("reason").reset_index(name="records").to_csv(
            paths["tables"] / "exclusion_summary.csv", index=False
        )

    metadata = {
        "source_rows_after_d007": int(start_rows),
        "final_analytical_rows": int(len(df)),
        "excluded_rows_total": int(start_rows - len(df)),
        "recovered_area_rows": int(recovered_count),
        "selected_property_types": SELECTED_PROPERTY_TYPES,
        "price_target": "log_price",
        "price_tier_method": "0.60 global price/sqm percentile + 0.40 within town/property-type percentile",
        "developer_analysis_status": "Unavailable: no explicit developer field in source data",
    }
    return df.reset_index(drop=True), audit, metadata


def choose_group_split(df: pd.DataFrame) -> SplitData:
    indices = np.arange(len(df))
    groups = df["split_group"].astype(str).to_numpy()
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    overall_mean = df["log_price"].mean()
    overall_mix = df["property_type"].value_counts(normalize=True)
    for seed in range(40, 90):
        gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
        trainval_idx, test_idx = next(gss.split(indices, groups=groups))
        test = df.iloc[test_idx]
        size_penalty = abs(len(test_idx) / len(df) - 0.15)
        price_penalty = abs(test["log_price"].mean() - overall_mean) / max(df["log_price"].std(), 1e-9)
        mix = test["property_type"].value_counts(normalize=True).reindex(overall_mix.index, fill_value=0)
        mix_penalty = float((mix - overall_mix).abs().mean())
        score = size_penalty * 5 + price_penalty + mix_penalty
        if best is None or score < best[0]:
            best = (score, trainval_idx, test_idx)
    assert best is not None
    trainval_idx, test_idx = best[1], best[2]
    tv_groups = groups[trainval_idx]
    best_val: tuple[float, np.ndarray, np.ndarray] | None = None
    tv_df = df.iloc[trainval_idx]
    for seed in range(90, 140):
        gss = GroupShuffleSplit(n_splits=1, test_size=0.1765, random_state=seed)  # ~15% total
        train_rel, val_rel = next(gss.split(trainval_idx, groups=tv_groups))
        val = tv_df.iloc[val_rel]
        size_penalty = abs(len(val_rel) / len(df) - 0.15)
        price_penalty = abs(val["log_price"].mean() - overall_mean) / max(df["log_price"].std(), 1e-9)
        mix = val["property_type"].value_counts(normalize=True).reindex(overall_mix.index, fill_value=0)
        mix_penalty = float((mix - overall_mix).abs().mean())
        score = size_penalty * 5 + price_penalty + mix_penalty
        if best_val is None or score < best_val[0]:
            best_val = (score, train_rel, val_rel)
    assert best_val is not None
    train_idx = trainval_idx[best_val[1]]
    val_idx = trainval_idx[best_val[2]]
    return SplitData(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx, group_column="split_group")


def regression_metrics(y_true_egp: np.ndarray, y_pred_egp: np.ndarray) -> dict[str, float]:
    ape = np.abs(y_true_egp - y_pred_egp) / np.maximum(np.abs(y_true_egp), 1.0)
    smape = 2 * np.abs(y_true_egp - y_pred_egp) / np.maximum(np.abs(y_true_egp) + np.abs(y_pred_egp), 1.0)
    return {
        "MAE_EGP": float(mean_absolute_error(y_true_egp, y_pred_egp)),
        "RMSE_EGP": float(math.sqrt(mean_squared_error(y_true_egp, y_pred_egp))),
        "R2": float(r2_score(y_true_egp, y_pred_egp)),
        "MedianAE_EGP": float(median_absolute_error(y_true_egp, y_pred_egp)),
        "MAPE": float(np.mean(ape)),
        "sMAPE": float(np.mean(smape)),
    }


def _fit_baseline(train: pd.DataFrame, target_log: pd.Series) -> dict[str, Any]:
    temp = train[["town", "property_type"]].copy()
    temp["target"] = target_log.to_numpy()
    by_town_type = temp.groupby(["town", "property_type"], observed=True)["target"].median().to_dict()
    by_type = temp.groupby("property_type", observed=True)["target"].median().to_dict()
    return {"by_town_type": by_town_type, "by_type": by_type, "global": float(target_log.median())}


def _predict_baseline(model: dict[str, Any], frame: pd.DataFrame) -> np.ndarray:
    values = []
    for row in frame[["town", "property_type"]].itertuples(index=False):
        values.append(model["by_town_type"].get((row.town, row.property_type), model["by_type"].get(row.property_type, model["global"])))
    return np.asarray(values)


def _sklearn_preprocessor(one_hot: bool) -> ColumnTransformer:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", RobustScaler()),
    ])
    if one_hot:
        categorical = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=20)),
        ])
    else:
        categorical = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])


def train_and_evaluate(df: pd.DataFrame, root: Path) -> tuple[pd.DataFrame, CatBoostRegressor, dict[str, Any], pd.DataFrame]:
    paths = ensure_dirs(root)
    split = choose_group_split(df)
    split_name = np.full(len(df), "train", dtype=object)
    split_name[split.val_idx] = "validation"
    split_name[split.test_idx] = "test"
    df["data_split"] = split_name

    train = df.iloc[split.train_idx].copy()
    val = df.iloc[split.val_idx].copy()
    test = df.iloc[split.test_idx].copy()
    X_train, y_train = train[MODEL_FEATURES], train["log_price"]
    X_val, y_val = val[MODEL_FEATURES], val["log_price"]
    X_test, y_test = test[MODEL_FEATURES], test["log_price"]

    comparison_rows: list[dict[str, Any]] = []
    fitted: dict[str, Any] = {}

    def assess(name: str, fit_fn, predict_fn, strengths: str, weaknesses: str, complexity: str) -> None:
        logging.info("Fitting %s", name)
        start = time.perf_counter()
        model = fit_fn()
        fit_time = time.perf_counter() - start
        start_pred = time.perf_counter()
        pred_log = predict_fn(model)
        pred_time = time.perf_counter() - start_pred
        metrics = regression_metrics(np.exp(y_val.to_numpy()), np.exp(pred_log))
        comparison_rows.append({
            "Model": name,
            "Validation MAE": metrics["MAE_EGP"],
            "Validation RMSE": metrics["RMSE_EGP"],
            "Validation R2": metrics["R2"],
            "Median absolute error": metrics["MedianAE_EGP"],
            "Validation MAPE": metrics["MAPE"],
            "Training seconds": fit_time,
            "Prediction seconds": pred_time,
            "Explainability": "High" if name in {"Median baseline", "Ridge"} else "Medium",
            "Deployment complexity": complexity,
            "Main strengths": strengths,
            "Main weaknesses": weaknesses,
        })
        fitted[name] = model
        logging.info("Finished %s in %.1fs; validation MAE EGP %.0f", name, fit_time, metrics["MAE_EGP"])

    assess(
        "Median baseline",
        lambda: _fit_baseline(train, y_train),
        lambda m: _predict_baseline(m, val),
        "Transparent and robust location/property-type benchmark",
        "Cannot model nonlinear interactions or unit characteristics",
        "Low",
    )

    ridge = Pipeline([("prep", _sklearn_preprocessor(one_hot=True)), ("model", Ridge(alpha=10.0))])
    assess(
        "Ridge",
        lambda: ridge.fit(X_train, y_train),
        lambda m: m.predict(X_val),
        "Stable, fast, interpretable linear benchmark",
        "Limited nonlinear interaction capture",
        "Medium",
    )

    rf = Pipeline([
        ("prep", _sklearn_preprocessor(one_hot=False)),
        ("model", RandomForestRegressor(
            n_estimators=100, max_depth=18, min_samples_leaf=3, max_features=0.65,
            n_jobs=4, random_state=RANDOM_SEED
        )),
    ])
    assess(
        "Random Forest",
        lambda: rf.fit(X_train, y_train),
        lambda m: m.predict(X_val),
        "Captures nonlinearities and interactions",
        "Large model and weaker extrapolation",
        "High",
    )

    # HistGradientBoosting was omitted after a container-specific OpenMP stall;
    # CatBoost provides the intended boosted-tree benchmark with native categorical handling.

    cat_features_idx = [MODEL_FEATURES.index(c) for c in CATEGORICAL_FEATURES]
    cat_model = CatBoostRegressor(
        iterations=380,
        depth=7,
        learning_rate=0.055,
        loss_function="RMSE",
        eval_metric="MAE",
        random_seed=RANDOM_SEED,
        l2_leaf_reg=5,
        random_strength=0.6,
        od_type="Iter",
        od_wait=60,
        verbose=False,
        allow_writing_files=False,
        thread_count=4,
    )

    def fit_cat() -> CatBoostRegressor:
        return cat_model.fit(
            X_train,
            y_train,
            cat_features=cat_features_idx,
            eval_set=(X_val, y_val),
            use_best_model=True,
            verbose=False,
        )

    assess(
        "CatBoost",
        fit_cat,
        lambda m: m.predict(X_val),
        "Strong native handling of high-cardinality categories and nonlinear effects",
        "More complex deployment and explanation than linear models",
        "Medium",
    )

    comparison = pd.DataFrame(comparison_rows).sort_values("Validation MAE").reset_index(drop=True)
    comparison.to_csv(paths["tables"] / "model_comparison.csv", index=False)

    logging.info("Model comparison complete")
    # Business-weighted validation choice. Random Forest minimized validation MAE/RMSE,
    # while CatBoost minimized median absolute error and MAPE and natively handles
    # unseen/high-cardinality categories. The deployable selection is documented
    # explicitly rather than being inferred from the test set.
    selected_name = "CatBoost"

    logging.info("Refitting final CatBoost on train+validation")
    # Export the CatBoost model as the final deployable model; retrain on train+validation with best iteration.
    best_cat: CatBoostRegressor = fitted["CatBoost"]
    best_iter = max(int(best_cat.get_best_iteration()) + 1, 100)
    trainval = pd.concat([train, val], ignore_index=True)
    final_model = CatBoostRegressor(
        iterations=best_iter,
        depth=7,
        learning_rate=0.055,
        loss_function="RMSE",
        random_seed=RANDOM_SEED,
        l2_leaf_reg=5,
        random_strength=0.6,
        verbose=False,
        allow_writing_files=False,
        thread_count=4,
    )
    final_model.fit(
        trainval[MODEL_FEATURES], trainval["log_price"], cat_features=cat_features_idx, verbose=False
    )
    final_model.save_model(str(paths["models"] / "catboost_fair_price.cbm"))
    logging.info("Final CatBoost saved")

    # Validation residuals from the model before train+validation refit calibrate the interval.
    val_pred_log = best_cat.predict(X_val)
    val_resid_log = y_val.to_numpy() - val_pred_log
    residual_calibration = {
        "lower_log_residual_q05": float(np.quantile(val_resid_log, 0.05)),
        "upper_log_residual_q95": float(np.quantile(val_resid_log, 0.95)),
        "median_abs_percentage_error": float(np.median(np.abs(np.exp(y_val) - np.exp(val_pred_log)) / np.exp(y_val))),
        "validation_rows": int(len(val)),
    }

    logging.info("Evaluating untouched test set")
    # Untouched test evaluation once.
    test_pred_log = final_model.predict(X_test)
    test_pred = np.exp(test_pred_log)
    test_actual = np.exp(y_test.to_numpy())
    test_metrics = regression_metrics(test_actual, test_pred)
    test_results = test[[
        "listing_id", "property_type", "market_region", "town", "district", "submarket_or_compound",
        "area_sqm", "bedrooms_num", "bathrooms_num", "price_egp", "price_per_sqm", "market_tier"
    ]].copy()
    test_results["predicted_fair_price"] = test_pred
    test_results["lower_fair_price"] = np.exp(test_pred_log + residual_calibration["lower_log_residual_q05"])
    test_results["upper_fair_price"] = np.exp(test_pred_log + residual_calibration["upper_log_residual_q95"])
    test_results["absolute_error"] = np.abs(test_results["price_egp"] - test_results["predicted_fair_price"])
    test_results["absolute_percentage_error"] = test_results["absolute_error"] / test_results["price_egp"]
    test_results["pricing_status"] = np.select(
        [
            test_results["price_egp"] < test_results["lower_fair_price"],
            test_results["price_egp"] > test_results["upper_fair_price"],
        ],
        ["Underpriced", "Overpriced"],
        default="Fairly priced",
    )
    test_results.to_parquet(paths["tables"] / "test_predictions.parquet", index=False)

    logging.info("Running limited grouped cross-validation")
    # CV limited to baseline and CatBoost on train+validation, grouped by project/area.
    cv_frame = trainval.reset_index(drop=True)
    n_groups = cv_frame["split_group"].nunique()
    n_splits = min(2, n_groups)
    cv_rows: list[dict[str, Any]] = []
    if n_splits >= 2:
        gkf = GroupKFold(n_splits=n_splits)
        for fold, (tr, va) in enumerate(gkf.split(cv_frame, groups=cv_frame["split_group"]), start=1):
            trf, vaf = cv_frame.iloc[tr], cv_frame.iloc[va]
            bm = _fit_baseline(trf, trf["log_price"])
            bp = np.exp(_predict_baseline(bm, vaf))
            cv_rows.append({"model": "Median baseline", "fold": fold, **regression_metrics(vaf["price_egp"].to_numpy(), bp)})
            cm = CatBoostRegressor(
                iterations=min(best_iter, 160), depth=7, learning_rate=0.06, loss_function="RMSE",
                random_seed=RANDOM_SEED + fold, l2_leaf_reg=5, verbose=False,
                allow_writing_files=False, thread_count=4,
            )
            cm.fit(trf[MODEL_FEATURES], trf["log_price"], cat_features=cat_features_idx, verbose=False)
            cp = np.exp(cm.predict(vaf[MODEL_FEATURES]))
            cv_rows.append({"model": "CatBoost", "fold": fold, **regression_metrics(vaf["price_egp"].to_numpy(), cp)})
    cv_results = pd.DataFrame(cv_rows)
    cv_results.to_csv(paths["tables"] / "cross_validation_results.csv", index=False)
    logging.info("Cross-validation complete")

    logging.info("Calculating feature importance and SHAP")
    # Feature importance and SHAP.
    importances = final_model.get_feature_importance()
    feature_importance = pd.DataFrame({"feature": MODEL_FEATURES, "importance": importances}).sort_values("importance", ascending=False)
    feature_importance.to_csv(paths["tables"] / "feature_importance.csv", index=False)
    shap_sample = test.sample(min(100, len(test)), random_state=RANDOM_SEED)
    shap_pool = Pool(shap_sample[MODEL_FEATURES], cat_features=cat_features_idx)
    shap_values = final_model.get_feature_importance(shap_pool, type="ShapValues")[:, :-1]
    shap_summary = pd.DataFrame({"feature": MODEL_FEATURES, "mean_abs_shap_log_price": np.abs(shap_values).mean(axis=0)}).sort_values(
        "mean_abs_shap_log_price", ascending=False
    )
    shap_summary.to_csv(paths["tables"] / "shap_global_importance.csv", index=False)
    local_row = shap_sample.iloc[[0]]
    local_pool = Pool(local_row[MODEL_FEATURES], cat_features=cat_features_idx)
    local_shap = final_model.get_feature_importance(local_pool, type="ShapValues")[0, :-1]
    local_explanation = pd.DataFrame({
        "feature": MODEL_FEATURES,
        "value": [str(local_row.iloc[0][c]) for c in MODEL_FEATURES],
        "shap_log_price": local_shap,
    }).assign(abs_shap=lambda x: x["shap_log_price"].abs()).sort_values("abs_shap", ascending=False)
    local_explanation.to_csv(paths["tables"] / "example_local_explanation.csv", index=False)
    logging.info("Explainability outputs complete")

    # Error slices.
    for group_col, filename in [
        ("property_type", "error_by_property_type.csv"),
        ("market_region", "error_by_market_region.csv"),
        ("market_tier", "error_by_market_tier.csv"),
        ("town", "error_by_town.csv"),
    ]:
        grouped = test_results.groupby(group_col, observed=True).agg(
            sample_size=("listing_id", "size"),
            mae_egp=("absolute_error", "mean"),
            median_ae_egp=("absolute_error", "median"),
            mape=("absolute_percentage_error", "mean"),
        ).reset_index().sort_values("sample_size", ascending=False)
        grouped.to_csv(paths["tables"] / filename, index=False)

    metadata = {
        "selected_model": "CatBoost",
        "selection_basis": "Validation-only business trade-off: Random Forest had the lowest MAE/RMSE, while CatBoost had lower median absolute error and MAPE and safer native handling of high-cardinality categorical inputs. CatBoost was selected without using test performance.",
        "best_iteration": int(best_iter),
        "features": MODEL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target": "log_price",
        "random_seed": RANDOM_SEED,
        "split_strategy": "Group-aware 70/15/15 split using named compound or town+district fallback",
        "train_rows": int(len(train)),
        "validation_rows": int(len(val)),
        "test_rows": int(len(test)),
        "test_metrics": test_metrics,
        "validation_comparison": comparison.to_dict(orient="records"),
        "residual_calibration": residual_calibration,
        "training_ranges": {
            c: {"q01": float(trainval[c].quantile(0.01)), "q99": float(trainval[c].quantile(0.99))}
            for c in ["area_sqm", "bedrooms_num", "bathrooms_num"]
        },
        "category_values": {c: sorted(trainval[c].astype(str).unique().tolist()) for c in CATEGORICAL_FEATURES},
        "support_counts": trainval.groupby(["town", "property_type"], observed=True).size().rename("count").reset_index().to_dict(orient="records"),
    }
    (paths["metadata"] / "model_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    (paths["metadata"] / "residual_calibration.json").write_text(json.dumps(residual_calibration, indent=2), encoding="utf-8")
    joblib.dump({"numeric_features": NUMERIC_FEATURES, "categorical_features": CATEGORICAL_FEATURES}, paths["metadata"] / "feature_schema.joblib")
    return comparison, final_model, metadata, test_results


def score_all_listings(df: pd.DataFrame, model: CatBoostRegressor, model_meta: dict[str, Any], root: Path) -> pd.DataFrame:
    """Score every analytical listing and add fair-value, interval, opportunity, and confidence fields."""
    residual = model_meta["residual_calibration"]
    pred_log = model.predict(df[MODEL_FEATURES])
    df["predicted_fair_price"] = np.exp(pred_log)
    df["lower_fair_price"] = np.exp(pred_log + residual["lower_log_residual_q05"])
    df["upper_fair_price"] = np.exp(pred_log + residual["upper_log_residual_q95"])
    df["predicted_fair_price_per_sqm"] = df["predicted_fair_price"] / df["area_sqm"]
    df["actual_vs_fair_pct"] = (df["price_egp"] - df["predicted_fair_price"]) / df["predicted_fair_price"]
    df["pricing_status"] = np.select(
        [df["price_egp"] < df["lower_fair_price"], df["price_egp"] > df["upper_fair_price"]],
        ["Underpriced", "Overpriced"],
        default="Fairly priced",
    )
    support_lookup = {(str(x["town"]), str(x["property_type"])): int(x["count"]) for x in model_meta.get("support_counts", [])}
    df["model_support_count"] = [support_lookup.get((str(t), str(p)), 0) for t, p in zip(df["town"], df["property_type"])]
    ranges = model_meta.get("training_ranges", {})
    ood = np.zeros(len(df), dtype=int)
    for c in ["area_sqm", "bedrooms_num", "bathrooms_num"]:
        if c in ranges:
            lo, hi = ranges[c]["q01"], ranges[c]["q99"]
            vals = pd.to_numeric(df[c], errors="coerce")
            ood += ((vals < lo) | (vals > hi) | vals.isna()).astype(int).to_numpy()
    df["out_of_distribution_flags"] = ood
    df["model_confidence"] = np.select(
        [(df["model_support_count"] >= 100) & (df["out_of_distribution_flags"] == 0),
         (df["model_support_count"] >= 30) & (df["out_of_distribution_flags"] <= 1)],
        ["High", "Medium"],
        default="Low",
    )
    opportunities = df.loc[df["pricing_status"].ne("Fairly priced"), [
        "listing_id", "property_type", "market_region", "town", "district", "submarket_or_compound",
        "area_sqm", "bedrooms_num", "bathrooms_num", "price_egp", "predicted_fair_price",
        "lower_fair_price", "upper_fair_price", "actual_vs_fair_pct", "pricing_status",
        "model_confidence", "detail_url", "title"
    ]].copy()
    opportunities["opportunity_magnitude"] = opportunities["actual_vs_fair_pct"].abs()
    opportunities.sort_values(["model_confidence", "opportunity_magnitude"], ascending=[True, False]).to_parquet(
        root / "data" / "dashboard" / "pricing_opportunities.parquet", index=False
    )
    return df


def create_eda_outputs(df: pd.DataFrame, root: Path, metadata: dict[str, Any], test_results: pd.DataFrame | None = None) -> None:
    paths = ensure_dirs(root)
    summary = df[["price_egp", "area_sqm", "price_per_sqm", "bedrooms_num", "bathrooms_num"]].describe(
        percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    ).T
    summary.to_csv(paths["tables"] / "summary_statistics.csv")

    property_summary = df.groupby("property_type", observed=True).agg(
        listings=("listing_id", "size"), median_price=("price_egp", "median"), mean_price=("price_egp", "mean"),
        median_price_per_sqm=("price_per_sqm", "median"), median_area=("area_sqm", "median")
    ).reset_index().sort_values("median_price_per_sqm", ascending=False)
    property_summary.to_csv(paths["tables"] / "property_type_summary.csv", index=False)

    location_summary = df.groupby(["market_region", "town"], observed=True).agg(
        listings=("listing_id", "size"), median_price=("price_egp", "median"), mean_price=("price_egp", "mean"),
        median_price_per_sqm=("price_per_sqm", "median"), median_area=("area_sqm", "median"),
        property_types=("property_type", "nunique")
    ).reset_index().sort_values(["listings", "median_price_per_sqm"], ascending=[False, False])
    location_summary.to_csv(paths["tables"] / "location_summary.csv", index=False)

    submarket_summary = df.groupby(["town", "submarket_or_compound"], observed=True).agg(
        listings=("listing_id", "size"), median_price=("price_egp", "median"),
        median_price_per_sqm=("price_per_sqm", "median"), property_types=("property_type", "nunique")
    ).reset_index().sort_values("listings", ascending=False)
    submarket_summary.to_csv(paths["tables"] / "submarket_summary.csv", index=False)

    tier_summary = df.groupby("market_tier", observed=True).agg(
        listings=("listing_id", "size"), median_price=("price_egp", "median"),
        median_price_per_sqm=("price_per_sqm", "median"), median_area=("area_sqm", "median")
    ).reindex(TIER_LABELS).reset_index()
    tier_summary["share"] = tier_summary["listings"] / tier_summary["listings"].sum()
    tier_summary.to_csv(paths["tables"] / "market_tier_summary.csv", index=False)

    missing = df[MODEL_FEATURES].isna().mean().sort_values(ascending=False).rename("missing_share").reset_index(name="missing_share").rename(columns={"index": "feature"})
    missing.to_csv(paths["tables"] / "model_feature_missingness.csv", index=False)

    # Dashboard data excludes contact details and long text.
    dashboard_cols = [
        "listing_id", "detail_url", "property_type", "market_region", "town", "district", "submarket_or_compound",
        "lat", "lon", "bedrooms_num", "bathrooms_num", "area_sqm", "area_original", "area_recovery_source",
        "price_egp", "price_per_sqm", "furnished", "payment_method", "completion_status", "listing_level",
        "listed_date", "listing_age_days", "amenities_count", "images_count", "is_premium", "is_featured",
        "has_view_360", "new_city_indicator", "compound_status", "luxury_location_indicator", "market_tier",
        "tier_score", "data_split", "split_group", "title",
        "predicted_fair_price", "lower_fair_price", "upper_fair_price",
        "predicted_fair_price_per_sqm", "actual_vs_fair_pct", "pricing_status",
        "model_support_count", "out_of_distribution_flags", "model_confidence",
    ]
    dashboard_df = df[[c for c in dashboard_cols if c in df.columns]].copy()
    dashboard_df.to_parquet(paths["dashboard"] / "market_listings.parquet", index=False)
    df.to_parquet(paths["processed"] / "cleaned_sale_residential_listings.parquet", index=False)

    # Charts.
    plt.figure(figsize=(9, 5))
    plt.hist(np.log10(df["price_egp"]), bins=45)
    plt.xlabel("Log10 asking price (EGP)")
    plt.ylabel("Listings")
    plt.title("Sale price distribution")
    plt.tight_layout(); plt.savefig(paths["figures"] / "price_distribution_log.png", dpi=160); plt.close()

    plt.figure(figsize=(10, 5))
    data = [df.loc[df["property_type"].eq(pt), "price_per_sqm"] / 1000 for pt in SELECTED_PROPERTY_TYPES]
    plt.boxplot(data, tick_labels=SELECTED_PROPERTY_TYPES, showfliers=False)
    plt.ylabel("Price per sqm (thousand EGP)")
    plt.title("Price per sqm by property type")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout(); plt.savefig(paths["figures"] / "ppsqm_by_property_type.png", dpi=160); plt.close()

    top_towns = location_summary[location_summary["listings"].ge(50)].nlargest(15, "median_price_per_sqm").sort_values("median_price_per_sqm")
    plt.figure(figsize=(9, 6))
    plt.barh(top_towns["town"], top_towns["median_price_per_sqm"] / 1000)
    plt.xlabel("Median price per sqm (thousand EGP)")
    plt.title("Highest-priced towns with at least 50 listings")
    plt.tight_layout(); plt.savefig(paths["figures"] / "median_ppsqm_by_town.png", dpi=160); plt.close()

    region_counts = df["market_region"].value_counts().sort_values()
    plt.figure(figsize=(9, 5))
    plt.barh(region_counts.index, region_counts.values)
    plt.xlabel("Listings")
    plt.title("Listing volume by market region")
    plt.tight_layout(); plt.savefig(paths["figures"] / "listing_count_by_region.png", dpi=160); plt.close()

    sample = df.sample(min(4000, len(df)), random_state=RANDOM_SEED)
    plt.figure(figsize=(8, 5))
    plt.scatter(sample["area_sqm"], sample["price_egp"] / 1_000_000, alpha=0.25, s=10)
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("Area sqm (log scale)")
    plt.ylabel("Price EGP millions (log scale)")
    plt.title("Area versus asking price")
    plt.tight_layout(); plt.savefig(paths["figures"] / "area_vs_price.png", dpi=160); plt.close()

    tiers = df["market_tier"].value_counts().reindex(TIER_LABELS)
    plt.figure(figsize=(8, 5))
    plt.bar(tiers.index, tiers.values)
    plt.ylabel("Listings")
    plt.title("Hybrid market-tier distribution")
    plt.xticks(rotation=20)
    plt.tight_layout(); plt.savefig(paths["figures"] / "market_tier_distribution.png", dpi=160); plt.close()

    if test_results is not None and not test_results.empty:
        plt.figure(figsize=(6, 6))
        plt.scatter(test_results["price_egp"] / 1e6, test_results["predicted_fair_price"] / 1e6, alpha=0.25, s=12)
        lim = max(test_results["price_egp"].quantile(0.99), test_results["predicted_fair_price"].quantile(0.99)) / 1e6
        plt.plot([0, lim], [0, lim], linestyle="--")
        plt.xlim(0, lim); plt.ylim(0, lim)
        plt.xlabel("Actual price (EGP millions)"); plt.ylabel("Predicted fair price (EGP millions)")
        plt.title("Actual versus predicted price — test set")
        plt.tight_layout(); plt.savefig(paths["figures"] / "model_actual_vs_predicted.png", dpi=160); plt.close()

        residual = (test_results["price_egp"] - test_results["predicted_fair_price"]) / 1e6
        plt.figure(figsize=(8, 5))
        plt.hist(residual.clip(residual.quantile(0.01), residual.quantile(0.99)), bins=45)
        plt.xlabel("Actual minus predicted (EGP millions)"); plt.ylabel("Listings")
        plt.title("Test residual distribution (1st–99th percentile)")
        plt.tight_layout(); plt.savefig(paths["figures"] / "residual_distribution.png", dpi=160); plt.close()

    shap_path = paths["tables"] / "shap_global_importance.csv"
    if shap_path.exists():
        shap_imp = pd.read_csv(shap_path).head(15).sort_values("mean_abs_shap_log_price")
        plt.figure(figsize=(8, 6))
        plt.barh(shap_imp["feature"], shap_imp["mean_abs_shap_log_price"])
        plt.xlabel("Mean absolute SHAP value (log-price scale)")
        plt.title("Global model importance")
        plt.tight_layout(); plt.savefig(paths["figures"] / "shap_feature_importance.png", dpi=160); plt.close()

    (paths["reports"] / "data_preparation_summary.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def create_business_report(root: Path, prep_meta: dict[str, Any], model_meta: dict[str, Any], df: pd.DataFrame) -> None:
    paths = ensure_dirs(root)
    location = pd.read_csv(paths["tables"] / "location_summary.csv")
    prop = pd.read_csv(paths["tables"] / "property_type_summary.csv")
    tiers = pd.read_csv(paths["tables"] / "market_tier_summary.csv")
    comparison = pd.read_csv(paths["tables"] / "model_comparison.csv")
    best_town = location[location["listings"].ge(50)].sort_values("median_price_per_sqm", ascending=False).iloc[0]
    best_prop = prop.sort_values("median_price_per_sqm", ascending=False).iloc[0]
    test = model_meta["test_metrics"]
    report = f"""# Egyptian Real Estate Market Intelligence — Final EDA and Model Report

## Executive summary

The final analytical dataset contains **{len(df):,}** unique, in-scope sale listings across the six approved residential property types. The median asking price is **EGP {df['price_egp'].median():,.0f}**, while the median price per square metre is **EGP {df['price_per_sqm'].median():,.0f}**.

Among towns with at least 50 usable listings, **{best_town['town']}** has the highest observed median price per square metre at approximately **EGP {best_town['median_price_per_sqm']:,.0f}**. By property type, **{best_prop['property_type']}** has the highest median price per square metre.

The selected CatBoost fair-price model achieved an untouched test-set **MAE of EGP {test['MAE_EGP']:,.0f}**, **RMSE of EGP {test['RMSE_EGP']:,.0f}**, and **R² of {test['R2']:.3f}**. Predictions estimate the typical market asking price, not a guaranteed transaction value.

## Data preparation decisions

- Kept repeated `reference` values but excluded the field from uniqueness logic and modeling.
- Restricted the analytical scope to sale listings, EGP, and Apartment, Villa, Townhouse, Duplex, Chalet, and Twin House.
- Recovered suspicious 1–20 sqm areas from explicit units in title/description where reliable: **{prep_meta['recovered_area_rows']:,} records**.
- Excluded only unrecoverable implausible areas, probable deposit/installment amounts, and extreme values outside conservative plausibility bounds. Every exclusion is preserved in `data/audit/model_scope_exclusions.parquet`.
- Missing categorical values are represented as `Unknown`; numeric missingness is handled inside model pipelines or natively by CatBoost.
- No developer ranking was produced because the source has no explicit developer field.

## Market tiers

Tiers use a hybrid score: **60% absolute price-per-sqm percentile and 40% percentile within the same town/property type**. This prevents all properties in expensive locations from automatically becoming luxury while preserving absolute market positioning.

{tiers.to_markdown(index=False)}

## Model design

- Primary target: `log(price_egp)` for stable errors across a highly skewed market.
- Leakage controls: no price-derived feature, title, description, ID, reference, URL, contact, agent, or broker field is used as a predictor.
- Split: group-aware train/validation/test separation by named compound, falling back to town and district.
- Model selection used validation results only. The untouched test set was evaluated once after final refitting.

{comparison.to_markdown(index=False)}

## Explainability and business use

SHAP values identify predictive associations, not causal price drivers. The dashboard presents global importance and local explanations, fair-price intervals, comparable listings, and out-of-distribution warnings.

## Limitations

- Listing prices are asking prices, not completed transaction prices.
- Some listings advertise deposits or installments despite a sale-price field; conservative rules reduce but cannot eliminate this contamination.
- Developer and formal project identifiers are absent.
- Geographic labels mix formal and marketing geographies.
- Market conditions may shift after the March 2026 scrape.
- Confidence intervals are empirical validation residual ranges, not legal or appraisal guarantees.
"""
    (paths["reports"] / "final_eda_model_report.md").write_text(report, encoding="utf-8")


def append_autonomous_decisions(root: Path, prep_meta: dict[str, Any], model_meta: dict[str, Any]) -> None:
    path = root / "reports" / "decision_log.csv"
    if path.exists():
        log = pd.read_csv(path)
    else:
        log = pd.DataFrame()
    columns = [
        "Decision ID", "Phase", "Date or sequence number", "Issue", "Options presented", "Recommended option",
        "My selected option", "Reason for the selection", "Number of affected records", "Implementation performed",
        "Resulting dataset shape", "Reversibility", "Notes"
    ]
    if log.empty:
        log = pd.DataFrame(columns=columns)
    for c in columns:
        if c not in log.columns:
            log[c] = ""
    decisions = [
        ("D008", "Phase 3", "Repeated references", "Keep but do not use as unique identifier", "A", "Avoids destructive changes to broker-entered references", "0", "Excluded `reference` from identifiers and model features"),
        ("D009", "Phase 3–8", "Analytical scope", "Sale, EGP, six residential types", "Recommended", "Matches the stated business question", str(prep_meta["excluded_rows_total"]), "Created scoped analytical dataset and audit exclusions"),
        ("D010", "Phase 6", "Suspicious area values", "Recover explicit text area; otherwise exclude <=20 or >2000 sqm", "Recommended", "Balances recovery and conservative validity", str(prep_meta["recovered_area_rows"]), "Added area_sqm, area_original, and recovery source"),
        ("D011", "Phase 6", "Price anomalies", "Exclude only extreme ppsqm/total and probable payment amounts", "Recommended", "Preserves legitimate luxury listings", "See exclusion summary", "Audit-safe anomaly exclusions"),
        ("D012", "Phase 8", "Feature set", "Deployment-time structured fields only", "Recommended", "Prevents text and target leakage", "All model rows", "Created numeric/categorical engineered features"),
        ("D013", "Phase 11", "Market tiering", "Hybrid absolute and segment-adjusted ppsqm percentiles", "Recommended", "Avoids location-only luxury labeling", "All analytical rows", "Created tier_score and market_tier"),
        ("D014", "Phase 12", "Target", "log(total price)", "Recommended", "Stable across skewed price distribution and directly deployable", "All analytical rows", "Trained models on log_price"),
        ("D015", "Phase 12", "Data split", "Grouped compound or town+district split", "Recommended", "Reduces project leakage", "All analytical rows", "Created 70/15/15 grouped split"),
        ("D016", "Phase 12–13", "Final model", "CatBoost", "Recommended", "Native categorical support and strong validation performance", str(model_meta["train_rows"] + model_meta["validation_rows"]), "Exported CatBoost model"),
        ("D017", "Phase 15", "Fair price classification", "90% empirical validation-residual interval", "Recommended", "Avoids arbitrary percentage thresholds", str(model_meta["test_rows"]), "Under/fair/over classifications use interval boundaries"),
        ("D018", "Phase 15", "Comparable listings", "Rule-filtered nearest standardized distance", "Recommended", "Transparent and deployable", "Dashboard inputs", "Implemented in dashboard"),
        ("D019", "Phase 16", "Dashboard framework", "Streamlit", "Default", "Matches requested default and supports interactive storytelling", "N/A", "Built and smoke-tested Streamlit app"),
    ]
    existing = set(log["Decision ID"].astype(str))
    new_rows = []
    for did, phase, issue, option, selected, reason, affected, implementation in decisions:
        if did in existing:
            continue
        new_rows.append({
            "Decision ID": did,
            "Phase": phase,
            "Date or sequence number": "2026-06-13",
            "Issue": issue,
            "Options presented": option,
            "Recommended option": option,
            "My selected option": selected,
            "Reason for the selection": reason,
            "Number of affected records": affected,
            "Implementation performed": implementation,
            "Resulting dataset shape": f"{prep_meta['final_analytical_rows']:,} analytical rows",
            "Reversibility": "Fully reversible from raw and audit datasets",
            "Notes": "Autonomously selected after user delegated remaining cautious decisions",
        })
    if new_rows:
        log = pd.concat([log, pd.DataFrame(new_rows)], ignore_index=True)
    log[columns].to_csv(path, index=False)


def run_pipeline(root: str | Path, retrain: bool = True) -> dict[str, Any]:
    setup_logging()
    root = Path(root).resolve()
    logging.info("Preparing analytical data")
    df, audit, prep_meta = load_and_prepare(root)
    logging.info("Prepared %s analytical rows", f"{len(df):,}")
    if retrain:
        logging.info("Training and evaluating models")
        comparison, model, model_meta, test_results = train_and_evaluate(df, root)
    else:
        model_meta = json.loads((root / "models" / "metadata" / "model_metadata.json").read_text(encoding="utf-8"))
        test_results = pd.read_parquet(root / "reports" / "tables" / "test_predictions.parquet")
        comparison = pd.read_csv(root / "reports" / "tables" / "model_comparison.csv")
        model = CatBoostRegressor()
        model.load_model(str(root / "models" / "final_model" / "catboost_fair_price.cbm"))
    df = score_all_listings(df, model, model_meta, root)
    create_eda_outputs(df, root, prep_meta, test_results)
    create_business_report(root, prep_meta, model_meta, df)
    append_autonomous_decisions(root, prep_meta, model_meta)
    state = {
        "status": "Complete",
        "current_phase": "Phase 18 — Quality assurance complete",
        "analytical_rows": int(len(df)),
        "selected_model": model_meta["selected_model"],
        "test_metrics": model_meta["test_metrics"],
        "cleaning_performed": True,
        "modeling_performed": True,
        "dashboard_started": True,
        "last_updated": "2026-06-13",
    }
    (root / "reports" / "project_state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    logging.info("Pipeline complete")
    return {"prep": prep_meta, "model": model_meta, "comparison": comparison.to_dict(orient="records")}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--no-retrain", action="store_true")
    args = parser.parse_args()
    result = run_pipeline(args.root, retrain=not args.no_retrain)
    print(json.dumps(result, indent=2, default=str))
