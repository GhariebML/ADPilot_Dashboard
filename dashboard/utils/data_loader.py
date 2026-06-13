from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]

@st.cache_data(show_spinner=False)
def load_market_data() -> pd.DataFrame:
    return pd.read_parquet(ROOT / "data" / "dashboard" / "market_listings.parquet")

@st.cache_data(show_spinner=False)
def load_opportunities() -> pd.DataFrame:
    return pd.read_parquet(ROOT / "data" / "dashboard" / "pricing_opportunities.parquet")

@st.cache_data(show_spinner=False)
def load_table(name: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / "reports" / "tables" / name)

@st.cache_data(show_spinner=False)
def load_metadata() -> dict:
    return json.loads((ROOT / "models" / "metadata" / "model_metadata.json").read_text(encoding="utf-8"))

@st.cache_data(show_spinner=False)
def load_preparation_summary() -> dict:
    return json.loads((ROOT / "reports" / "data_preparation_summary.json").read_text(encoding="utf-8"))
