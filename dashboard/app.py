from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.utils.data_loader import load_market_data, load_opportunities, load_table, load_metadata, load_preparation_summary
from dashboard.utils.model import build_prediction_row, predict_price, local_explanation, estimate_tier, confidence_label, comparable_listings
from dashboard.components.charts import price_distribution, ppsqm_box, location_bar, actual_predicted, apply_premium_style

st.set_page_config(page_title="Egypt Real Estate Intelligence", page_icon="🏙️", layout="wide")

# Custom CSS styling including Sidebar overrides
st.html("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
    /* Global Styles & Animations */
    .main .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        max-width: 1200px !important;
        animation: fadeIn 0.5s ease-out forwards;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Sidebar styling overrides */
    [data-testid="stSidebar"] {
        background-color: #080c14 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
    }
    [data-testid="stSidebar"] [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.02) !important;
        margin-bottom: 12px !important;
        padding: 10px 14px !important;
        border: 1px solid rgba(255, 255, 255, 0.04) !important;
    }
    
    /* Headers styling */
    h1 {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 800 !important;
        background: linear-gradient(135deg, #a5b4fc 0%, #6366f1 50%, #14b8a6 100%);
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        padding-bottom: 0.1em !important;
        margin-bottom: 0.2rem !important;
    }
    h2, h3, h4 {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
        color: #f3f4f6 !important;
    }
    
    /* Card/Container styling */
    div[data-testid="stForm"], 
    div[data-testid="element-container"] .stElementContainerContainer {
        border-radius: 12px !important;
    }
    
    /* Glassmorphic Metric styling */
    [data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.45) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
        padding: 16px 20px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
        transition: all 0.2s ease-in-out !important;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px) !important;
        border-color: rgba(99, 102, 241, 0.4) !important;
        box-shadow: 0 10px 15px -3px rgba(99, 102, 241, 0.1), 0 4px 6px -2px rgba(99, 102, 241, 0.05) !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        font-family: 'Outfit', sans-serif !important;
        color: #f3f4f6 !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        color: #9ca3af !important;
    }
    
    /* Tab Styling */
    [data-baseweb="tab-list"] {
        gap: 8px !important;
        background-color: transparent !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08) !important;
        padding-bottom: 8px !important;
    }
    [data-baseweb="tab"] {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 500 !important;
        background-color: rgba(30, 41, 59, 0.2) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 8px !important;
        padding: 8px 16px !important;
        color: #9ca3af !important;
        transition: all 0.2s ease !important;
    }
    [data-baseweb="tab"]:hover {
        color: #f3f4f6 !important;
        background-color: rgba(99, 102, 241, 0.1) !important;
        border-color: rgba(99, 102, 241, 0.3) !important;
    }
    [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
        color: #ffffff !important;
        border-color: #6366f1 !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 10px rgba(99, 102, 241, 0.3) !important;
    }
    
    /* Styled Selectbox and Inputs */
    div[data-baseweb="select"] > div {
        background-color: rgba(30, 41, 59, 0.45) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 8px !important;
        color: #f3f4f6 !important;
    }
    div[data-baseweb="select"] > div:hover {
        border-color: rgba(99, 102, 241, 0.3) !important;
    }
    
    /* Styled Button */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
        color: white !important;
        border: none !important;
        padding: 10px 24px !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-family: 'Outfit', sans-serif !important;
        box-shadow: 0 4px 6px -1px rgba(99, 102, 241, 0.3), 0 2px 4px -1px rgba(99, 102, 241, 0.2) !important;
        transition: all 0.25s ease !important;
        width: 100% !important;
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #4f46e5 0%, #3730a3 100%) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 10px 15px -3px rgba(99, 102, 241, 0.4), 0 4px 6px -2px rgba(99, 102, 241, 0.3) !important;
    }
    
    /* Clean alert styles */
    div.stAlert {
        background-color: rgba(30, 41, 59, 0.4) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
    }
    div.stAlert [data-testid="stNotificationContent"] {
        color: #e2e8f0 !important;
    }
    
    /* Dataframe wrapping */
    .stDataFrame {
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 12px !important;
        overflow: hidden !important;
    }
</style>
""")

# Load raw datasets
market_raw = load_market_data()
opp_raw = load_opportunities()
loc_raw = load_table("location_summary.csv")
meta = load_metadata()
prep = load_preparation_summary()

# Elegant Sidebar Layout
with st.sidebar:
    st.html("""
    <div style="text-align: center; padding: 15px 0 20px 0; border-bottom: 1px solid rgba(255, 255, 255, 0.08); margin-bottom: 25px;">
        <span style="font-size: 2.8rem; filter: drop-shadow(0 0 12px rgba(99, 102, 241, 0.4));">🏙️</span>
        <h2 style="font-family: 'Outfit', sans-serif; font-weight: 800; font-size: 1.35rem; background: linear-gradient(135deg, #a5b4fc, #6366f1); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 8px 0 2px 0;">ADPilot Egypt</h2>
        <div style="font-size: 0.7rem; color: #9ca3af; letter-spacing: 0.08em; text-transform: uppercase;">Real Estate Intelligence</div>
    </div>
    """)
    
    # Currency Settings
    st.markdown("### Settings")
    selected_currency = st.selectbox("Global Currency Selector", ["EGP", "USD"], index=0)
    conversion_factor = 48.0 if selected_currency == "USD" else 1.0

# Dynamic formatter functions
def fmt(v):
    if pd.isna(v) or v == "—" or v is None: return "—"
    val = float(v) / conversion_factor
    if selected_currency == "USD":
        return f"${val:,.0f}"
    else:
        return f"EGP {val:,.0f}"

def fmt_sqm(v):
    if pd.isna(v) or v == "—" or v is None: return "—"
    val = float(v) / conversion_factor
    if selected_currency == "USD":
        return f"${val:,.0f}/sqm"
    else:
        return f"{val:,.0f} EGP/sqm"

def insight(text): st.info(text)

# Convert datasets based on selected currency
market = market_raw.copy()
opp = opp_raw.copy()
loc = loc_raw.copy()

if selected_currency == "USD":
    market["price_egp"] = market["price_egp"] / 48.0
    market["price_per_sqm"] = market["price_per_sqm"] / 48.0
    market["predicted_fair_price"] = market["predicted_fair_price"] / 48.0
    
    opp["price_egp"] = opp["price_egp"] / 48.0
    opp["predicted_fair_price"] = opp["predicted_fair_price"] / 48.0
    opp["lower_fair_price"] = opp["lower_fair_price"] / 48.0
    opp["upper_fair_price"] = opp["upper_fair_price"] / 48.0
    
    loc["median_price"] = loc["median_price"] / 48.0
    loc["median_price_per_sqm"] = loc["median_price_per_sqm"] / 48.0
    loc["mean_price"] = loc["mean_price"] / 48.0
    loc["mean_price_per_sqm"] = loc["mean_price_per_sqm"] / 48.0

# Update Sidebar context with dynamic metrics
with st.sidebar:
    st.markdown("### Persistent Context")
    st.metric("Total Listings", f"{len(market):,}")
    st.metric("Median Price", fmt(market_raw.price_egp.median()))
    st.metric("Median Price / sqm", fmt_sqm(market_raw.price_per_sqm.median()))
    
    st.html("""
    <div style="margin-top: 35px; padding: 14px; border-radius: 8px; background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.05); font-size: 0.72rem; color: #9ca3af; line-height: 1.45; font-family: 'Inter', sans-serif;">
        <span style="color: #6366f1; font-weight: bold;">SYSTEM STATUS:</span> Online<br>
        <span style="color: #6366f1; font-weight: bold;">ARCHIVE DATE:</span> March 2026<br>
        <span style="color: #6366f1; font-weight: bold;">PREDICTIVE MODEL:</span> CatBoost v1.2<br>
        <span style="color: #6366f1; font-weight: bold;">DATA SOURCE:</span> PropertyFinder EG
    </div>
    """)

st.title("Egyptian Real Estate Market Intelligence")
st.caption(f"Observed asking-price analytics and model estimates from PropertyFinder Egypt listings scraped in March 2026. Displayed in {selected_currency}.")

tabs = st.tabs([
    "Executive Overview",
    "Market Explorer",
    "Location Intelligence",
    "Project Comparison",
    "Fair-Price Predictor",
    "Model Performance",
    "Pricing Opportunities",
    "Data Quality & Methodology"
])

with tabs[0]:
    st.subheader("What is the clearest market story?")
    c = st.columns(6)
    c[0].metric("Listings", f"{len(market):,}")
    c[1].metric("Median Price", fmt(market_raw.price_egp.median()))
    c[2].metric("Mean Price", fmt(market_raw.price_egp.mean()))
    c[3].metric("Median Price/sqm", fmt_sqm(market_raw.price_per_sqm.median()))
    c[4].metric("Towns", f"{market.town.nunique():,}")
    c[5].metric("Named submarkets", f"{market.submarket_or_compound.ne('Unknown').sum():,}")
    
    tiers = market.market_tier.value_counts(normalize=True).reindex(["Affordable","Mid-market","Upper-mid","Luxury"]).fillna(0)
    insight(f"The market is broadly balanced across tiers. Median asking price is {fmt(market_raw.price_egp.median())}; luxury represents {tiers['Luxury']:.1%}. Model estimates are decision support, not appraisals.")
    
    a, b = st.columns(2)
    a.plotly_chart(price_distribution(market, curr=selected_currency), width="stretch")
    
    # Styled Tier Share
    fig_tiers = px.bar(
        tiers.reset_index(),
        x="market_tier",
        y="proportion",
        title="Market-Tier Share",
        labels={"proportion": "Share", "market_tier": "Tier"},
        color="market_tier",
        color_discrete_sequence=["#6366f1", "#14b8a6", "#8b5cf6", "#ec4899"]
    )
    fig_tiers.update_layout(showlegend=False)
    fig_tiers.update_yaxes(tickformat=".0%")
    apply_premium_style(fig_tiers)
    b.plotly_chart(fig_tiers, width="stretch")
    
    st.caption(f"Model test MAE: {fmt(meta['test_metrics']['MAE_EGP'])} · R²: {meta['test_metrics']['R2']:.3f} · Typical median absolute error: {fmt(meta['test_metrics']['MedianAE_EGP'])}")

with tabs[1]:
    st.subheader("How do prices change across property and location filters?")
    with st.container(border=True):
        f1, f2, f3, f4 = st.columns(4)
        region = f1.multiselect("Market region", sorted(market.market_region.unique()), key="explorer_region")
        town_options = sorted(market[market.market_region.isin(region)].town.unique()) if region else sorted(market.town.unique())
        town = f2.multiselect("Town", town_options, key="explorer_town")
        ptype = f3.multiselect("Property type", sorted(market.property_type.unique()), key="explorer_property_type")
        tier = f4.multiselect("Market tier", ["Affordable", "Mid-market", "Upper-mid", "Luxury"], key="explorer_tier")
        
        ranges = st.columns(2)
        amin, amax = float(market.area_sqm.min()), float(market.area_sqm.quantile(.99))
        area_range = ranges[0].slider("Area sqm", amin, amax, (amin, amax))
        pmin, pmax = float(market.price_egp.min()), float(market.price_egp.quantile(.99))
        price_range = ranges[1].slider(f"Price ({selected_currency})", pmin, pmax, (pmin, pmax), step=1000.0 if selected_currency == "USD" else 100000.0)
        
    filtered = market.copy()
    if region: filtered = filtered[filtered.market_region.isin(region)]
    if town: filtered = filtered[filtered.town.isin(town)]
    if ptype: filtered = filtered[filtered.property_type.isin(ptype)]
    if tier: filtered = filtered[filtered.market_tier.isin(tier)]
    filtered = filtered[filtered.area_sqm.between(*area_range) & filtered.price_egp.between(*price_range)]
    
    c = st.columns(4)
    c[0].metric("Filtered listings", f"{len(filtered):,}")
    c[1].metric("Median price", fmt(filtered.price_egp.median() * conversion_factor) if len(filtered) else "—")
    c[2].metric("Median Price/sqm", fmt_sqm(filtered.price_per_sqm.median() * conversion_factor) if len(filtered) else "—")
    c[3].metric("Median area", f"{filtered.area_sqm.median():,.0f} sqm" if len(filtered) else "—")
    
    if len(filtered):
        a, b = st.columns(2)
        a.plotly_chart(price_distribution(filtered, curr=selected_currency), width="stretch")
        b.plotly_chart(ppsqm_box(filtered, curr=selected_currency), width="stretch")
        
        display_cols = {
            "title": "Title",
            "market_region": "Region",
            "town": "Town",
            "district": "District",
            "property_type": "Type",
            "area_sqm": "Area (sqm)",
            "price_egp": f"Price ({selected_currency})",
            "price_per_sqm": f"Price/sqm ({selected_currency})",
            "market_tier": "Tier",
            "model_confidence": "Confidence"
        }
        
        st.dataframe(
            filtered[list(display_cols.keys())].rename(columns=display_cols).sort_values(f"Price ({selected_currency})", ascending=False).head(200),
            column_config={
                "Area (sqm)": st.column_config.NumberColumn(format="%d sqm"),
                f"Price ({selected_currency})": st.column_config.NumberColumn(format="$%d" if selected_currency == "USD" else "EGP %d"),
                f"Price/sqm ({selected_currency})": st.column_config.NumberColumn(format="$%d" if selected_currency == "USD" else "EGP %d")
            },
            width="stretch",
            hide_index=True
        )

with tabs[2]:
    st.subheader("Which locations command the strongest price per sqm?")
    insight("Use locations with adequate sample sizes. Small markets may appear extreme because a few listings dominate their median.")
    st.plotly_chart(location_bar(loc, curr=selected_currency), width="stretch")
    
    heat = market.groupby(["town", "property_type"], observed=True).agg(median_ppsqm=("price_per_sqm", "median"), n=("listing_id", "size")).reset_index()
    top_towns = market.town.value_counts().head(20).index
    heat = heat[heat.town.isin(top_towns) & heat.n.ge(10)]
    pivot = heat.pivot(index="town", columns="property_type", values="median_ppsqm")
    
    fig_heat = px.imshow(
        pivot,
        aspect="auto",
        title=f"Town × Property-Type Median {selected_currency}/sqm",
        labels={"color": f"{selected_currency}/SQM", "town": "Town", "property_type": "Property Type"},
        color_continuous_scale=[[0, "#111827"], [0.5, "#6366f1"], [1, "#14b8a6"]]
    )
    fig_heat.update_layout(coloraxis_colorbar=dict(title=f"{selected_currency}/SQM", tickformat=","))
    apply_premium_style(fig_heat)
    st.plotly_chart(fig_heat, width="stretch")
    
    # Geographic Market Explorer Map
    st.markdown("---")
    st.subheader("🗺️ Geographic Market Explorer")
    st.markdown("Explore listing density and pricing tiers across Egypt. Circle size corresponds to the property area, and color represents the market tier.")
    
    with st.container(border=True):
        m1, m2 = st.columns(2)
        map_region = m1.multiselect("Map Region Filter", sorted(market.market_region.unique()), key="map_region")
        map_town_opts = sorted(market[market.market_region.isin(map_region)].town.unique()) if map_region else sorted(market.town.unique())
        map_town = m2.multiselect("Map Town Filter", map_town_opts, key="map_town")
        
    map_df = market.copy()
    if map_region: map_df = map_df[map_df.market_region.isin(map_region)]
    if map_town: map_df = map_df[map_df.town.isin(map_town)]
    
    map_df = map_df.dropna(subset=["lat", "lon"])
    if len(map_df) > 0:
        # Sample to avoid lag
        if len(map_df) > 3000:
            map_df = map_df.sample(3000, random_state=42)
            
        fig_map = px.scatter_mapbox(
            map_df,
            lat="lat",
            lon="lon",
            color="market_tier",
            size="area_sqm",
            hover_name="title",
            hover_data={
                "price_egp": True,
                "price_per_sqm": True,
                "area_sqm": True,
                "town": True
            },
            color_discrete_map={
                "Affordable": "#a5b4fc",
                "Mid-market": "#6366f1",
                "Upper-mid": "#8b5cf6",
                "Luxury": "#ec4899"
            },
            zoom=9 if map_town else 6,
            center=dict(lat=map_df.lat.median(), lon=map_df.lon.median()),
            title="Geographic Listing Map",
            mapbox_style="open-street-map",
            labels={
                "price_egp": f"Price ({selected_currency})",
                "price_per_sqm": f"Price/sqm ({selected_currency})",
                "market_tier": "Market Tier",
                "area_sqm": "Area (sqm)"
            }
        )
        apply_premium_style(fig_map)
        st.plotly_chart(fig_map, width="stretch")
    else:
        st.info("No geocoded properties found for the selected location filters.")
        
    loc_display = loc.copy()
    loc_display.columns = [c.replace("_", " ").title() for c in loc_display.columns]
    
    st.dataframe(
        loc_display.head(100),
        column_config={
            "Median Price": st.column_config.NumberColumn(format="$%d" if selected_currency == "USD" else "EGP %d"),
            "Mean Price": st.column_config.NumberColumn(format="$%d" if selected_currency == "USD" else "EGP %d"),
            "Median Price Per Sqm": st.column_config.NumberColumn(format="$%d" if selected_currency == "USD" else "EGP %d"),
            "Median Area": st.column_config.NumberColumn(format="%d sqm"),
            "Listings": st.column_config.NumberColumn(format="%d")
        },
        width="stretch",
        hide_index=True
    )

with tabs[3]:
    st.subheader("How do named submarkets or compounds compare?")
    st.warning("The source has no reliable developer field. This page compares the mixed `submarket_or_compound` field and should not be interpreted as a definitive project registry.")
    counts = market[market.submarket_or_compound.ne("Unknown")].submarket_or_compound.value_counts()
    options = counts[counts>=10].index.tolist()
    selected = st.multiselect("Select up to five", options[:300], default=options[:3], max_selections=5)
    if selected:
        d = market[market.submarket_or_compound.isin(selected)]
        summary = d.groupby("submarket_or_compound", observed=True).agg(
            listings=("listing_id", "size"),
            median_price=("price_egp", "median"),
            median_ppsqm=("price_per_sqm", "median"),
            property_types=("property_type", "nunique"),
            luxury_share=("market_tier", lambda s: (s=="Luxury").mean()),
            median_fair_price=("predicted_fair_price", "median")
        ).reset_index()
        
        summary_display = summary.copy()
        summary_display.columns = [c.replace("_", " ").title() for c in summary_display.columns]
        
        st.dataframe(
            summary_display,
            column_config={
                "Listings": st.column_config.NumberColumn(format="%d"),
                "Median Price": st.column_config.NumberColumn(format="$%d" if selected_currency == "USD" else "EGP %d"),
                "Median Ppsqm": st.column_config.NumberColumn(format="$%d" if selected_currency == "USD" else "EGP %d"),
                "Property Types": st.column_config.NumberColumn(format="%d"),
                "Luxury Share": st.column_config.NumberColumn(format="%.1%"),
                "Median Fair Price": st.column_config.NumberColumn(format="$%d" if selected_currency == "USD" else "EGP %d")
            },
            width="stretch",
            hide_index=True
        )
        
        # Compound Bubble Chart
        fig_bubble = px.scatter(
            summary,
            x="listings",
            y="median_price",
            size="median_ppsqm",
            color="submarket_or_compound",
            hover_name="submarket_or_compound",
            title=f"Compound Market Position (Bubble Size = {selected_currency} per SQM)",
            labels={"listings": "Number of Listings", "median_price": f"Median Price ({selected_currency})", "submarket_or_compound": "Compound"},
            color_discrete_sequence=["#6366f1", "#14b8a6", "#8b5cf6", "#ec4899", "#f59e0b"]
        )
        fig_bubble.update_yaxes(tickformat=",")
        apply_premium_style(fig_bubble)
        st.plotly_chart(fig_bubble, width="stretch")
        
        fig_box = px.box(
            d,
            x="submarket_or_compound",
            y="price_per_sqm",
            color="property_type",
            points=False,
            title="Price-per-SQM Dispersion",
            labels={"price_per_sqm": f"{selected_currency} per SQM", "submarket_or_compound": "Submarket/Compound", "property_type": "Property Type"},
            color_discrete_sequence=["#6366f1", "#14b8a6", "#8b5cf6", "#ec4899", "#f59e0b"]
        )
        apply_premium_style(fig_box)
        st.plotly_chart(fig_box, width="stretch")
        best = summary.sort_values("median_ppsqm", ascending=False).iloc[0]
        insight(f"{best.submarket_or_compound} has the highest median price per sqm among the selected entities, based on {int(best.listings)} listings. Compare sample size and property mix before acting.")

with tabs[4]:
    st.subheader("What is a reasonable asking-price range for a property?")
    with st.container(border=True):
        left, right = st.columns(2)
        region = left.selectbox("Market region", sorted(market.market_region.unique()))
        town_opts = sorted(market.loc[market.market_region.eq(region), "town"].unique())
        town = right.selectbox("Town", town_opts)
        town_df = market[market.town.eq(town)]
        district = left.selectbox("District", sorted(town_df.district.unique()))
        sub_opts = sorted(town_df.submarket_or_compound.unique())
        submarket = right.selectbox("Submarket/compound", sub_opts)
        ptype = left.selectbox("Property type", sorted(market.property_type.unique()))
        area = right.number_input("Area sqm", min_value=21.0, max_value=2000.0, value=float(market[market.property_type.eq(ptype)].area_sqm.median()), step=5.0)
        beds = left.number_input("Bedrooms", min_value=0.0, max_value=20.0, value=3.0, step=1.0)
        baths = right.number_input("Bathrooms", min_value=0.0, max_value=20.0, value=2.0, step=1.0)
        furnished = left.selectbox("Furnished", sorted(market.furnished.unique()))
        payment = right.selectbox("Payment method", sorted(market.payment_method.unique()))
        completion = left.selectbox("Completion status", sorted(market.completion_status.unique()))
        listing_level = right.selectbox("Listing level", sorted(market.listing_level.unique()))
        
    if st.button("Estimate fair price", type="primary"):
        med = town_df["listing_age_days"].median()
        inputs = {
            "market_region": region,
            "town": town,
            "district": district,
            "submarket_or_compound": submarket,
            "property_type": ptype,
            "area_sqm": area,
            "bedrooms_num": beds,
            "bathrooms_num": baths,
            "furnished": furnished,
            "payment_method": payment,
            "completion_status": completion,
            "listing_level": listing_level,
            "new_city_indicator": town_df.new_city_indicator.mode().iloc[0] if not town_df.new_city_indicator.mode().empty else "No",
            "compound_status": "Broad/standalone" if submarket=="Unknown" or "Compounds" in submarket else "Named submarket/compound",
            "listing_age_days": float(med),
            "lat": float(town_df.lat.median()),
            "lon": float(town_df.lon.median()),
            "images_count": float(town_df.images_count.median()),
            "amenities_count": float(town_df.amenities_count.median()),
            "is_premium": 0,
            "is_featured": 0,
            "has_view_360": 0
        }
        
        # Build raw prediction row in EGP (model expects EGP features and outputs)
        row = build_prediction_row(inputs, meta, market_raw)
        pred = predict_price(row, meta)
        
        # Suggested Tier
        tier_name, tier_score = estimate_tier(pred["predicted"] / area, town, ptype, market_raw)
        conf, flags = confidence_label(row, meta, market_raw)
        
        # Premium Glowing Card for Estimation Results
        st.html(f"""
        <div style="background: linear-gradient(135deg, rgba(99, 102, 241, 0.12) 0%, rgba(20, 184, 166, 0.12) 100%); border: 1px solid rgba(99, 102, 241, 0.25); border-radius: 16px; padding: 28px; margin-bottom: 24px; text-align: center; box-shadow: 0 10px 30px -10px rgba(99, 102, 241, 0.3);">
            <div style="font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.15em; color: #a5b4fc; margin-bottom: 8px;">Estimated Fair Price</div>
            <div style="font-size: 2.8rem; font-weight: 800; color: #ffffff; font-family: 'Outfit', sans-serif; text-shadow: 0 0 15px rgba(99, 102, 241, 0.4); margin-bottom: 8px;">{fmt(pred["predicted"])}</div>
            <div style="font-size: 0.95rem; color: #e2e8f0; margin-bottom: 24px; font-family: 'Inter', sans-serif;">Reliability: <span style="color: {'#10b981' if conf == 'High' else '#f59e0b' if conf == 'Medium' else '#ef4444'}; font-weight: 700;">{conf}</span></div>
            <div style="display: flex; justify-content: space-around; flex-wrap: wrap; margin-top: 16px; border-top: 1px solid rgba(255, 255, 255, 0.08); padding-top: 20px;">
                <div style="padding: 8px 16px;">
                    <div style="font-size: 0.75rem; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">Empirical Range (90% CI)</div>
                    <div style="font-size: 1.15rem; font-weight: 600; color: #f3f4f6;">{fmt(pred['lower'])} – {fmt(pred['upper'])}</div>
                </div>
                <div style="padding: 8px 16px;">
                    <div style="font-size: 0.75rem; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">Estimated Price/SQM</div>
                    <div style="font-size: 1.15rem; font-weight: 600; color: #f3f4f6;">{fmt_sqm(pred['predicted']/area)}</div>
                </div>
                <div style="padding: 8px 16px;">
                    <div style="font-size: 0.75rem; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">Suggested Tier</div>
                    <div style="font-size: 1.15rem; font-weight: 600; color: #f3f4f6;">{tier_name}</div>
                </div>
            </div>
        </div>
        """)
        
        if flags: st.warning("; ".join(flags))
        exp = local_explanation(row, meta).head(8)
        
        # Convert SHAP log values to percentage impact on total price
        exp["pct_impact"] = (np.exp(exp["impact_log_price"]) - 1) * 100
        
        # Build beautiful visual presentation
        contrib_rows = []
        for idx, r in exp.iterrows():
            val = r["pct_impact"]
            color = "#14b8a6" if val < 0 else "#ec4899"
            badge_bg = "rgba(20, 184, 166, 0.12)" if val < 0 else "rgba(236, 72, 153, 0.12)"
            sign = "" if val < 0 else "+"
            
            friendly_names = {
                "market_region": "Market Region",
                "town": "Town",
                "district": "District",
                "submarket_or_compound": "Submarket/Compound",
                "property_type": "Property Type",
                "area_sqm": "Area (sqm)",
                "bedrooms_num": "Bedrooms Count",
                "bathrooms_num": "Bathrooms Count",
                "furnished": "Furnishing Status",
                "payment_method": "Payment Type",
                "completion_status": "Completion Status",
                "listing_level": "Listing Service Level",
                "new_city_indicator": "New City Status",
                "compound_status": "Compound Classification",
                "listing_age_days": "Listing Age (days)",
                "lat": "Latitude",
                "lon": "Longitude",
                "images_count": "Photos Count",
                "amenities_count": "Amenities Count"
            }
            fname = friendly_names.get(r['feature'], r['feature'])
            
            contrib_rows.append(f"""
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 12px 16px; border-bottom: 1px solid rgba(255, 255, 255, 0.05); transition: background 0.2s;">
                <div>
                    <div style="font-size: 0.9rem; font-weight: 600; color: #f3f4f6; font-family: 'Outfit', sans-serif;">{fname}</div>
                    <div style="font-size: 0.75rem; color: #9ca3af; font-family: 'Inter', sans-serif;">Value: <span style="color: #cbd5e1; font-weight: 600;">{r['value']}</span></div>
                </div>
                <div style="background: {badge_bg}; color: {color}; font-size: 0.85rem; font-weight: 700; padding: 6px 12px; border-radius: 6px; border: 1px solid {color}25; font-family: 'Outfit', sans-serif;">
                    {sign}{val:.1f}%
                </div>
            </div>
            """)
        
        contrib_html = f"""
        <div style="background: rgba(30, 41, 59, 0.25); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; overflow: hidden; margin-top: 10px;">
            {''.join(contrib_rows)}
        </div>
        """
        
        # Build beautiful listing cards
        comps = comparable_listings(row, market, 8)
        comps_html = []
        for idx, r in comps.iterrows():
            sim_pct = r["similarity_score"] * 100
            price_str = fmt(r['price_egp'] * conversion_factor)
            ppsqm_str = fmt_sqm(r['price_per_sqm'] * conversion_factor)
            
            tier_color = "#6366f1" if r["market_tier"] == "Luxury" else "#14b8a6" if r["market_tier"] == "Upper-mid" else "#8b5cf6" if r["market_tier"] == "Mid-market" else "#9ca3af"
            tier_badge = f"<span style='background: {tier_color}18; color: {tier_color}; border: 1px solid {tier_color}30; font-size: 0.65rem; font-weight: 700; padding: 2px 6px; border-radius: 4px; text-transform: uppercase;'>{r['market_tier']}</span>"
            
            spec_str = f"{int(r['bedrooms_num'])} Beds · {int(r['bathrooms_num'])} Baths · {int(r['area_sqm'])} sqm"
            loc_str = f"{r['town']}, {r['district']}"
            if r['submarket_or_compound'] != "Unknown":
                loc_str += f" ({r['submarket_or_compound']})"
            
            comps_html.append(f"""
            <div style="background: rgba(30, 41, 59, 0.25); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 16px; margin-bottom: 12px; transition: transform 0.2s, border-color 0.2s;">
                <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 6px;">
                    <div style="font-family: 'Outfit', sans-serif; font-size: 0.95rem; font-weight: 700; color: #f3f4f6; max-width: 70%; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;" title="{r['title']}">{r['title']}</div>
                    {tier_badge}
                </div>
                <div style="font-family: 'Inter', sans-serif; font-size: 0.75rem; color: #9ca3af; margin-bottom: 10px;">
                    📍 {loc_str}<br>
                    📏 {spec_str}
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; border-top: 1px solid rgba(255, 255, 255, 0.05); padding-top: 10px; margin-top: 8px;">
                    <div>
                        <div style="font-family: 'Outfit', sans-serif; font-size: 1.1rem; font-weight: 700; color: #ffffff;">{price_str}</div>
                        <div style="font-family: 'Inter', sans-serif; font-size: 0.7rem; color: #9ca3af;">{ppsqm_str}</div>
                    </div>
                    <div style="width: 40%; text-align: right;">
                        <div style="display: flex; align-items: center; justify-content: flex-end; gap: 6px; margin-bottom: 4px;">
                            <span style="font-family: 'Outfit', sans-serif; font-size: 0.75rem; color: #a5b4fc; font-weight: 700;">{sim_pct:.0f}% match</span>
                        </div>
                        <div style="width: 100%; height: 5px; background: rgba(255,255,255,0.06); border-radius: 3px; overflow: hidden;">
                            <div style="width: {sim_pct:.0f}%; height: 100%; background: linear-gradient(90deg, #6366f1, #14b8a6); border-radius: 3px;"></div>
                        </div>
                    </div>
                </div>
            </div>
            """)
        comps_list_html = f"<div style='margin-top: 10px;'>{''.join(comps_html)}</div>"
        
        a, b = st.columns(2)
        with a:
            st.subheader("Feature contribution to estimate")
            st.html(contrib_html)
        with b:
            st.subheader("Comparable market listings")
            st.html(comps_list_html)
            
        # Comparable Listings Map
        st.markdown("---")
        st.subheader("🗺️ Comparable Listings Map")
        st.markdown("Visual spatial distribution of comparable listings relative to the estimated subject property location.")
        subj_lat = float(row.iloc[0]["lat"])
        subj_lon = float(row.iloc[0]["lon"])
        
        fig_comp_map = px.scatter_mapbox(
            comps,
            lat="lat",
            lon="lon",
            color="similarity_score",
            size="area_sqm",
            hover_name="title",
            hover_data={
                "price_egp": True,
                "price_per_sqm": True,
                "similarity_score": True
            },
            color_continuous_scale=[[0, "#14b8a6"], [1, "#6366f1"]],
            zoom=13,
            center=dict(lat=subj_lat, lon=subj_lon),
            mapbox_style="open-street-map",
            labels={
                "price_egp": f"Price ({selected_currency})",
                "price_per_sqm": f"Price/sqm ({selected_currency})",
                "similarity_score": "Similarity",
                "area_sqm": "Area (sqm)"
            }
        )
        # Overlay subject property as a larger glowing marker
        fig_comp_map.add_trace(go.Scattermapbox(
            lat=[subj_lat],
            lon=[subj_lon],
            mode="markers",
            marker=go.scattermapbox.Marker(
                size=18,
                color="#ec4899",
                opacity=0.9
            ),
            name="Subject Property",
            text=["Subject Property"],
            hoverinfo="text"
        ))
        apply_premium_style(fig_comp_map)
        st.plotly_chart(fig_comp_map, width="stretch")
        
        # Valuation Exporter CSV
        st.markdown("---")
        st.subheader("📥 Export Valuation Report")
        st.markdown("Download a structured CSV report summarizing this valuation run and the associated comparables.")
        
        export_data = []
        export_data.append({
            "Type": "Subject Property",
            "Title": f"Valuation for {ptype} in {district}, {town}",
            "Price": pred["predicted"] / conversion_factor,
            "Area (sqm)": area,
            "Bedrooms": beds,
            "Bathrooms": baths,
            "Town": town,
            "District": district,
            "Submarket/Compound": submarket,
            "Similarity": 1.0,
            "Detail URL": ""
        })
        for idx, r in comps.iterrows():
            export_data.append({
                "Type": "Comparable Listing",
                "Title": r["title"],
                "Price": r["price_egp"],
                "Area (sqm)": r["area_sqm"],
                "Bedrooms": r["bedrooms_num"],
                "Bathrooms": r["bathrooms_num"],
                "Town": r["town"],
                "District": r["district"],
                "Submarket/Compound": r["submarket_or_compound"],
                "Similarity": r["similarity_score"],
                "Detail URL": r["detail_url"]
            })
        export_df = pd.DataFrame(export_data)
        
        st.download_button(
            label="Download Valuation Report (CSV)",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name=f"valuation_report_{town}_{district}.csv",
            mime="text/csv",
            key="download_val_report"
        )

with tabs[5]:
    st.subheader("How accurate and reliable is the model?")
    comparison = load_table("model_comparison.csv")
    comparison_display = comparison.copy()
    
    # Apply currency division to evaluation metrics in model_comparison table
    if selected_currency == "USD":
        comparison_display["Validation MAE"] = comparison_display["Validation MAE"] / 48.0
        comparison_display["Validation RMSE"] = comparison_display["Validation RMSE"] / 48.0
        comparison_display["Median absolute error"] = comparison_display["Median absolute error"] / 48.0
        
    st.dataframe(
        comparison_display,
        column_config={
            "Validation MAE": st.column_config.NumberColumn(format="$%d" if selected_currency == "USD" else "EGP %d"),
            "Validation RMSE": st.column_config.NumberColumn(format="$%d" if selected_currency == "USD" else "EGP %d"),
            "Validation R2": st.column_config.NumberColumn(format="%.3f"),
            "Median absolute error": st.column_config.NumberColumn(format="$%d" if selected_currency == "USD" else "EGP %d"),
            "Validation MAPE": st.column_config.NumberColumn(format="%.2%"),
            "Training seconds": st.column_config.NumberColumn(format="%.3f s"),
            "Prediction seconds": st.column_config.NumberColumn(format="%.4f s")
        },
        width="stretch",
        hide_index=True
    )
    
    insight("Random Forest had the lowest validation MAE. CatBoost was retained as the deployable model because it produced lower median and percentage error and handles high-cardinality categories natively. This trade-off is documented rather than hidden.")
    test = pd.read_parquet(ROOT / "reports" / "tables" / "test_predictions.parquet")
    
    # Apply currency division to test dataset for charts
    if selected_currency == "USD":
        test["price_egp"] = test["price_egp"] / 48.0
        test["predicted_fair_price"] = test["predicted_fair_price"] / 48.0
        
    a, b = st.columns(2)
    a.plotly_chart(actual_predicted(test, curr=selected_currency), width="stretch")
    
    fig_err = px.histogram(
        test,
        x="absolute_percentage_error",
        nbins=50,
        range_x=[0, float(test.absolute_percentage_error.quantile(.95))],
        title="Test Absolute Percentage Error (APE) Distribution",
        labels={"absolute_percentage_error": "Absolute Percentage Error"},
        color_discrete_sequence=["#14b8a6"]
    )
    fig_err.update_traces(marker_line_color="#090d16", marker_line_width=0.5, opacity=0.85)
    apply_premium_style(fig_err)
    b.plotly_chart(fig_err, width="stretch")
    
    fi = load_table("shap_global_importance.csv").head(15).sort_values("mean_abs_shap_log_price")
    fig_fi = px.bar(
        fi,
        x="mean_abs_shap_log_price",
        y="feature",
        orientation="h",
        title="Global Feature Importance (SHAP)",
        labels={"mean_abs_shap_log_price": "Mean |SHAP Value| (log price impact)", "feature": "Feature"},
        color="mean_abs_shap_log_price",
        color_continuous_scale=[[0, "#6366f1"], [1, "#ec4899"]]
    )
    fig_fi.update_layout(coloraxis_showscale=False)
    apply_premium_style(fig_fi)
    st.plotly_chart(fig_fi, width="stretch")
    
    st.caption("SHAP describes predictive association, not causation.")
    c = st.columns(4)
    c[0].metric("Test MAE", fmt(meta['test_metrics']['MAE_EGP']))
    c[1].metric("Test RMSE", fmt(meta['test_metrics']['RMSE_EGP']))
    c[2].metric("R²", f"{meta['test_metrics']['R2']:.3f}")
    c[3].metric("Median AE", fmt(meta['test_metrics']['MedianAE_EGP']))
    
    # Feature Correlation Matrix
    st.markdown("---")
    st.subheader("📊 Feature Correlation Matrix")
    st.markdown("Linear correlations between key numeric characteristics across all listings (Pearson correlation coefficient).")
    
    corr_cols = ["price_egp", "area_sqm", "bedrooms_num", "bathrooms_num", "listing_age_days"]
    corr_df = market[corr_cols].corr(method="pearson")
    friendly_names = {
        "price_egp": f"Price ({selected_currency})",
        "area_sqm": "Area (sqm)",
        "bedrooms_num": "Bedrooms",
        "bathrooms_num": "Bathrooms",
        "listing_age_days": "Listing Age (Days)"
    }
    corr_df = corr_df.rename(index=friendly_names, columns=friendly_names)
    
    fig_corr = px.imshow(
        corr_df,
        text_auto=".2f",
        color_continuous_scale=[[0, "#ec4899"], [0.5, "#1e293b"], [1, "#14b8a6"]],
        title="Pearson Correlation Heatmap",
        labels={"color": "Correlation Coefficient"}
    )
    apply_premium_style(fig_corr)
    st.plotly_chart(fig_corr, width="stretch")

with tabs[6]:
    st.subheader("Which listings differ materially from the model-estimated range?")
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        status = c1.multiselect("Status", sorted(opp.pricing_status.unique()), default=sorted(opp.pricing_status.unique()), key="opportunity_status")
        confidence = c2.multiselect("Confidence", ["High", "Medium", "Low"], default=["High", "Medium"], key="opportunity_confidence")
        ptypes = c3.multiselect("Property type", sorted(opp.property_type.unique()), key="opportunity_property_type")
        
    d = opp[opp.pricing_status.isin(status) & opp.model_confidence.isin(confidence)]
    if ptypes: d = d[d.property_type.isin(ptypes)]
    d = d.sort_values("opportunity_magnitude", ascending=False)
    
    opp_display = d.copy()
    opp_display = opp_display[["title", "town", "district", "property_type", "area_sqm", "price_egp", "predicted_fair_price", "actual_vs_fair_pct", "opportunity_magnitude", "pricing_status", "model_confidence"]]
    display_cols = {
        "title": "Title",
        "town": "Town",
        "district": "District",
        "property_type": "Type",
        "area_sqm": "Area (sqm)",
        "price_egp": f"Price ({selected_currency})",
        "predicted_fair_price": f"Estimated Fair Price ({selected_currency})",
        "actual_vs_fair_pct": "Difference %",
        "opportunity_magnitude": "Opportunity Magnitude",
        "pricing_status": "Status",
        "model_confidence": "Confidence"
    }
    opp_display = opp_display.rename(columns=display_cols)
    
    st.dataframe(
        opp_display.head(500),
        column_config={
            "Area (sqm)": st.column_config.NumberColumn(format="%d sqm"),
            f"Price ({selected_currency})": st.column_config.NumberColumn(format="$%d" if selected_currency == "USD" else "EGP %d"),
            f"Estimated Fair Price ({selected_currency})": st.column_config.NumberColumn(format="$%d" if selected_currency == "USD" else "EGP %d"),
            "Difference %": st.column_config.NumberColumn(format="%.1f%%"),
            "Opportunity Magnitude": st.column_config.NumberColumn(format="%.2f")
        },
        width="stretch",
        hide_index=True
    )
    st.download_button("Download filtered opportunities", d.to_csv(index=False).encode("utf-8"), "pricing_opportunities.csv", "text/csv")
    st.caption("Large model differences can reflect missing property details, unusual listings, stale advertisements, or genuine pricing opportunities. Manual verification is required.")

with tabs[7]:
    st.subheader("Data Preparation & Scope Metrics")
    st.markdown("Detailed breakdown of the data cleaning pipeline, exclusions, and analytical parameters.")
    
    # Beautifully styled metrics cards
    c1, c2, c3 = st.columns(3)
    c1.metric("Raw Rows Scraped", f"{prep['source_rows_after_d007']:,}")
    c2.metric("Final Analytical Scope", f"{prep['final_analytical_rows']:,}")
    c3.metric("Excluded Records", f"{prep['excluded_rows_total']:,}")
    
    c4, c5, c6 = st.columns(3)
    c4.metric("Recovered Area Rows", f"{prep['recovered_area_rows']:,}")
    c5.metric("Price Tier Method", f"{prep['price_tier_method']}")
    c6.metric("Target Variable", "log(asking_price)")
    
    st.markdown("### Methodology & Limitations")
    st.markdown("""
    - **Split Strategy:** Group-aware 70/15/15 split using named compound or town + district fallback.
    - **Exclusions:** Retained in audit parquet files with reason logging.
    - **Data Limitations:** Asking prices are from listing advertisements and do not represent final completed sales transaction registries. Developer identity is absent. Geographic fields mix official administrative names and developer marketing names. Market conditions may change.
    """)
    
    st.markdown("### Decision Log")
    st.markdown("Formal decision log documenting architectural selections, data pipeline steps, and reasoning.")
    decision_log = pd.read_csv(ROOT / "reports" / "decision_log.csv")
    decision_log.columns = [c.replace("_", " ").title() for c in decision_log.columns]
    st.dataframe(decision_log, width="stretch", hide_index=True)
