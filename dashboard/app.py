from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from dashboard.utils.data_loader import load_market_data, load_opportunities, load_table, load_metadata, load_preparation_summary
from dashboard.utils.model import build_prediction_row, predict_price, local_explanation, estimate_tier, confidence_label, comparable_listings
from dashboard.components.charts import price_distribution, ppsqm_box, location_bar, actual_predicted, apply_premium_style

st.set_page_config(page_title="Egypt Real Estate Intelligence", page_icon="🏙️", layout="wide")

# Custom CSS styling
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
    /* Global Styles */
    .main .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        max-width: 1200px !important;
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
""", unsafe_allow_html=True)

st.title("Egyptian Real Estate Market Intelligence")
st.caption("Observed asking-price analytics and model estimates from PropertyFinder Egypt listings scraped in March 2026.")

market=load_market_data(); meta=load_metadata(); prep=load_preparation_summary()

def egp(v): return f"EGP {v:,.0f}"
def insight(text): st.info(text)

tabs=st.tabs([
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
    c=st.columns(6)
    c[0].metric("Listings",f"{len(market):,}")
    c[1].metric("Median price",egp(market.price_egp.median()))
    c[2].metric("Mean price",egp(market.price_egp.mean()))
    c[3].metric("Median EGP/sqm",f"{market.price_per_sqm.median():,.0f}")
    c[4].metric("Towns",f"{market.town.nunique():,}")
    c[5].metric("Named submarkets",f"{market.submarket_or_compound.ne('Unknown').sum():,}")
    
    tiers=market.market_tier.value_counts(normalize=True).reindex(["Affordable","Mid-market","Upper-mid","Luxury"]).fillna(0)
    insight(f"The market is broadly balanced across tiers. Median asking price is {egp(market.price_egp.median())}; luxury represents {tiers['Luxury']:.1%}. Model estimates are decision support, not appraisals.")
    
    a,b=st.columns(2)
    a.plotly_chart(price_distribution(market),width="stretch")
    
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
    
    st.caption(f"Model test MAE: {egp(meta['test_metrics']['MAE_EGP'])} · R²: {meta['test_metrics']['R2']:.3f} · Typical median absolute error: {egp(meta['test_metrics']['MedianAE_EGP'])}")

with tabs[1]:
    st.subheader("How do prices change across property and location filters?")
    with st.container(border=True):
        f1,f2,f3,f4=st.columns(4)
        region=f1.multiselect("Market region",sorted(market.market_region.unique()), key="explorer_region")
        town_options=sorted(market[market.market_region.isin(region)].town.unique()) if region else sorted(market.town.unique())
        town=f2.multiselect("Town",town_options, key="explorer_town")
        ptype=f3.multiselect("Property type",sorted(market.property_type.unique()), key="explorer_property_type")
        tier=f4.multiselect("Market tier",["Affordable","Mid-market","Upper-mid","Luxury"], key="explorer_tier")
        
        ranges=st.columns(2)
        amin,amax=float(market.area_sqm.min()),float(market.area_sqm.quantile(.99))
        area_range=ranges[0].slider("Area sqm",amin,amax,(amin,amax))
        pmin,pmax=float(market.price_egp.min()),float(market.price_egp.quantile(.99))
        price_range=ranges[1].slider("Price EGP",pmin,pmax,(pmin,pmax),step=100000.0)
        
    filtered=market.copy()
    if region: filtered=filtered[filtered.market_region.isin(region)]
    if town: filtered=filtered[filtered.town.isin(town)]
    if ptype: filtered=filtered[filtered.property_type.isin(ptype)]
    if tier: filtered=filtered[filtered.market_tier.isin(tier)]
    filtered=filtered[filtered.area_sqm.between(*area_range)&filtered.price_egp.between(*price_range)]
    
    c=st.columns(4)
    c[0].metric("Filtered listings",f"{len(filtered):,}")
    c[1].metric("Median price",egp(filtered.price_egp.median()) if len(filtered) else "—")
    c[2].metric("Median EGP/sqm",f"{filtered.price_per_sqm.median():,.0f}" if len(filtered) else "—")
    c[3].metric("Median area",f"{filtered.area_sqm.median():,.0f} sqm" if len(filtered) else "—")
    
    if len(filtered):
        a,b=st.columns(2)
        a.plotly_chart(price_distribution(filtered),width="stretch")
        b.plotly_chart(ppsqm_box(filtered),width="stretch")
        st.dataframe(filtered[["title","market_region","town","district","property_type","area_sqm","price_egp","price_per_sqm","market_tier","model_confidence"]].sort_values("price_egp",ascending=False).head(200),width="stretch",hide_index=True)

with tabs[2]:
    st.subheader("Which locations command the strongest price per sqm?")
    loc=load_table("location_summary.csv")
    insight("Use locations with adequate sample sizes. Small markets may appear extreme because a few listings dominate their median.")
    st.plotly_chart(location_bar(loc),width="stretch")
    
    heat=market.groupby(["town","property_type"],observed=True).agg(median_ppsqm=("price_per_sqm","median"),n=("listing_id","size")).reset_index()
    top_towns=market.town.value_counts().head(20).index
    heat=heat[heat.town.isin(top_towns)&heat.n.ge(10)]
    pivot=heat.pivot(index="town",columns="property_type",values="median_ppsqm")
    
    fig_heat = px.imshow(
        pivot,
        aspect="auto",
        title="Town × Property-Type Median EGP/sqm",
        labels={"color": "EGP/SQM", "town": "Town", "property_type": "Property Type"},
        color_continuous_scale=[[0, "#111827"], [0.5, "#6366f1"], [1, "#14b8a6"]]
    )
    fig_heat.update_layout(coloraxis_colorbar=dict(title="EGP/SQM", tickformat=","))
    apply_premium_style(fig_heat)
    st.plotly_chart(fig_heat, width="stretch")
    st.dataframe(loc.head(100),width="stretch",hide_index=True)

with tabs[3]:
    st.subheader("How do named submarkets or compounds compare?")
    st.warning("The source has no reliable developer field. This page compares the mixed `submarket_or_compound` field and should not be interpreted as a definitive project registry.")
    counts=market[market.submarket_or_compound.ne("Unknown")].submarket_or_compound.value_counts()
    options=counts[counts>=10].index.tolist()
    selected=st.multiselect("Select up to five",options[:300],default=options[:3],max_selections=5)
    if selected:
        d=market[market.submarket_or_compound.isin(selected)]
        summary=d.groupby("submarket_or_compound",observed=True).agg(listings=("listing_id","size"),median_price=("price_egp","median"),median_ppsqm=("price_per_sqm","median"),property_types=("property_type","nunique"),luxury_share=("market_tier",lambda s:(s=="Luxury").mean()),median_fair_price=("predicted_fair_price","median")).reset_index()
        st.dataframe(summary,width="stretch",hide_index=True)
        
        fig_box = px.box(
            d,
            x="submarket_or_compound",
            y="price_per_sqm",
            color="property_type",
            points=False,
            title="Price-per-SQM Dispersion",
            labels={"price_per_sqm": "EGP per SQM", "submarket_or_compound": "Submarket/Compound", "property_type": "Property Type"},
            color_discrete_sequence=["#6366f1", "#14b8a6", "#8b5cf6", "#ec4899", "#f59e0b"]
        )
        apply_premium_style(fig_box)
        st.plotly_chart(fig_box,width="stretch")
        best=summary.sort_values("median_ppsqm",ascending=False).iloc[0]
        insight(f"{best.submarket_or_compound} has the highest median price per sqm among the selected entities, based on {int(best.listings)} listings. Compare sample size and property mix before acting.")

with tabs[4]:
    st.subheader("What is a reasonable asking-price range for a property?")
    with st.container(border=True):
        left,right=st.columns(2)
        region=left.selectbox("Market region",sorted(market.market_region.unique()))
        town_opts=sorted(market.loc[market.market_region.eq(region),"town"].unique())
        town=right.selectbox("Town",town_opts)
        town_df=market[market.town.eq(town)]
        district=left.selectbox("District",sorted(town_df.district.unique()))
        sub_opts=sorted(town_df.submarket_or_compound.unique())
        submarket=right.selectbox("Submarket/compound",sub_opts)
        ptype=left.selectbox("Property type",sorted(market.property_type.unique()))
        area=right.number_input("Area sqm",min_value=21.0,max_value=2000.0,value=float(market[market.property_type.eq(ptype)].area_sqm.median()),step=5.0)
        beds=left.number_input("Bedrooms",min_value=0.0,max_value=20.0,value=3.0,step=1.0)
        baths=right.number_input("Bathrooms",min_value=0.0,max_value=20.0,value=2.0,step=1.0)
        furnished=left.selectbox("Furnished",sorted(market.furnished.unique()))
        payment=right.selectbox("Payment method",sorted(market.payment_method.unique()))
        completion=left.selectbox("Completion status",sorted(market.completion_status.unique()))
        listing_level=right.selectbox("Listing level",sorted(market.listing_level.unique()))
        
    if st.button("Estimate fair price",type="primary"):
        med=town_df["listing_age_days"].median()
        inputs={"market_region":region,"town":town,"district":district,"submarket_or_compound":submarket,"property_type":ptype,"area_sqm":area,"bedrooms_num":beds,"bathrooms_num":baths,"furnished":furnished,"payment_method":payment,"completion_status":completion,"listing_level":listing_level,"new_city_indicator":town_df.new_city_indicator.mode().iloc[0] if not town_df.new_city_indicator.mode().empty else "No","compound_status":"Broad/standalone" if submarket=="Unknown" or "Compounds" in submarket else "Named submarket/compound","listing_age_days":float(med),"lat":float(town_df.lat.median()),"lon":float(town_df.lon.median()),"images_count":float(town_df.images_count.median()),"amenities_count":float(town_df.amenities_count.median()),"is_premium":0,"is_featured":0,"has_view_360":0}
        row=build_prediction_row(inputs,meta,market)
        pred=predict_price(row,meta)
        tier_name,tier_score=estimate_tier(pred["predicted"]/area,town,ptype,market)
        conf,flags=confidence_label(row,meta,market)
        
        # Premium Glowing Card for Estimation Results
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(99, 102, 241, 0.12) 0%, rgba(20, 184, 166, 0.12) 100%); border: 1px solid rgba(99, 102, 241, 0.25); border-radius: 16px; padding: 28px; margin-bottom: 24px; text-align: center; box-shadow: 0 10px 30px -10px rgba(99, 102, 241, 0.3);">
            <div style="font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.15em; color: #a5b4fc; margin-bottom: 8px;">Estimated Fair Price</div>
            <div style="font-size: 2.8rem; font-weight: 800; color: #ffffff; font-family: 'Outfit', sans-serif; text-shadow: 0 0 15px rgba(99, 102, 241, 0.4); margin-bottom: 8px;">{egp(pred["predicted"])}</div>
            <div style="font-size: 0.95rem; color: #e2e8f0; margin-bottom: 24px; font-family: 'Inter', sans-serif;">Reliability: <span style="color: {'#10b981' if conf == 'High' else '#f59e0b' if conf == 'Medium' else '#ef4444'}; font-weight: 700;">{conf}</span></div>
            <div style="display: flex; justify-content: space-around; flex-wrap: wrap; margin-top: 16px; border-top: 1px solid rgba(255, 255, 255, 0.08); padding-top: 20px;">
                <div style="padding: 8px 16px;">
                    <div style="font-size: 0.75rem; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">Empirical Range (90% CI)</div>
                    <div style="font-size: 1.15rem; font-weight: 600; color: #f3f4f6;">{egp(pred['lower'])} – {egp(pred['upper'])}</div>
                </div>
                <div style="padding: 8px 16px;">
                    <div style="font-size: 0.75rem; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">Estimated EGP/SQM</div>
                    <div style="font-size: 1.15rem; font-weight: 600; color: #f3f4f6;">{pred['predicted']/area:,.0f} EGP</div>
                </div>
                <div style="padding: 8px 16px;">
                    <div style="font-size: 0.75rem; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">Suggested Tier</div>
                    <div style="font-size: 1.15rem; font-weight: 600; color: #f3f4f6;">{tier_name}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if flags: st.warning("; ".join(flags))
        exp=local_explanation(row,meta).head(8)
        a,b=st.columns(2)
        with a:
            st.subheader("Feature contribution to estimate")
            st.dataframe(exp[["feature","value","direction","impact_log_price"]],hide_index=True,width="stretch")
        with b:
            st.subheader("Comparable market listings")
            comps=comparable_listings(row,market,8)
            st.dataframe(comps.drop(columns=["detail_url"]),hide_index=True,width="stretch")

with tabs[5]:
    st.subheader("How accurate and reliable is the model?")
    comparison=load_table("model_comparison.csv")
    st.dataframe(comparison,width="stretch",hide_index=True)
    insight("Random Forest had the lowest validation MAE. CatBoost was retained as the deployable model because it produced lower median and percentage error and handles high-cardinality categories natively. This trade-off is documented rather than hidden.")
    test=pd.read_parquet(ROOT/"reports"/"tables"/"test_predictions.parquet")
    
    a,b=st.columns(2)
    a.plotly_chart(actual_predicted(test),width="stretch")
    
    fig_err = px.histogram(
        test,
        x="absolute_percentage_error",
        nbins=50,
        range_x=[0,float(test.absolute_percentage_error.quantile(.95))],
        title="Test Absolute Percentage Error (APE) Distribution",
        labels={"absolute_percentage_error": "Absolute Percentage Error"},
        color_discrete_sequence=["#14b8a6"]
    )
    fig_err.update_traces(marker_line_color="#090d16", marker_line_width=0.5, opacity=0.85)
    apply_premium_style(fig_err)
    b.plotly_chart(fig_err,width="stretch")
    
    fi=load_table("shap_global_importance.csv").head(15).sort_values("mean_abs_shap_log_price")
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
    st.plotly_chart(fig_fi,width="stretch")
    
    st.caption("SHAP describes predictive association, not causation.")
    c=st.columns(4)
    c[0].metric("Test MAE",egp(meta['test_metrics']['MAE_EGP']))
    c[1].metric("Test RMSE",egp(meta['test_metrics']['RMSE_EGP']))
    c[2].metric("R²",f"{meta['test_metrics']['R2']:.3f}")
    c[3].metric("Median AE",egp(meta['test_metrics']['MedianAE_EGP']))

with tabs[6]:
    st.subheader("Which listings differ materially from the model-estimated range?")
    opp=load_opportunities()
    with st.container(border=True):
        c1,c2,c3=st.columns(3)
        status=c1.multiselect("Status",sorted(opp.pricing_status.unique()),default=sorted(opp.pricing_status.unique()), key="opportunity_status")
        confidence=c2.multiselect("Confidence",["High","Medium","Low"],default=["High","Medium"], key="opportunity_confidence")
        ptypes=c3.multiselect("Property type",sorted(opp.property_type.unique()), key="opportunity_property_type")
        
    d=opp[opp.pricing_status.isin(status)&opp.model_confidence.isin(confidence)]
    if ptypes: d=d[d.property_type.isin(ptypes)]
    d=d.sort_values("opportunity_magnitude",ascending=False)
    st.dataframe(d.head(500),width="stretch",hide_index=True)
    st.download_button("Download filtered opportunities",d.to_csv(index=False).encode("utf-8"),"pricing_opportunities.csv","text/csv")
    st.caption("Large model differences can reflect missing property details, unusual listings, stale advertisements, or genuine pricing opportunities. Manual verification is required.")

with tabs[7]:
    st.subheader("What was changed, and what should users be cautious about?")
    st.markdown(f"""
**Source:** {prep['source_rows_after_d007']:,} records after the approved exact-text duplicate step.  
**Final analytical scope:** {prep['final_analytical_rows']:,} sale listings in six residential property types.  
**Area recovery:** {prep['recovered_area_rows']:,} suspicious areas were recovered only when a plausible explicit area appeared in listing text.  
**Excluded records:** {prep['excluded_rows_total']:,}, retained in an audit parquet with reasons.  
**Tier method:** {prep['price_tier_method']}.  
**Target:** log total asking price. No price-derived predictor is used.  
**Split:** {meta['split_strategy']}.  
**Limitations:** Asking prices are not completed transactions; developer identity is absent; geographic fields mix administrative and marketing labels; market conditions may change after March 2026.
""")
    st.dataframe(pd.read_csv(ROOT/"reports"/"decision_log.csv"),width="stretch",hide_index=True)

