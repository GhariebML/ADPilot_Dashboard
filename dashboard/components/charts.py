from __future__ import annotations
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def apply_premium_style(fig):
    fig.update_layout(
        font_family="Outfit, Inter, sans-serif",
        font_color="#f3f4f6",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=50, r=20, t=60, b=50),
        title=dict(
            font=dict(size=16, color="#f3f4f6", family="Outfit, sans-serif"),
            pad=dict(b=10)
        ),
        legend=dict(
            bgcolor="rgba(17, 24, 39, 0.7)",
            bordercolor="rgba(255, 255, 255, 0.1)",
            borderwidth=1,
            font=dict(size=11, color="#f3f4f6"),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    fig.update_xaxes(
        gridcolor="rgba(255, 255, 255, 0.06)",
        linecolor="rgba(255, 255, 255, 0.1)",
        zerolinecolor="rgba(255, 255, 255, 0.1)",
        title_font=dict(size=12, color="#9ca3af", family="Inter, sans-serif"),
        tickfont=dict(size=10, color="#9ca3af")
    )
    fig.update_yaxes(
        gridcolor="rgba(255, 255, 255, 0.06)",
        linecolor="rgba(255, 255, 255, 0.1)",
        zerolinecolor="rgba(255, 255, 255, 0.1)",
        title_font=dict(size=12, color="#9ca3af", family="Inter, sans-serif"),
        tickfont=dict(size=10, color="#9ca3af")
    )
    return fig


def price_distribution(df: pd.DataFrame, curr: str = "EGP"):
    fig = px.histogram(
        df,
        x="price_egp",
        nbins=55,
        log_y=True,
        labels={"price_egp": f"Asking Price ({curr})"},
        title=f"Asking Price Distribution ({curr}, Log Scale)",
        color_discrete_sequence=["#6366f1"]
    )
    fig.update_layout(xaxis_tickformat=",")
    fig.update_traces(marker_line_color="#090d16", marker_line_width=0.5, opacity=0.85)
    return apply_premium_style(fig)


def ppsqm_box(df: pd.DataFrame, curr: str = "EGP"):
    top = df["property_type"].value_counts().index
    sample = df[df["property_type"].isin(top)].copy()
    cap = sample["price_per_sqm"].quantile(.99)
    fig = px.box(
        sample[sample["price_per_sqm"] <= cap],
        x="property_type",
        y="price_per_sqm",
        color="property_type",
        points=False,
        title=f"Price per SQM by Property Type ({curr})",
        labels={"price_per_sqm": f"{curr} per SQM", "property_type": "Property Type"},
        color_discrete_sequence=["#6366f1", "#14b8a6", "#8b5cf6", "#ec4899", "#f59e0b", "#3b82f6"]
    )
    fig.update_layout(showlegend=False)
    fig.update_yaxes(tickformat=",")
    return apply_premium_style(fig)


def location_bar(summary: pd.DataFrame, curr: str = "EGP"):
    d = summary[summary["listings"] >= 30].nlargest(15, "median_price_per_sqm").sort_values("median_price_per_sqm")
    fig = px.bar(
        d,
        x="median_price_per_sqm",
        y="town",
        orientation="h",
        hover_data=["listings", "median_price"],
        title=f"Highest-Priced Towns (Min 30 Listings, {curr})",
        labels={"median_price_per_sqm": f"Median {curr} per SQM", "town": "Town"},
        color="median_price_per_sqm",
        color_continuous_scale=[[0, "#6366f1"], [1, "#14b8a6"]]
    )
    fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(tickformat=",")
    return apply_premium_style(fig)


def actual_predicted(test: pd.DataFrame, curr: str = "EGP"):
    cap = max(test["price_egp"].quantile(.99), test["predicted_fair_price"].quantile(.99))
    s = test[(test["price_egp"] <= cap) & (test["predicted_fair_price"] <= cap)]
    fig = px.scatter(
        s,
        x="price_egp",
        y="predicted_fair_price",
        opacity=0.35,
        title=f"Actual vs Predicted Price ({curr}, Untouched Test Set)",
        labels={"price_egp": f"Actual {curr}", "predicted_fair_price": f"Predicted {curr}"},
        color_discrete_sequence=["#6366f1"]
    )
    fig.update_traces(marker=dict(size=6, line=dict(width=0.5, color="rgba(255,255,255,0.1)")))
    fig.add_trace(go.Scatter(
        x=[0, cap],
        y=[0, cap],
        mode="lines",
        name="Perfect Prediction",
        line=dict(color="#ec4899", width=2, dash="dash")
    ))
    fig.update_xaxes(tickformat=",")
    fig.update_yaxes(tickformat=",")
    return apply_premium_style(fig)

