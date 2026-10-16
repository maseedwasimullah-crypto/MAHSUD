"""Chart generation for MAHSUD (Matplotlib files + Plotly figures).

Charts describe recorded historical aggregates only. Titles and notes make
clear that variation may reflect reporting coverage and that no causal or
targeting inference should be drawn.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from config import FIGURES_DIR

SOURCE_NOTE = "Source: loaded MAHSUD dataset (recorded incidents only; coverage may vary)."
DEMO_NOTE = "NOTE: dataset in use is SYNTHETIC demonstration data unless replaced with verified sources."


def _style(ax, title: str, xlabel: str, ylabel: str, note: str = SOURCE_NOTE):
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=8)
    if note:
        ax.annotate(
            note,
            xy=(0.0, -0.18),
            xycoords="axes fraction",
            fontsize=7,
            color="#555555",
        )


def plot_yearly_incidents(yearly: pd.DataFrame, out_path: Path | None = None, demo: bool = False) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(yearly["year"], yearly["incidents"], color="#2c6e9b", label="Recorded incidents")
    ax.plot(yearly["year"], yearly["fatalities"], color="#c0392b", marker="o", ms=3, label="Recorded fatalities")
    note = SOURCE_NOTE + (" " + DEMO_NOTE if demo else "")
    _style(ax, "Yearly Recorded Terrorism Incidents — KPK Dataset", "Year", "Count", note)
    fig.tight_layout()
    out_path = out_path or FIGURES_DIR / "yearly_incidents.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_monthly_incidents(monthly: pd.DataFrame, out_path: Path | None = None, demo: bool = False) -> Path:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = pd.to_datetime(monthly["month_start"] if "month_start" in monthly else monthly["period"])
    ax.plot(x, monthly["incidents"], color="#2c6e9b", lw=1.2, label="Monthly recorded incidents")
    if len(monthly) >= 12:
        roll = monthly["incidents"].rolling(12).mean()
        ax.plot(x, roll, color="#e67e22", lw=1.6, label="12-month rolling mean")
    note = SOURCE_NOTE + (" " + DEMO_NOTE if demo else "")
    _style(ax, "Monthly Recorded Incident Counts — KPK Dataset", "Month", "Count", note)
    fig.autofmt_xdate()
    fig.tight_layout()
    out_path = out_path or FIGURES_DIR / "monthly_incidents.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_casualties(yearly: pd.DataFrame, out_path: Path | None = None, demo: bool = False) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(yearly["year"] - 0.2, yearly["fatalities"], width=0.4, color="#c0392b", label="Fatalities (reported)")
    ax.bar(yearly["year"] + 0.2, yearly["injuries"], width=0.4, color="#d35400", label="Injuries (reported)")
    note = SOURCE_NOTE + " Missing casualty fields excluded (not treated as zero)." + (
        " " + DEMO_NOTE if demo else ""
    )
    _style(ax, "Yearly Recorded Casualties — KPK Dataset", "Year", "Count", note)
    fig.tight_layout()
    out_path = out_path or FIGURES_DIR / "yearly_casualties.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_district_comparison(districts: pd.DataFrame, top_n: int = 15, out_path: Path | None = None, demo: bool = False) -> Path:
    top = districts.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["district"], top["incidents"], color="#2c6e9b")
    note = (
        "Recorded incidents in this dataset only. Differences may reflect "
        "reporting coverage; they do not imply any inherent trait of a district "
        "or its residents." + (" " + DEMO_NOTE if demo else "")
    )
    _style(ax, f"Top {len(top)} Districts by Recorded Incidents", "Recorded incidents", "District", note)
    fig.tight_layout()
    out_path = out_path or FIGURES_DIR / "district_comparison.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_category_distribution(categories: pd.DataFrame, out_path: Path | None = None, demo: bool = False) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(categories["incident_category"], categories["incidents"], color="#5d6d7e")
    plt.xticks(rotation=30, ha="right")
    note = SOURCE_NOTE + (" " + DEMO_NOTE if demo else "")
    _style(ax, "Recorded Incident Categories", "Category", "Count", note)
    fig.tight_layout()
    out_path = out_path or FIGURES_DIR / "category_distribution.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_forecast(
    series: pd.DataFrame,
    forecast: pd.DataFrame,
    test_pred: pd.DataFrame | None = None,
    out_path: Path | None = None,
    demo: bool = False,
) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    hist_x = pd.to_datetime(series["month_start"])
    hist_y = series["incidents"] if "incidents" in series.columns else series["incident_count"]
    ax.plot(hist_x, hist_y, color="#2c6e9b", lw=1.1, label="Recorded history")
    if test_pred is not None and len(test_pred):
        ax.plot(
            pd.to_datetime(test_pred["month_start"]),
            test_pred["predicted"],
            color="#8e44ad",
            lw=1.4,
            ls="--",
            label="Model test-set fit",
        )
    fx = pd.to_datetime(forecast["month_start"])
    ax.plot(fx, forecast["forecast_count"], color="#c0392b", lw=1.6, label="Uncertain forecast (central)")
    ax.fill_between(
        fx,
        forecast["lower_95"],
        forecast["upper_95"],
        color="#c0392b",
        alpha=0.18,
        label="Rough 95% residual interval",
    )
    note = (
        "Forecasts are uncertain AGGREGATE estimates of recorded incident counts "
        "assuming past patterns continue. They are NOT predictions of specific "
        "attacks, dates, or locations." + (" " + DEMO_NOTE if demo else "")
    )
    _style(ax, "Monthly Incident Counts with Uncertain Aggregate Forecast", "Month", "Count", note)
    fig.autofmt_xdate()
    fig.tight_layout()
    out_path = out_path or FIGURES_DIR / "forecast.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plotly_yearly(yearly: pd.DataFrame, demo: bool = False) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(x=yearly["year"], y=yearly["incidents"], name="Recorded incidents", marker_color="#2c6e9b")
    fig.add_scatter(x=yearly["year"], y=yearly["fatalities"], name="Recorded fatalities", line=dict(color="#c0392b"))
    fig.update_layout(
        title="Yearly Recorded Incidents and Fatalities" + (" (SYNTHETIC demo data)" if demo else ""),
        xaxis_title="Year",
        yaxis_title="Count",
        legend_title="Series",
        hovermode="x unified",
    )
    return fig


def plotly_monthly(monthly: pd.DataFrame, demo: bool = False) -> go.Figure:
    x = pd.to_datetime(monthly["month_start"] if "month_start" in monthly else monthly["period"])
    fig = go.Figure()
    fig.add_scatter(x=x, y=monthly["incidents"], mode="lines", name="Recorded incidents", line=dict(color="#2c6e9b"))
    if len(monthly) >= 12:
        fig.add_scatter(
            x=x,
            y=monthly["incidents"].rolling(12).mean(),
            mode="lines",
            name="12-month rolling mean",
            line=dict(color="#e67e22", width=2),
        )
    fig.update_layout(
        title="Monthly Recorded Incident Counts" + (" (SYNTHETIC demo data)" if demo else ""),
        xaxis_title="Month",
        yaxis_title="Count",
        hovermode="x unified",
    )
    return fig


def plotly_districts(districts: pd.DataFrame, top_n: int = 15, demo: bool = False) -> go.Figure:
    top = districts.head(top_n)
    fig = px.bar(
        top,
        x="incidents",
        y="district",
        orientation="h",
        color="fatalities",
        color_continuous_scale="Reds",
        title=f"Top {len(top)} Districts by Recorded Incidents" + (" (SYNTHETIC demo data)" if demo else ""),
        labels={
            "incidents": "Recorded incidents",
            "district": "District",
            "fatalities": "Recorded fatalities (sum)",
        },
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    fig.add_annotation(
        text=(
            "Recorded dataset counts only; coverage may vary. "
            "This does not imply any district or community is inherently dangerous."
        ),
        xref="paper",
        yref="paper",
        x=0,
        y=-0.18,
        showarrow=False,
        font=dict(size=10, color="#666666"),
    )
    return fig


def plotly_forecast(series: pd.DataFrame, forecast: pd.DataFrame, demo: bool = False) -> go.Figure:
    fig = go.Figure()
    hx = pd.to_datetime(series["month_start"])
    hy = series["incidents"] if "incidents" in series.columns else series["incident_count"]
    fx = pd.to_datetime(forecast["month_start"])
    fig.add_scatter(x=hx, y=hy, mode="lines", name="Recorded history", line=dict(color="#2c6e9b"))
    fig.add_scatter(
        x=fx, y=forecast["forecast_count"], mode="lines+markers", name="Uncertain forecast", line=dict(color="#c0392b")
    )
    fig.add_scatter(
        x=fx, y=forecast["upper_95"], mode="lines", name="Rough upper bound", line=dict(dash="dot", color="#c0392b")
    )
    fig.add_scatter(
        x=fx, y=forecast["lower_95"], mode="lines", name="Rough lower bound", line=dict(dash="dot", color="#c0392b"),
        fill="tonexty", fillcolor="rgba(192,57,43,0.15)",
    )
    fig.update_layout(
        title="Aggregate Forecast of Recorded Monthly Counts (uncertain estimate)"
        + (" (SYNTHETIC demo data)" if demo else ""),
        xaxis_title="Month",
        yaxis_title="Recorded incident count",
        hovermode="x unified",
    )
    fig.add_annotation(
        text=(
            "NOT a prediction of specific attacks. Assumes past recorded patterns continue; "
            "intervals are rough residual-based bounds."
        ),
        xref="paper",
        yref="paper",
        x=0,
        y=-0.16,
        showarrow=False,
        font=dict(size=10, color="#666666"),
    )
    return fig


def generate_all_figures(
    yearly: pd.DataFrame,
    monthly: pd.DataFrame,
    districts: pd.DataFrame,
    categories: pd.DataFrame,
    series: pd.DataFrame | None = None,
    forecast: pd.DataFrame | None = None,
    test_pred: pd.DataFrame | None = None,
    demo: bool = False,
) -> list[Path]:
    paths = [
        plot_yearly_incidents(yearly, demo=demo),
        plot_monthly_incidents(monthly, demo=demo),
        plot_casualties(yearly, demo=demo),
        plot_district_comparison(districts, demo=demo),
        plot_category_distribution(categories, demo=demo),
    ]
    if series is not None and forecast is not None:
        paths.append(plot_forecast(series, forecast, test_pred, demo=demo))
    return paths
