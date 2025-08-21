"""
Panel app that renders Plotly (plotly.go) figures.

This combines Panel widgets with Plotly charts for:
- Time series over time
- Top/Bottom N routes
"""

import pandas as pd
import plotly.graph_objects as go
import panel as pn

pn.extension('plotly', sizing_mode="stretch_width")


def load_cleaned(cleaned_path="Outputs/clean/cleaned_data.csv"):
    """Read cleaned data with parsed monthly date."""
    return pd.read_csv(cleaned_path, parse_dates=["date"])  # date parsed as datetime


def build_app(cleaned_path="Outputs/clean/cleaned_data.csv"):
    """Build a Panel dashboard with simple filters and Plotly charts using pn.bind for live updates."""
    df = load_cleaned(cleaned_path)

    # Widgets
    y_metric = pn.widgets.Select(name="Y-axis", value="passengers_total", options=[
        "passengers_total", "passengers_total_z", "freight_total_tonnes", "mail_total_tonnes", "load_balance_ratio"
    ])
    year_range = pn.widgets.IntRangeSlider(name="Year", start=int(df["year"].min()), end=int(df["year"].max()), value=(int(df["year"].min()), int(df["year"].max())))
    month_range = pn.widgets.IntRangeSlider(name="Month # (1-12)", start=1, end=12, value=(1, 12))
    port = pn.widgets.MultiSelect(name="Australian Port", options=sorted(df["australian_port"].dropna().unique()))
    fport = pn.widgets.MultiSelect(name="Foreign Port", options=sorted(df["foreign_port"].dropna().unique()))
    continent = pn.widgets.MultiSelect(name="Continent", options=sorted(df["continent"].dropna().unique()))
    country = pn.widgets.MultiSelect(name="Country", options=sorted(df["country"].dropna().unique()))
    top_n = pn.widgets.IntInput(name="Top N", value=10, start=1, end=100)
    bottom_n = pn.widgets.IntInput(name="Bottom N", value=10, start=1, end=100)

    def filter_df(y_metric_val, year_rng, month_rng, port_vals, fport_vals, continent_vals, country_vals):
        sub = df.copy()
        sub = sub[(sub["month_num"] >= month_rng[0]) & (sub["month_num"] <= month_rng[1])]
        sub = sub[(sub["year"] >= year_rng[0]) & (sub["year"] <= year_rng[1])]
        if port_vals:
            sub = sub[sub["australian_port"].isin(port_vals)]
        if fport_vals:
            sub = sub[sub["foreign_port"].isin(fport_vals)]
        if continent_vals:
            sub = sub[sub["continent"].isin(continent_vals)]
        if country_vals:
            sub = sub[sub["country"].isin(country_vals)]
        return sub

    def make_time_series(y_metric_val, year_rng, month_rng, port_vals, fport_vals, continent_vals, country_vals):
        sub = filter_df(y_metric_val, year_rng, month_rng, port_vals, fport_vals, continent_vals, country_vals)
        ts = sub.groupby("date")[y_metric_val].sum().sort_index()
        fig = go.Figure(data=[go.Scatter(x=ts.index, y=ts.values, mode="lines")])
        fig.update_layout(title=f"{y_metric_val} over time", xaxis_title="Date", yaxis_title=y_metric_val)
        return pn.pane.Plotly(fig, config={"responsive": True})

    def make_top_bottom(y_metric_val, year_rng, month_rng, port_vals, fport_vals, continent_vals, country_vals, topn, bottomn):
        sub = filter_df(y_metric_val, year_rng, month_rng, port_vals, fport_vals, continent_vals, country_vals)
        agg = sub.groupby("route")[y_metric_val].sum().sort_values(ascending=False)
        top = agg.head(int(topn) if topn else 10)
        bottom = agg.tail(int(bottomn) if bottomn else 10)
        top_fig = go.Figure(data=[go.Bar(x=top.values, y=top.index, orientation='h')])
        top_fig.update_layout(title=f"Top {len(top)} routes by {y_metric_val}", xaxis_title=y_metric_val, yaxis_title="route")
        bottom_fig = go.Figure(data=[go.Bar(x=bottom.values, y=bottom.index, orientation='h')])
        bottom_fig.update_layout(title=f"Bottom {len(bottom)} routes by {y_metric_val}", xaxis_title=y_metric_val, yaxis_title="route")
        return pn.Row(pn.pane.Plotly(top_fig, config={"responsive": True}), pn.pane.Plotly(bottom_fig, config={"responsive": True}))

    # Bind reactive views
    ts_view = pn.bind(make_time_series, y_metric, year_range, month_range, port, fport, continent, country)
    tb_view = pn.bind(make_top_bottom, y_metric, year_range, month_range, port, fport, continent, country, top_n, bottom_n)

    # Layout
    sidebar = pn.Column(
        "Filters",
        y_metric, year_range, month_range, port, fport, continent, country, top_n, bottom_n,
    )
    template = pn.template.BootstrapTemplate(title="AeroConnect Route Dashboard")
    template.sidebar.append(sidebar)
    template.main.append(ts_view)
    template.main.append(tb_view)
    return template


if __name__ == "__main__":
    pn.serve(build_app, show=True)