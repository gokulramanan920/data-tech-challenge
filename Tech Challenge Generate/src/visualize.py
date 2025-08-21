"""
Static figures using matplotlib/seaborn with simple, readable code.
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def set_style():
    sns.set_theme(style="whitegrid", context="talk")


def save_fig(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_top_bottom_routes(df, outputs_dir, top_n=10):
    set_style()
    agg = df.groupby("route")["passengers_total"].sum().sort_values(ascending=False)
    top = agg.head(top_n)
    bottom = agg.tail(top_n)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=top.values, y=top.index, ax=ax, palette="viridis")
    ax.set_title(f"Top {top_n} Routes by Total Passengers")
    ax.set_xlabel("Passengers (sum)")
    ax.set_ylabel("Route")
    save_fig(fig, os.path.join(outputs_dir, "figures", "top_routes.png"))

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=bottom.values, y=bottom.index, ax=ax, palette="magma")
    ax.set_title(f"Bottom {top_n} Routes by Total Passengers")
    ax.set_xlabel("Passengers (sum)")
    ax.set_ylabel("Route")
    save_fig(fig, os.path.join(outputs_dir, "figures", "bottom_routes.png"))


def plot_seasonality(df, outputs_dir, route_filter=None):
    set_style()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    if route_filter is not None:
        df = df[df["route"] == route_filter]

    monthly = df.groupby(df["date"].dt.month)["passengers_total"].mean()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(x=monthly.index, y=monthly.values, marker="o", ax=ax)
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("Month")
    ax.set_ylabel("Average Passengers")
    title_suffix = f" — {route_filter}" if route_filter else ""
    ax.set_title("Average Monthly Seasonality (All Routes" + title_suffix + ")")
    filename = "seasonality.png" if route_filter is None else f"seasonality_{route_filter.replace(' ', '_')}.png"
    save_fig(fig, os.path.join(outputs_dir, "figures", filename))


def plot_continent_share(df, outputs_dir):
    set_style()
    cont = df.groupby("continent")["passengers_total"].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(cont.values, labels=cont.index, autopct="%1.1f%%", startangle=90, counterclock=False)
    ax.set_title("Passenger Share by Continent")
    save_fig(fig, os.path.join(outputs_dir, "figures", "continent_share.png"))


def plot_time_series(df, outputs_dir, route_filter=None):
    set_style()
    if route_filter is not None:
        df = df[df["route"] == route_filter]
    ts = df.groupby("date")["passengers_total"].sum().sort_index()
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.lineplot(x=ts.index, y=ts.values, ax=ax)
    ax.set_title("Total Passengers Over Time" + (f" — {route_filter}" if route_filter else ""))
    ax.set_xlabel("Date")
    ax.set_ylabel("Passengers (sum)")
    filename = "time_series.png" if route_filter is None else f"time_series_{route_filter.replace(' ', '_')}.png"
    save_fig(fig, os.path.join(outputs_dir, "figures", filename))

def main():
    df = pd.read_csv("Outputs/clean/cleaned_data.csv")
    plot_top_bottom_routes(df, "Outputs", top_n=10)
    plot_seasonality(df, "Outputs", "Perth → Singapore")
    plot_continent_share(df, "Outputs")
    plot_time_series(df, "Outputs")

main()