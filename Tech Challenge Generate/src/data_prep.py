"""
Data preparation helpers. Kept intentionally simple with clear docstrings.

Steps:
- Read raw CSV
- Rename columns to snake_case
- Create a proper monthly date
- Add route keys and load balance features
- Add a simple continent column (fallback mapping)
- Add per-route z-scores for passengers_total
"""

import os
import pandas as pd


def ensure_output_dirs(base_dir):
    """Create output subfolders we will use throughout the project."""
    folders = [
        os.path.join(base_dir, "clean"),
        os.path.join(base_dir, "figures"),
        os.path.join(base_dir, "evaluations"),
        os.path.join(base_dir, "models"),
    ]
    for path in folders:
        os.makedirs(path, exist_ok=True)


def load_raw_csv(csv_path):
    """Load the provided CSV into a pandas DataFrame."""
    return pd.read_csv(csv_path)


def normalize_columns(df):
    """Rename columns to easier names and keep the same information."""
    return df.rename(
        columns={
            "Month": "month_label",
            "AustralianPort": "australian_port",
            "ForeignPort": "foreign_port",
            "Country": "country",
            "Passengers_In": "passengers_in",
            "Freight_In_(tonnes)": "freight_in_tonnes",
            "Mail_In_(tonnes)": "mail_in_tonnes",
            "Passengers_Out": "passengers_out",
            "Freight_Out_(tonnes)": "freight_out_tonnes",
            "Mail_Out_(tonnes)": "mail_out_tonnes",
            "Passengers_Total": "passengers_total",
            "Freight_Total_(tonnes)": "freight_total_tonnes",
            "Mail_Total_(tonnes)": "mail_total_tonnes",
            "Year": "year",
            "Month_num": "month_num",
        }
    )


def parse_dates(df):
    """Build a monthly date using the provided year and month number."""
    df = df.copy()
    df["date"] = pd.to_datetime(dict(year=df["year"], month=df["month_num"], day=1))
    return df


def add_route_keys(df):
    """Create route names like "Sydney → Auckland" and a city_pair helper."""
    df = df.copy()
    df["route"] = df["australian_port"].astype(str) + " → " + df["foreign_port"].astype(str)
    df["city_pair"] = df["australian_port"].astype(str) + " | " + df["country"].astype(str)
    return df


def add_load_balance_features(df):
    """Add simple load balance features: difference and ratio (in vs out)."""
    df = df.copy()
    df["load_balance_diff"] = df["passengers_in"] - df["passengers_out"]
    df["load_balance_ratio"] = df["passengers_in"] / df["passengers_out"].replace({0: pd.NA})
    return df


def add_continent(df):
    """
    Add a continent column using a small fallback map.
    This avoids extra dependencies and is easy to understand.
    """
    fallback_map = {
        "United Kingdom": "Europe",
        "UK": "Europe",
        "England": "Europe",
        "Germany": "Europe",
        "Italy": "Europe",
        "France": "Europe",
        "Netherlands": "Europe",
        "Bahrain": "Asia",
        "Oman": "Asia",
        "United Arab Emirates": "Asia",
        "Qatar": "Asia",
        "Kuwait": "Asia",
        "Saudi Arabia": "Asia",
        "Israel": "Asia",
        "India": "Asia",
        "Pakistan": "Asia",
        "Bangladesh": "Asia",
        "Sri Lanka": "Asia",
        "Nepal": "Asia",
        "Singapore": "Asia",
        "Malaysia": "Asia",
        "Thailand": "Asia",
        "Japan": "Asia",
        "Hong Kong": "Asia",
        "Taiwan": "Asia",
        "South Korea": "Asia",
        "Philippines": "Asia",
        "Indonesia": "Asia",
        "Brunei": "Asia",
        "Papua New Guinea": "Oceania",
        "Solomon Islands": "Oceania",
        "Vanuatu": "Oceania",
        "Fiji": "Oceania",
        "New Caledonia": "Oceania",
        "New Zealand": "Oceania",
        "USA": "North America",
        "United States": "North America",
        "Canada": "North America",
        "Mexico": "North America",
        "China": "Asia",
    }
    df = df.copy()
    df["continent"] = df["country"].map(fallback_map).fillna("Unknown")
    return df


def add_z_scores(df):
    """Compute a simple z-score per route for passengers_total (mean 0, std 1)."""
    df = df.copy()

    def compute_group_z(group):
        values = group["passengers_total"].astype(float)
        mean_value = values.mean()
        std_value = values.std(ddof=0)
        if std_value == 0 or pd.isna(std_value):
            return pd.Series([0.0] * len(group), index=group.index)
        return (values - mean_value) / std_value

    df["passengers_total_z"] = df.groupby("route", group_keys=False).apply(compute_group_z)
    return df


def clean_and_enrich(csv_path, outputs_dir):
    """
    Run the full cleaning/enrichment and save the cleaned CSV.

    Returns the DataFrame and the path to the saved CSV.
    """
    ensure_output_dirs(outputs_dir)
    raw = load_raw_csv(csv_path)
    df = normalize_columns(raw)
    df = parse_dates(df)
    df = add_route_keys(df)
    df = add_load_balance_features(df)
    df = add_continent(df)
    df = add_z_scores(df)

    cleaned_path = os.path.join(outputs_dir, "clean", "cleaned_data.csv")
    df.to_csv(cleaned_path, index=False)
    return df, cleaned_path

def main():
    df_clean, cleaned_path = clean_and_enrich("TechChallenge_Data.csv", "Outputs")
    print(df_clean)

main()