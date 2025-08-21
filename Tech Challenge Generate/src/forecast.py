"""
Advanced forecasting using SARIMA; tried Linear Regression - not as successful

Top approach:
SARIMA (best): Traditional time series model for seasonal data
"""

import json
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
import seaborn as sns
import matplotlib.pyplot as plt


def select_route(df, australian_port, country, foreign_port):
    """Get a monthly series for the specified route."""
    sub = df[(df["australian_port"] == australian_port) & (df["country"] == country) & (df["foreign_port"] == foreign_port)]
    ts = sub.groupby("date")["passengers_total"].sum().sort_index()
    return ts


def train_test_split_time_series(ts, train_end, test_end):
    """Split by date strings like '1988-12-01' and '1989-07-01'."""
    train = ts[:train_end]
    test = ts[pd.to_datetime(train_end) + pd.offsets.MonthBegin(1) : test_end]
    return train, test

def fit_sarima(train_series, seasonal_period=12):
    """Fit SARIMA model with seasonal components."""
    # SARIMA(1,1,1)(1,1,1,12) - handles trend and seasonality
    model = SARIMAX(
            train_series,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, seasonal_period),
            enforce_stationarity=False,
            enforce_invertibility=False)
    fitted = model.fit(disp=False)
    return fitted


def forecast_sarima(fitted_model, steps):
    """Forecast using fitted SARIMA model."""
    forecast = fitted_model.forecast(steps=steps)
    return forecast

# ===== EVALUATION =====
def evaluate_forecast(y_true, y_pred):
    """Compute MAE, RMSE, and MAPE."""
    y_pred = y_pred.reindex(y_true.index)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100.0)
    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
    }


# ===== MAIN BACKTEST FUNCTIONS =====
def run_sarima_backtest(df, outputs_dir, australian_port, foreign_port, country, train_end, test_end, label):
    """Run SARIMA backtest."""
    ts = select_route(df, australian_port=australian_port, country=country, foreign_port=foreign_port)
    if ts.empty:
        raise ValueError(f"No data for route {australian_port} → {foreign_port} ({country})")

    train, test = train_test_split_time_series(ts, train_end=train_end, test_end=test_end)
    steps = len(test)

    # Fit SARIMA
    fitted_model = fit_sarima(train)
    if fitted_model is None:
        raise ValueError("SARIMA model fitting failed")

    # Forecast
    preds = forecast_sarima(fitted_model, steps)
    if preds is None:
        raise ValueError("SARIMA forecasting failed")

    preds.index = test.index
    result = evaluate_forecast(test, preds)

    # Save results
    os.makedirs(os.path.join(outputs_dir, "models"), exist_ok=True)
    metrics_path = os.path.join(outputs_dir, "models", f"{label}_sarima_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"mae": result["mae"], "rmse": result["rmse"], "mape": result["mape"]}, f, indent=2)

    backtest_df = pd.DataFrame({"y_true": result["y_true"], "y_pred": result["y_pred"]})
    backtest_df.to_csv(os.path.join(outputs_dir, "models", f"{label}_sarima_backtest.csv"))

    return result

def forecast_future_sarima(df, outputs_dir, australian_port, foreign_port, country,
                           train_end="1989-06-01", forecast_months=6, label="future_forecast"):
    """
    Forecast future months using SARIMA model trained on all available data.

    Parameters:
    - df: Cleaned DataFrame
    - australian_port, foreign_port, country: Route details
    - train_end: Train on data up to this date
    - forecast_months: How many months to predict ahead
    - label: Output file prefix
    """
    # Get the route data
    ts = select_route(df, australian_port=australian_port, country=country, foreign_port=foreign_port)
    if ts.empty:
        raise ValueError(f"No data for route {australian_port} → {foreign_port} ({country})")

    # Train on ALL available data (up to July 1989)
    train = ts[:train_end]

    # Fit SARIMA model
    fitted_model = fit_sarima(train)
    if fitted_model is None:
        raise ValueError("SARIMA model fitting failed")

    # Forecast next 6 months
    future_dates = pd.date_range(start=pd.to_datetime(train_end) + pd.offsets.MonthBegin(1),
                                 periods=forecast_months, freq="MS")
    future_forecast = fitted_model.forecast(steps=forecast_months)
    future_forecast.index = future_dates

    # Save forecast
    os.makedirs(os.path.join(outputs_dir, "models"), exist_ok=True)
    forecast_path = os.path.join(outputs_dir, "models", f"{label}.csv")

    forecast_df = pd.DataFrame({
        "date": future_forecast.index,
        "predicted_passengers": future_forecast.values
    })
    forecast_df.to_csv(forecast_path, index=False)

    return forecast_df

def main():
    df = pd.read_csv("Outputs/clean/cleaned_data.csv", parse_dates=["date"])

    # Test both approaches
    results = run_sarima_backtest(
        df=df,
        outputs_dir="Outputs",
        australian_port="Sydney",
        foreign_port="Auckland",
        country="New Zealand",
        train_end="1988-12-01",
        test_end="1989-06-01",
        label="syd_akl"
    )

    print(f"SARIMA MAE: {results['mae']:.2f}, MAPE: {results['mape']:.2f}%")

    # Forecast next n months for Melbourne → Singapore
    future_df = forecast_future_sarima(
        df=df,
        outputs_dir="Outputs",
        australian_port="Melbourne",
        foreign_port="Singapore",
        country="Singapore",
        train_end="1989-06-01",  # Train on all available data
        forecast_months=12,
        label="melbourne_singapore_future_twelve"
    )

    df_pass = df[df["route"] == "Melbourne → Singapore"][["date", "passengers_total"]]
    df_merged = (df_pass
                 .merge(future_df[["date", "predicted_passengers"]],
                        on="date", how="outer")
                 .sort_values("date")
                 .reset_index(drop=True))

    df_long = df_merged.melt(
        id_vars="date",
        value_vars=["passengers_total", "predicted_passengers"],
        var_name="series",
        value_name="passengers"
    )

    plt.figure(figsize=(11, 5))
    sns.lineplot(data=df_long, x="date", y="passengers", hue="series", marker="o")
    plt.title("Melbourne → Singapore: Actual vs Predicted Passengers")
    plt.xlabel("Date")
    plt.ylabel("Passengers")
    plt.tight_layout()
    plt.savefig("Outputs/figures/mel_sin_actual_pred.png")
    plt.show()

    print(df_merged)

main()