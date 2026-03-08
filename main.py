import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from src.models import arima_model, theta_model, symbolic_genetic_model

metadata = {
    "hourly": dict(path="./data/train_hourly.csv", freq=24, horizon=48),
    "daily": dict(path="./data/train_daily.csv", freq=1, horizon=14),
    "weekly": dict(path="./data/train_weekly.csv", freq=1, horizon=13),
    "monthly": dict(path="./data/train_monthly.csv", freq=12, horizon=18),
    "quarterly": dict(path="./data/train_quarterly.csv", freq=4, horizon=8),
    "yearly": dict(path="./data/train_yearly.csv", freq=1, horizon=6),
}


def smape(actual: pd.Series, forecast: pd.Series):
    # Convert to numpy to avoid Pandas index alignment issues
    a = actual.values
    f = forecast.values

    # Handle infinities and NaNs
    f = np.nan_to_num(f, nan=np.nanmean(a), posinf=np.nanmax(a) * 2, neginf=0)

    # Calculate sMAPE components
    numerator = np.abs(f - a)
    denominator = (np.abs(a) + np.abs(f)) / 2

    # Avoid division by zero and compute mean
    with np.errstate(divide="ignore", invalid="ignore"):
        elements = np.divide(numerator, denominator)
        elements = np.nan_to_num(elements, nan=0.0)

    # Cap at 200% (2.0)
    return 100 * np.mean(np.clip(elements, 0, 2.0))


def process_category(category_name, metadata, create_plots=False):
    """
    Load, clean and save decomposition plots for a single category.
    """
    info = metadata
    file_path = info["path"]
    freq = info["freq"]

    # we store all the time series
    category_ts_dict = {}

    with open(file_path, "r") as f:
        # skip the header row
        next(f)
        for line in f:
            tokens = line.strip().split(",")
            serie_id = tokens[0].replace('"', "")

            values = pd.Series([float(x) for x in tokens[1:] if x != "NA"]).dropna()

            # plot if the flag is on
            if create_plots:
                decomposition = seasonal_decompose(
                    values, model="multiplicative", period=freq
                )
                fig = decomposition.plot()
                fig.set_size_inches(10, 10)

                # access the first axis of the decomposition plot to customize the title
                fig.axes[0].set_title(f"ID: {serie_id} - Multiplicative Decomposition")

                # Save and Clean up
                plt.tight_layout()
                plt.savefig(f"plots/{category_name}_{serie_id}.png", dpi=300)
                plt.close(fig)

            category_ts_dict[serie_id] = values

    return category_ts_dict


def split_train_dev(category_ts_dict, horizon):
    """
    Cuts the dictionary of series of a category into train and dev according to their horizon.
    """
    train_dict = {}
    dev_dict = {}

    for serie_id, values in category_ts_dict.items():
        train_dict[serie_id] = values.iloc[:-horizon]
        dev_dict[serie_id] = values.iloc[-horizon:]

    return train_dict, dev_dict


def main():
    results = {}
    model = symbolic_genetic_model

    for key, info in metadata.items():
        print(f"Processing Category: {key}")

        horizon = info["horizon"]
        freq = info["freq"]

        # Load and split series
        series = process_category(key, info)

        train, dev = split_train_dev(series, horizon)

        # List to store ONLY forecast rows
        forecast_rows = []

        for serie_id in series.keys():
            prediction = model(train[serie_id], horizon, freq, serie_id)
            forecast_rows.append([serie_id] + list(prediction))
            score = smape(dev[serie_id], prediction)
            results[serie_id] = score

        # Create DataFrame with exactly horizon columns (e.g., 48 for hourly)
        df_forecasts = pd.DataFrame(forecast_rows)
        df_forecasts.columns = ["ID"] + [f"F{i}" for i in range(1, horizon + 1)]

        # Save results to CSV
        output_file = f"results/{key}_1.csv"
        df_forecasts.to_csv(output_file, index=False)
        print(f"Saved: {output_file} with {len(df_forecasts.columns)} columns")

    # Save summary of sMAPE scores
    df_results = pd.DataFrame(list(results.items()), columns=["ID", "sMAPE"])
    df_results.to_csv(f"{model.__name__}_scores.csv", index=False)
    print("FINAL MEAN sMAPE: ", df_results["sMAPE"].mean())


if __name__ == "__main__":
    main()
