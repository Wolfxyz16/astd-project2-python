import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from src.models import arima_model, theta_model

metadata = {
    "hourly": dict(path="./data/train_hourly.csv", freq=24, horizon=48),
    "daily": dict(path="./data/train_daily.csv", freq=1, horizon=14),
    "weekly": dict(path="./data/train_weekly.csv", freq=1, horizon=13),
    "monthly": dict(path="./data/train_monthly.csv", freq=12, horizon=18),
    "quarterly": dict(path="./data/train_quarterly.csv", freq=4, horizon=8),
    "yearly": dict(path="./data/train_yearly.csv", freq=1, horizon=6),
}


def smape(true, prediction):
    numerator = (true - prediction).abs()
    denominator = true.abs() + prediction.abs()
    result = 200 * (numerator / denominator).mean()
    return result


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

            decomposition = seasonal_decompose(
                values, model="multiplicative", period=freq
            )

            # plot if the flag is on
            if create_plots:
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
    model = theta_model
    # test loop
    for key, value in metadata.items():
        print({key})
        # file path for model 1
        filename = f"results/{key}_1.csv"

        # process all time series from a csv file
        horizon = metadata[key]["horizon"]
        series = process_category(key, metadata[key])
        train, dev = split_train_dev(series, horizon)

        # convert dict in a list of list (ID + value)
        data_rows = [[serie_id] + list(s.values) for serie_id, s in dev.items()]
        df_final = pd.DataFrame(data_rows)
        df_final.columns = ["ID"] + [f"V{i}" for i in range(1, df_final.shape[1])]
        df_final.to_csv(filename, index=False)

        for id in train.keys():
            print(f"{id}: smape(horizon = {horizon}, period = {value['freq']})")
            results[id] = smape(dev[id], model(train[id], horizon, value["freq"]))

    df_results = pd.DataFrame(list(results.items()))
    df_results.to_csv(f"{model.__name__}_results.csv", index=False)
    print("FINAL SCORE: ", df_results[1].mean())


if __name__ == "__main__":
    main()
