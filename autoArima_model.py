import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import argparse
import numpy as np
import statsmodels.api as sm
import pmdarima as pm
import os

metadata = {
    "hourly": dict(path="./data/train_hourly.csv", freq=24, horizon=48),
    "daily": dict(path="./data/train_daily.csv", freq=1, horizon=14),
    "weekly": dict(path="./data/train_weekly.csv", freq=1, horizon=13),
    "monthly": dict(path="./data/train_monthly.csv", freq=12, horizon=18),
    "quarterly": dict(path="./data/train_quarterly.csv", freq=4, horizon=8),
    "yearly": dict(path="./data/train_yearly.csv", freq=1, horizon=6),
}

def process_category(sp, metadata, create_plots=False):
    """
    Load, clean and save decomposition plots for a single category.
    """
    info = metadata[sp]
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
                plt.savefig(f"plots/{sp}_{serie_id}.png", dpi=300)
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

def smape(y_true, y_pred):
    """
    Calculate the Symmetric Mean Absolute Percentage Error (SMAPE) between two series.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    numerator = np.abs(y_true - y_pred) * 2.0
    denominator = np.abs(y_true) + np.abs(y_pred)

    # avoid division by zero by replacing zeros in the denominator with a small number
    denominator = np.where(denominator == 0, 1e-10, denominator)

    return np.mean(numerator / denominator) * 100

def fit_predict_auto_arima(train_series, horizon, freq):
    """
    Looks for the best hyperparameters and trains the model automatically.
    """
    try:
        stationary = True if freq > 1 else False
        m_val = freq if stationary else 1

        model_auto = pm.auto_arima(
            train_series,
            seasonal=stationary,
            m=m_val,
            start_p=0, start_q=0,
            max_p=2, max_q=2,
            start_P=0, start_Q=0,
            max_P=2, max_Q=2,
            trace=True,
            error_action='ignore',
            suppress_warnings=True,
            out_of_sample_size=horizon,
            scoring='mae',
        )
        
        # once it finds the best model, it makes the prediction
        prediccion = model_auto.predict(n_periods=horizon)
        
        if np.sum(np.abs(prediccion)) < 0.001:
            print(" The model has predicted all zeros. Using Naive method.")
            return np.array([train_series.iloc[-1]] * horizon)
        
        return np.array(prediccion)        
    except Exception as e:
        print(f"The auto_arima has failed with this series. Using Naive method.")
        return np.array([train_series.iloc[-1]] * horizon)

def main():    
    output_dir = "./model_predictions/arima"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for category, info in metadata.items():
        horizon = info["horizon"]
        freq = info["freq"]
        full_series = process_category(category, metadata)
        file_name = f"{category}_1.csv"
        save_path = os.path.join(output_dir, file_name)   
        open(save_path, 'w').close() 
        
        counter = 0
        with open(save_path, "a") as f_out:
            for serie_id, full_values in full_series.items():
                counter += 1       
                prediction = fit_predict_auto_arima(full_values, horizon, freq)
                pred_str = ",".join([str(val) for val in prediction])
                line = f'"{serie_id}",{pred_str}\n'
                f_out.write(line)
        
if __name__ == "__main__":
    main()