import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.multioutput import MultiOutputRegressor
import argparse
import warnings
import os

warnings.filterwarnings("ignore")

metadata_small = {
    "hourly": dict(train="./data/train_hourly.csv", freq=24, horizon=48),
    "daily": dict(train="./data/train_daily.csv", freq=1, horizon=14),
    "weekly": dict(train="./data/train_weekly.csv", freq=1, horizon=13),
    "monthly": dict(train="./data/train_monthly.csv", freq=12, horizon=18),
    "quarterly": dict(train="./data/train_quarterly.csv", freq=4, horizon=8),
    "yearly": dict(train="./data/train_yearly.csv", freq=1, horizon=6),
}

metadata = {
    "hourly": dict(train="./train_limpio/Hourly-train.csv", freq=24, horizon=48),
    "daily": dict(train="./train_limpio/Daily-train.csv", freq=1, horizon=14),
    "weekly": dict(train="./train_limpio/Weekly-train.csv", freq=1, horizon=13),
    "monthly": dict(train="./train_limpio/Monthly-train.csv", freq=12, horizon=18),
    "quarterly": dict(train="./train_limpio/Quarterly-train.csv", freq=4, horizon=8),
    "yearly": dict(train="./train_limpio/Yearly-train.csv", freq=1, horizon=6),
}

# data loading
def load_series(filepath):
    """
    Load time series data from a CSV file into a dictionary.
    """
    data = {}
    with open(filepath, "r") as f:
        next(f)
        for line in f:
            tokens = line.strip().split(",")
            if len(tokens) < 2: continue
            serie_id = tokens[0].replace('"', "")
            values = np.array([float(x) for x in tokens[1:] if x != "NA" and x != ""])
            data[serie_id] = values
    return data

def create_tabular_data(data_array, window_size, horizon):
    """
    This is the magic part: turning a time series into a tabular ML problem.
    We convert historical steps into 'features' (columns) and future steps into 'targets'.
    """
    X, Y = [], []
    for i in range(len(data_array) - window_size - horizon + 1):
        X.append(data_array[i : (i + window_size)])
        Y.append(data_array[(i + window_size) : (i + window_size + horizon)])
    return X, Y

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

def performance():    
    """
    This function trains a LightGBM model on the large dataset and evaluates it on the small dataset.
    It prints the sMAPE for each category and a macro average at the end.
    """
    final_results = {}
    
    for category, info in metadata.items():
        horizon = info["horizon"]
        freq = info["freq"]
        
        raw_data_dict = load_series(info["train"])
        if not raw_data_dict:
            continue
        
        window_size = max(freq * 2, horizon * 2)
        
        train_dict, test_dict = {} , {}
        all_X, all_Y = [], []

        for serie_id, values in raw_data_dict.items():
            min_required_length = window_size + horizon
            if len(values) <= min_required_length:
                continue 
                
            test_part = values[-horizon:]
            train_part = values[:-horizon]
            
            train_dict[serie_id] = train_part
            test_dict[serie_id] = test_part
            
            X_wins, Y_wins = create_tabular_data(train_part, window_size, horizon)
            all_X.extend(X_wins)
            all_Y.extend(Y_wins)

        if not all_X:
            print(f"Not enough data to train {category}. Skipping...")
            continue

        X_train = np.array(all_X)
        Y_train = np.array(all_Y)

        base_model = lgb.LGBMRegressor(
            n_estimators=2000,
            learning_rate=0.03,
            max_depth=5, 
            random_state=42,
            n_jobs=-1,
            verbosity=-1 
        )
        
        model = MultiOutputRegressor(base_model)
        
        # Training all rows in one go
        model.fit(X_train, Y_train)
        
        
        test_smape_list = []
        
        # Load the small dataset just to retrieve the exact IDs we need to evaluate
        data_small = load_series(metadata_small[category]["train"])
        target_ids = set(data_small.keys())
        
        for serie_id in train_dict.keys():
            if serie_id not in target_ids:
                continue
                
            train_part = train_dict[serie_id]
            test_part = test_dict[serie_id]
            
            # The history window is just the last part of the training set
            last_train_window = train_part[-window_size:].reshape(1, -1)
            pred_test = model.predict(last_train_window).flatten()
            
            if np.sum(np.abs(pred_test)) < 0.001:
                pred_test = np.array([train_part[-1]] * horizon)
                
            test_smape_list.append(smape(test_part, pred_test))

        if test_smape_list:
            cat_smape = np.mean(test_smape_list)
            final_results[category] = cat_smape
            print(f"  [✔] {category.upper():<10} processed -> sMAPE: {cat_smape:>6.2f}%")

    print("Mean sMAPE LightGBM:\n")
    for cat, result in final_results.items():
        print(f" - {cat.upper():<10}: {result:>6.2f}%")
        
    if final_results:
        print(f"Macro average: {np.mean(list(final_results.values())):.2f}%")


def predict():
    """
    This function trains a LightGBM model on the large dataset and generates predictions for the small dataset.
    It saves the predictions in CSV files, one per category.
    """    

    output_dir = "./model_predictions/lightgbm12"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for category in metadata.keys():
        horizon = metadata[category]["horizon"]
        freq = metadata[category]["freq"]
        window_size = max(freq * 2, horizon * 2)
        
        raw_data_large = load_series(metadata[category]["train"])
        
        all_X, all_Y = [], []
        for serie_id, values in raw_data_large.items():
            if len(values) <= window_size + horizon:
                continue 
            X_wins, Y_wins = create_tabular_data(values, window_size, horizon)
            all_X.extend(X_wins)
            all_Y.extend(Y_wins)

        if not all_X:
            print(f"Not enough data to train {category}. Skipping...")
            continue

        X_train = np.array(all_X)
        Y_train = np.array(all_Y)
        
        base_model = lgb.LGBMRegressor(
            n_estimators=2000,
            learning_rate=0.03,
            max_depth=5,            
            random_state=42,
            n_jobs=-1,
            verbosity=-1            
        )
        model = MultiOutputRegressor(base_model)
        model.fit(X_train, Y_train)

        raw_data_small = load_series(metadata_small[category]["train"])
        
        file_name = f"{category}_lightgbm.csv"
        save_path = os.path.join(output_dir, file_name)
        
        open(save_path, 'w').close() 
                
        counter = 0
        with open(save_path, "a") as f_out:
            for serie_id, values in raw_data_small.items():
                counter += 1
                
                if len(values) < window_size:
                    prediction = np.array([values[-1]] * horizon) #serie is too short, we predict the last value repeated
                else:
                    last_window = values[-window_size:].reshape(1, -1)
                    prediction = model.predict(last_window).flatten()
                    
                    if np.sum(np.abs(prediction)) < 0.001:
                        prediction = np.array([values[-1]] * horizon)
                        
                pred_str = ",".join([str(val) for val in prediction])
                line = f'"{serie_id}",{pred_str}\n'
                f_out.write(line)

if __name__ == "__main__":
    performance()
    #predict()