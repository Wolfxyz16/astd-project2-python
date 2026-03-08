import pandas as pd
import numpy as np
import torch
from chronos import Chronos2Pipeline
import warnings
import os

warnings.filterwarnings("ignore")

metadata = {
    "hourly": dict(train="./data/train_hourly.csv", pd_freq="h", horizon=48),
    "daily": dict(train="./data/train_daily.csv", pd_freq="D", horizon=14),
    "weekly": dict(train="./data/train_weekly.csv", pd_freq="W", horizon=13),
    "monthly": dict(train="./data/train_monthly.csv", pd_freq="M", horizon=18),
    "quarterly": dict(train="./data/train_quarterly.csv", pd_freq="Q", horizon=8),
    "yearly": dict(train="./data/train_yearly.csv", pd_freq="Y", horizon=6),
}

def load_series(filepath):
    """
    Load time series data from a CSV file into a dictionary.
    """
    data = {}
    if not os.path.exists(filepath): return data
    with open(filepath, "r") as f:
        next(f)
        for line in f:
            tokens = line.strip().split(",")
            if len(tokens) < 2: continue
            serie_id = tokens[0].replace('"', "")
            values = np.array([float(x) for x in tokens[1:] if x != "NA" and x != ""])
            data[serie_id] = values
    return data

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
    Evaluate the performance of the Chronos model on the dataset.
    """
    # load starting dates
    ts_info = pd.read_csv("./data/TSinfo.csv")
    ts_info['StartingDate'] = pd.to_datetime(ts_info['StartingDate'], dayfirst=True)
    
    sd_dict = dict(zip(ts_info['ID'], ts_info['StartingDate']))

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[INFO] Aceleración por hardware: {device.upper()}")
    
    print("[INFO] Despertando al modelo fundacional en memoria (solo se hace una vez)...")
    pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=device)
    
    final_results = {}

    for category, info in metadata.items():
        horizon = info["horizon"]
        freq_alias = info.get("pd_freq", "D")
        
        raw_data_dict = load_series(info["train"])
        if not raw_data_dict: continue
        
        context_test_rows = []
        test_targets = {}
        
        for serie_id, values in raw_data_dict.items():
            min_len = horizon * 2
            if len(values) < min_len: 
                continue
                
            test_part = values[-horizon:]
            train_part = values[:-horizon]
            
            test_targets[serie_id] = test_part
            
            start_date = sd_dict.get(serie_id)
            dates_test = pd.date_range(start=start_date, periods=len(train_part), freq=freq_alias)
            
            for d, val in zip(dates_test, train_part):
                context_test_rows.append({"id": serie_id, "timestamp": d, "target": val})

        if not context_test_rows:
            print(f"Datos insuficientes para {category}")
            continue

        df_context_test = pd.DataFrame(context_test_rows)
        
        pred_test_df = pipeline.predict_df(
            df_context_test,
            prediction_length=horizon,
            quantile_levels=[0.5], 
            id_column="id",
            timestamp_column="timestamp",
            target="target"
        )
        
        test_smape_list = []
        
        numeric_columns = pred_test_df.select_dtypes(include=[np.number]).columns
        col_pred = [col for col in numeric_columns if col not in ["id", "timestamp"]][0]
        
        for serie_id in test_targets.keys():
            preds_test = pred_test_df[pred_test_df["id"] == serie_id][col_pred].values
            test_smape_list.append(smape(test_targets[serie_id], preds_test))

        cat_smape = np.mean(test_smape_list)
        final_results[category] = cat_smape

    print("Mean smape CHRONOS-2:\n")
    for cat, result in final_results.items():
        print(f" - {cat.upper():<10}: {result:>6.2f}%")
        
    if final_results:
        print(f"Macro average: {np.mean(list(final_results.values())):.2f}%")

def predict():    
    """
    This function trains a Chronos-2 model on the large dataset and generates predictions for the small dataset.
    It saves the predictions in CSV files, one per category.
    """
    # load starting dates
    ts_info = pd.read_csv("./data/TSinfo.csv")
    ts_info['StartingDate'] = pd.to_datetime(ts_info['StartingDate'], dayfirst=True)
    
    sd_dict = dict(zip(ts_info['ID'], ts_info['StartingDate']))
    
    output_dir = "./model_predictions/chronosSec"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    device = "mps" if torch.backends.mps.is_available() else "cpu"
        
    pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=device)
    
    for category, info in metadata.items():
        horizon = info["horizon"]
        freq_alias = info["pd_freq"]
        
        raw_data_dict = load_series(info["train"])
        if not raw_data_dict: continue
        
        context_rows = []
        series_order = []
        
        for serie_id, values in raw_data_dict.items():
            series_order.append(serie_id)
            
            start_date = sd_dict.get(serie_id)
            real_dates = pd.date_range(start=start_date, periods=len(values), freq=freq_alias)
            
            for d, val in zip(real_dates, values):
                context_rows.append({"id": serie_id, "timestamp": d, "target": val})

        df_context = pd.DataFrame(context_rows)
        
        file_name = f"{category}_chronos.csv"
        save_path = os.path.join(output_dir, file_name)
        
        pred_df = pipeline.predict_df(
            df_context,
            prediction_length=horizon,
            quantile_levels=[0.5],
            id_column="id",
            timestamp_column="timestamp",
            target="target"
        )
                
        numeric_columns = pred_df.select_dtypes(include=[np.number]).columns
        col_pred = [col for col in numeric_columns if col not in ["id", "timestamp"]][0]
        
        with open(save_path, "w") as f_out:
            header = "id," + ",".join([f"V{i+1}" for i in range(horizon)]) + "\n"
            f_out.write(header)
            
            for serie_id in series_order:
                preds = pred_df[pred_df["id"] == serie_id][col_pred].values
                pred_str = ",".join([f"{val:.4f}" for val in preds])
                f_out.write(f'"{serie_id}",{pred_str}\n')

if __name__ == "__main__":
    #performance()
    predict()