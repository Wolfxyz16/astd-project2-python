import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import argparse
import warnings
import copy

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

# architecture
class GlobalLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_step_out = lstm_out[:, -1, :]
        prediction = self.linear(last_step_out)
        return prediction

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

def create_sliding_windows(data_array, window_size, horizon):
    """
    Tturns a time series into a tabular ML problem.
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

def trainSmall():
    """
    This function trains and evaluates the LSTM model on the SMALL dataset.
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    resultados_dev = {}
    resultados_test = {}

    for category, info in metadata_small.items():
        horizon = info["horizon"]
        freq = info["freq"]
        
        raw_data_dict = load_series(info["train"])
        window_size = max(freq * 2, horizon * 2)
        
        train_dict, dev_dict, test_dict = {}, {}, {}
        scalers = {}
        all_X, all_Y = [], []
        val_X_list, val_Y_list = [], []

        for serie_id, values in raw_data_dict.items():
            min_required_length = window_size + (2 * horizon)
            if len(values) <= min_required_length:
                continue 
                
            test_part = values[-horizon:]
            dev_part = values[-(2 * horizon) : -horizon]
            train_part = values[:- (2 * horizon)]
            
            train_dict[serie_id] = train_part
            dev_dict[serie_id] = dev_part
            test_dict[serie_id] = test_part
            
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_train = scaler.fit_transform(train_part.reshape(-1, 1))
            scalers[serie_id] = scaler
            
            X_wins, Y_wins = create_sliding_windows(scaled_train, window_size, horizon)
            all_X.extend(X_wins)
            all_Y.extend(Y_wins)
            
            last_train_window = scaled_train[-window_size:]
            scaled_dev_part = scaler.transform(dev_part.reshape(-1, 1))
            
            val_X_list.append(last_train_window)
            val_Y_list.append(scaled_dev_part.flatten())

        if not all_X:
            print(f"Not enough data to train {category}. Skipping...")
            continue

        X_tensor = torch.tensor(np.array(all_X), dtype=torch.float32)
        Y_tensor = torch.tensor(np.array(all_Y), dtype=torch.float32).squeeze(-1)
        
        val_X_tensor = torch.tensor(np.array(val_X_list), dtype=torch.float32).to(device)
        val_Y_tensor = torch.tensor(np.array(val_Y_list), dtype=torch.float32).to(device)
        
        batch_size = 64
        dataset = TensorDataset(X_tensor, Y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        model = GlobalLSTM(input_size=1, hidden_size=64, num_layers=2, output_size=horizon).to(device)
        criterion = nn.L1Loss() 
        optimizer = optim.Adam(model.parameters(), lr=0.005)

        epochs = 100 
        patience = 5
        patience_counter = 0
        best_val_loss = float('inf')
        
        import copy
        best_model_weights = None
        
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            for batch_X, batch_Y in dataloader:
                batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_Y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                
            train_loss_avg = epoch_loss / len(dataloader)
            
            model.eval()
            with torch.no_grad():
                val_outputs = model(val_X_tensor)
                val_loss = criterion(val_outputs, val_Y_tensor).item()
                
            print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss_avg:.4f} | Val Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_weights = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n Early Stopping: the validation loss hasn't improved for {patience} epochs. Stopping early!")
                    break

        model.load_state_dict(best_model_weights)
        model.eval()
        
        dev_smape_list, test_smape_list = [], []
        
        with torch.no_grad():
            for serie_id in train_dict.keys():
                scaler = scalers[serie_id]
                train_part = train_dict[serie_id]
                dev_part = dev_dict[serie_id]
                test_part = test_dict[serie_id]
                
                last_train_window = scaler.transform(train_part[-window_size:].reshape(-1, 1))
                input_dev = torch.tensor(last_train_window, dtype=torch.float32).unsqueeze(0).to(device)
                
                pred_dev_scaled = model(input_dev).cpu()
                pred_dev = scaler.inverse_transform(pred_dev_scaled.numpy().reshape(-1, 1)).flatten()
                
                if np.sum(np.abs(pred_dev)) < 0.001:
                    pred_dev = np.array([train_part[-1]] * horizon)
                    
                dev_smape_list.append(smape(dev_part, pred_dev))
                
                history_for_test = np.concatenate((train_part, dev_part))
                last_history_window = scaler.transform(history_for_test[-window_size:].reshape(-1, 1))
                input_test = torch.tensor(last_history_window, dtype=torch.float32).unsqueeze(0).to(device)
                
                pred_test_scaled = model(input_test).cpu()
                pred_test = scaler.inverse_transform(pred_test_scaled.numpy().reshape(-1, 1)).flatten()
                
                if np.sum(np.abs(pred_test)) < 0.001:
                    pred_test = np.array([history_for_test[-1]] * horizon)
                    
                test_smape_list.append(smape(test_part, pred_test))

        cat_dev_smape = np.mean(dev_smape_list)
        cat_test_smape = np.mean(test_smape_list)
        resultados_dev[category] = cat_dev_smape
        resultados_test[category] = cat_test_smape

    print("Mean sMAPE LSTM (Small Dataset):\n")
    for cat in resultados_test.keys():
        print(f" - {cat.upper():<10} | DEV: {resultados_dev[cat]:>6.2f}% | TEST: {resultados_test[cat]:>6.2f}%")
        
    if resultados_test:
        print(f"{'-'*50}")
        print(f"Macro Average: {np.mean(list(resultados_test.values())):.2f}%")

def trainBig():
    """
    This function trains the LSTM model on the BIG dataset and evaluates it on the small dataset.
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    resultados_dev = {}
    resultados_test = {}

    for category, info in metadata.items():
        horizon = info["horizon"]
        freq = info["freq"]
        
        raw_data_dict = load_series(info["train"])
        window_size = max(freq * 2, horizon * 2)
        
        train_dict, dev_dict, test_dict = {}, {}, {}
        scalers = {}
        all_X, all_Y = [], []
        val_X_list, val_Y_list = [], []

        for serie_id, values in raw_data_dict.items():
            min_required_length = window_size + (2 * horizon)
            if len(values) <= min_required_length:
                continue 
                
            test_part = values[-horizon:]
            dev_part = values[-(2 * horizon) : -horizon]
            train_part = values[:- (2 * horizon)]
            
            train_dict[serie_id] = train_part
            dev_dict[serie_id] = dev_part
            test_dict[serie_id] = test_part
            
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_train = scaler.fit_transform(train_part.reshape(-1, 1))
            scalers[serie_id] = scaler
            
            X_wins, Y_wins = create_sliding_windows(scaled_train, window_size, horizon)
            all_X.extend(X_wins)
            all_Y.extend(Y_wins)
            
            last_train_window = scaled_train[-window_size:]
            scaled_dev_part = scaler.transform(dev_part.reshape(-1, 1))
            
            val_X_list.append(last_train_window)
            val_Y_list.append(scaled_dev_part.flatten())

        if not all_X:
            print(f"Not enough data to train {category}. Skipping...")
            continue

        X_tensor = torch.tensor(np.array(all_X), dtype=torch.float32)
        Y_tensor = torch.tensor(np.array(all_Y), dtype=torch.float32).squeeze(-1)
        
        val_X_tensor = torch.tensor(np.array(val_X_list), dtype=torch.float32).to(device)
        val_Y_tensor = torch.tensor(np.array(val_Y_list), dtype=torch.float32).to(device)
        
        batch_size = 64
        dataset = TensorDataset(X_tensor, Y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        model = GlobalLSTM(input_size=1, hidden_size=64, num_layers=2, output_size=horizon).to(device)
        criterion = nn.L1Loss() 
        optimizer = optim.Adam(model.parameters(), lr=0.005)

        epochs = 100 
        patience = 5
        patience_counter = 0
        best_val_loss = float('inf')
        
        import copy
        best_model_weights = None
        
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            for batch_X, batch_Y in dataloader:
                batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_Y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                
            train_loss_avg = epoch_loss / len(dataloader)
            
            model.eval()
            with torch.no_grad():
                val_outputs = model(val_X_tensor)
                val_loss = criterion(val_outputs, val_Y_tensor).item()
                
            print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss_avg:.4f} | Val Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_weights = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n Early Stopping: validation loss hasn't improved for {patience} epochs.")
                    break

        model.load_state_dict(best_model_weights)
        model.eval()
        
        dev_smape_list, test_smape_list = [], []
        
        # only evaluate the series that are in the small dataset test set
        try:
            datos_pequenos = load_series(metadata_small[category]["train"])
            ids_del_examen = set(datos_pequenos.keys())
        except Exception:
            print(f"  [!] No se pudo cargar el archivo pequeño para {category}. Evaluando todo.")
            ids_del_examen = set(train_dict.keys())
        
        with torch.no_grad():
            for serie_id in train_dict.keys():
                if serie_id not in ids_del_examen:
                    continue
                    
                scaler = scalers[serie_id]
                train_part = train_dict[serie_id]
                dev_part = dev_dict[serie_id]
                test_part = test_dict[serie_id]
                
                last_train_window = scaler.transform(train_part[-window_size:].reshape(-1, 1))
                input_dev = torch.tensor(last_train_window, dtype=torch.float32).unsqueeze(0).to(device)
                
                pred_dev_scaled = model(input_dev).cpu()
                pred_dev = scaler.inverse_transform(pred_dev_scaled.numpy().reshape(-1, 1)).flatten()
                
                if np.sum(np.abs(pred_dev)) < 0.001:
                    pred_dev = np.array([train_part[-1]] * horizon)
                    
                dev_smape_list.append(smape(dev_part, pred_dev))
                
                history_for_test = np.concatenate((train_part, dev_part))
                last_history_window = scaler.transform(history_for_test[-window_size:].reshape(-1, 1))
                input_test = torch.tensor(last_history_window, dtype=torch.float32).unsqueeze(0).to(device)
                
                pred_test_scaled = model(input_test).cpu()
                pred_test = scaler.inverse_transform(pred_test_scaled.numpy().reshape(-1, 1)).flatten()
                
                if np.sum(np.abs(pred_test)) < 0.001:
                    pred_test = np.array([history_for_test[-1]] * horizon)
                    
                test_smape_list.append(smape(test_part, pred_test))

        if test_smape_list:
            cat_dev_smape = np.mean(dev_smape_list)
            cat_test_smape = np.mean(test_smape_list)
            resultados_dev[category] = cat_dev_smape
            resultados_test[category] = cat_test_smape

    print("Mean sMAPE LSTM:\n")
    for cat in resultados_test.keys():
        print(f" - {cat.upper():<10} | DEV: {resultados_dev[cat]:>6.2f}% | TEST: {resultados_test[cat]:>6.2f}%")
        
    if resultados_test:
        print(f"🎯 Macro Average: {np.mean(list(resultados_test.values())):.2f}%")


if __name__ == "__main__":
    trainSmall()
    trainBig()