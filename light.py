import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.multioutput import MultiOutputRegressor
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

def load_series(filepath):
    data = {}
    if not os.path.exists(filepath):
        print(f"[!] Archivo no encontrado: {filepath}")
        return data
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
    X, Y = [], []
    for i in range(len(data_array) - window_size - horizon + 1):
        X.append(data_array[i : (i + window_size)])
        Y.append(data_array[(i + window_size) : (i + window_size + horizon)])
    return X, Y

def smape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    numerator = np.abs(y_true - y_pred) * 2.0
    denominator = np.abs(y_true) + np.abs(y_pred)
    denominator = np.where(denominator == 0, 1e-10, denominator)
    return np.mean(numerator / denominator) * 100

def main():
    print("\n[INFO] Iniciando Custom Grid Search (Entrenando en LARGE, evaluando en SMALL)...")
    
    print("[INFO] Cargando los datasets MASSIVE en memoria (para entrenamiento)...")
    loaded_data_large = {}
    for cat, meta in metadata.items():
        loaded_data_large[cat] = load_series(meta["train"])
        
    print("[INFO] Cargando los datasets SMALL en memoria (para saber qué evaluar)...")
    loaded_data_small = {}
    for cat, meta in metadata_small.items():
        loaded_data_small[cat] = load_series(meta["train"])
        
    hyperparameters = [
        (100, 0.05), (150, 0.05), (300, 0.05), 
        (500, 0.03), (500, 0.05), (500, 0.08), (500, 0.1), 
        (1000, 0.05), (2000, 0.05), (2500, 0.05), (2000, 0.1), (2000, 0.03)
    ]
    
    resultados_para_csv = []

    for n_est, lr in hyperparameters:
        print(f"\n{'='*50}")
        print(f"🛠️  EVALUANDO: n_estimators={n_est} | learning_rate={lr}")
        print(f"{'='*50}")
        
        results = {}

        for category, meta_info in metadata.items():
            horizon = meta_info["horizon"]
            freq = meta_info["freq"]
            window_size = max(freq * 2, horizon * 2)
            
            raw_data_large = loaded_data_large[category]
            if not raw_data_large: continue

            train_dict = {}
            test_dict = {}
            all_X, all_Y = [], []

            # 1. Preparar TODO el dataset grande para entrenar
            for serie_id, values in raw_data_large.items():
                min_required_length = window_size + horizon
                if len(values) <= min_required_length:
                    continue 
                
                # Separamos el final para poder testear (si es que la serie nos interesa luego)
                test_part = values[-horizon:]
                train_part = values[:-horizon]
                
                train_dict[serie_id] = train_part
                test_dict[serie_id] = test_part
                
                # Tabularizamos para que el modelo aprenda
                X_wins, Y_wins = create_tabular_data(train_part, window_size, horizon)
                all_X.extend(X_wins)
                all_Y.extend(Y_wins)

            if not all_X: continue

            X_train = np.array(all_X)
            Y_train = np.array(all_Y)
            
            # 2. Entrenamos el modelo con TODA la información
            model = MultiOutputRegressor(lgb.LGBMRegressor(
                n_estimators=n_est,
                learning_rate=lr,
                max_depth=5,
                random_state=42,
                n_jobs=-1,
                verbosity=-1
            ))
            model.fit(X_train, Y_train)
            
            # 3. Evaluamos SOLO las series que están en el dataset pequeño
            target_ids = set(loaded_data_small[category].keys())
            cat_smapes = []
            
            for serie_id in train_dict.keys():
                if serie_id not in target_ids:
                    continue # Ignoramos las series de relleno, solo nos interesan las del examen
                
                train_part = train_dict[serie_id]
                test_part = test_dict[serie_id]
                
                last_train_window = train_part[-window_size:].reshape(1, -1)
                pred_test = model.predict(last_train_window).flatten()
                
                if np.sum(np.abs(pred_test)) < 0.001:
                    pred_test = np.array([train_part[-1]] * horizon)
                    
                cat_smapes.append(smape(test_part, pred_test))
            
            if cat_smapes:
                avg_smape = np.mean(cat_smapes)
                results[category] = avg_smape
                print(f"  [✔] {category.upper():<10} procesado -> sMAPE: {avg_smape:>6.2f}%")
            else:
                print(f"  [!] {category.upper():<10} procesado -> Sin series válidas para evaluar.")

        if results:
            macro_avg = np.mean(list(results.values()))
            print(f"{'-'*35}")
            print(f"🎯 MACRO AVERAGE GLOBAL:  {macro_avg:>6.2f}%")
            
            fila_csv = {
                'n_estimators': n_est,
                'learning_rate': lr,
                'hourly_smape': results.get('hourly', np.nan),
                'daily_smape': results.get('daily', np.nan),
                'weekly_smape': results.get('weekly', np.nan),
                'monthly_smape': results.get('monthly', np.nan),
                'quarterly_smape': results.get('quarterly', np.nan),
                'yearly_smape': results.get('yearly', np.nan),
                'macro_average': macro_avg
            }
            resultados_para_csv.append(fila_csv)

    df_resultados = pd.DataFrame(resultados_para_csv)
    df_resultados = df_resultados.sort_values(by='macro_average')
    
    nombre_archivo = "historial_grid_search_massive.csv"
    df_resultados.to_csv(nombre_archivo, index=False)

    print(f"\n\n{'='*50}")
    print("🏆 RANKING FINAL DE HIPERPARÁMETROS (Train: Large, Test: Small)")
    print(f"{'='*50}")
    
    for index, row in df_resultados.iterrows():
        print(f"n={int(row['n_estimators'])}, lr={row['learning_rate']} -> {row['macro_average']:.2f}%")
        
    print(f"\n💾 [ÉXITO] Todos los detalles se han guardado en '{nombre_archivo}'")

if __name__ == "__main__":
    main()