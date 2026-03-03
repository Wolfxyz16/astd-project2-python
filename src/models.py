from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.forecasting.theta import ThetaModel
from pysr import PySRRegressor
import numpy as np
import pandas as pd


# must return an array of size horizon with the predictions
def arima_model(train, horizon):
    model = ARIMA(train, order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=horizon)
    return forecast


def theta_model(train, horizon, period=24):
    model = ThetaModel(train, period=period, deseasonalize=True)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=horizon)
    return forecast


def symbolic_genetic_model(serie, horizon, freq):
    # 1. Preparar los datos: Convertimos la serie en (X, y)
    # Usamos el índice como 'tiempo' (característica básica)
    y = serie.values
    X = np.arange(len(y)).reshape(-1, 1)  # Tiempo: 0, 1, 2, 3...

    # 2. Configurar el Algoritmo Genético
    model = PySRRegressor(
        niterations=40,  # Cuántas "generaciones" evoluciona
        binary_operators=["+", "*", "-", "/"],
        unary_operators=["sin", "exp", "cos"],
        model_selection="best",  # Elige la fórmula con mejor balance complejidad/error
        verbosity=0,  # Para que no ensucie tu consola de bucle
    )

    # 3. FIT: Aquí ocurre la selección natural
    model.fit(X, y)

    # 4. PREDICT: Predecimos los pasos futuros en el horizonte
    X_future = np.arange(len(y), len(y) + horizon).reshape(-1, 1)
    forecast = model.predict(X_future)

    return pd.Series(forecast)
