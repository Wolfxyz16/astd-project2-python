from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.forecasting.theta import ThetaModel


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
