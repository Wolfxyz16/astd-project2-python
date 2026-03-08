from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.forecasting.theta import ThetaModel
from pysr import PySRRegressor
import numpy as np
import pandas as pd


# must return an array of size horizon with the predictions
def arima_model(train, horizon, id):
    model = ARIMA(train, order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=horizon)
    return forecast


def theta_model(train, horizon, period, id):
    """
    One of our very first approaches to the forecasting problem. Performs very well but it is not what we were looking for
    """
    model = ThetaModel(train, period=period, deseasonalize=True)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=horizon)
    return forecast


def forecast_symbolic(model, last_window, horizon):
    """
    Generates n-horizon predictions using a recursive approach.

    Args:
        model: The trained PySRRegressor.
        last_window (np.array): The last 'lookback' values of your series.
        horizon (int): Number of steps to forecast (e.g., 48).
    """
    predictions = []
    current_window = list(last_window)
    # The last time index used in training (needed for X0)
    current_time_idx = len(last_window)

    for _ in range(horizon):
        # 1. Prepare the input row: [Time_Index, Lag_N, ..., Lag_1]
        # X0 is time, the rest are the window values
        X_input = np.array([[current_time_idx] + current_window])

        # 2. Predict the next step
        next_val = model.predict(X_input)[0]
        predictions.append(next_val)

        # 3. Update for next iteration
        # Slide the window: remove oldest, add the new prediction
        current_window.pop(0)
        current_window.append(next_val)
        current_time_idx += 1

    return np.array(predictions)


def symbolic_genetic_model(serie, horizon, freq, id):
    lookback = max(freq, 5)

    # Convert series to numpy
    y_raw = serie.values
    n_points = len(y_raw)

    X_list = []
    y_list = []

    # 2. Feature Engineering: Create the Training Matrix
    # X0: Time index (t) -> Captures the overall trend
    # X1, X2...: Past values (lags) -> Captures local patterns
    for i in range(lookback, n_points):
        # Time index feature
        time_idx = [i]
        # Past values features
        lags = y_raw[i - lookback : i].tolist()

        X_list.append(time_idx + lags)
        y_list.append(y_raw[i])

    X = np.array(X_list)
    y = np.array(y_list)

    loss_function = """
        function my_loss(tree, dataset, options)
            prediction, flag = eval_tree_array(tree, dataset.X, options)
            if !flag
                return Inf
            end

            base_loss = sum(abs.(prediction .- dataset.y)) / length(dataset.y)

            for node in tree
                if node.degree == 0 && node.feature == 1
                    break
                end
            end

            return base_loss
        end
    """

    # PySRRegressor library has a tons of different parameters
    # we will configure these parameter in different dictionaries and then
    # combine them together
    creating_search_space = {
        "binary_operators": ["+", "*", "-", "/"],
        "unary_operators": ["sin", "cos"],
        "maxsize": 30,
        "maxdepth": None,
    }

    setting_search_space = {
        # my i5-1340p has 6 performance cores, we will use 6 population
        "niterations": 100,
        "populations": 6 * 3,
        "population_size": 50,
        "ncycles_per_iteration": 500,
    }

    objective = {
        "elementwise_loss": None,
        "loss_function": loss_function,  # we could implement our custom loss function in julia
        "model_selection": "best",
        "nested_constraints": {  # dont nest sin and cos
            "sin": {"sin": 0, "cos": 0},
            "cos": {"sin": 0, "cos": 0},
        },
    }
    stopping_criteria = {
        "max_evals": None,
        "timeout_in_seconds": 60 * 60 * 12,
        "early_stop_condition": None,
    }

    monitoring = {
        "verbosity": 0,
        "update_verbosity": None,
        "print_precision": 5,
        "progress": True,
    }

    # combine all dictionary and unroll them as parameters
    model = PySRRegressor(
        **(
            creating_search_space
            | setting_search_space
            | objective
            | stopping_criteria
            | monitoring
        )
    )

    # FIT, here is where the genetic happens
    model.fit(X, y)

    # Save the latex equations in the directory
    filename = "./pysr_equations/" + id + ".tex"

    with open(filename, "w", encoding="utf-8") as file:
        file.write(model.latex())

    last_observed_values = serie.values[-lookback:]

    return pd.Series(forecast_symbolic(model, last_observed_values, horizon))
