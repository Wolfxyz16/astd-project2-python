import pandas as pd
import matplotlib.pyplot as plt
# from statsmodels.tsa.seasonal import seasonal_decompose

metadata = {
    "hourly": dict(path="./data/train_hourly.csv", freq=24, horizon=48),
    "daily": dict(path="./data/train_daily.csv", freq=1, horizon=14),
    "weekly": dict(path="./data/train_weekly.csv", freq=1, horizon=13),
    "monthly": dict(path="./data/train_monthly.csv", freq=12, horizon=18),
    "quarterly": dict(path="./data/train_quarterly.csv", freq=4, horizon=8),
    "yearly": dict(path="./data/train_yearly.csv", freq=1, horizon=1),
}

# in ts_dict we store all the different categories (hourly, daily, ...)
# and inside each category we store all the time series
ts_dict = {}

for key, value in metadata.items():
    file_path = value["path"]

    file_dict = {}  # in this dict we store all the ts in one file

    with open(file_path, "r") as f:
        next(f)  # skip the header row
        for line in f:
            tokens = line.strip().split(",")
            serie_id = tokens[0].replace('"', "")
            values = pd.Series([float(x) for x in tokens[1:] if x != "NA"]).dropna()

            # lets plot each time serie and store it in plots directory
            plt.figure()
            plt.grid(True)
            values.plot(title=serie_id, color="teal")
            plt.savefig(f"plots/{key}_{serie_id}.png", dpi=300)
            plt.close()

            file_dict[serie_id] = values

    ts_dict[key] = file_dict

# print(ts_dict["hourly"].keys())
print(type(ts_dict["hourly"]["H1"]))
