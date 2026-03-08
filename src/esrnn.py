import pandas as pd
from ESRNN.m4_data import prepare_m4_data
from ESRNN import ESRNN
from data.data_charger import get_m4_data, train_val_split
from pathlib import Path
import json
import torch
from metrics.evaluation_metrics import smape


frequencies = ["Hourly", "Daily", "Weekly", "Monthly", "Quarterly", "Yearly"]
val_fraction = 0.2

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

DATA_DIR = Path.cwd().parent.resolve() / "data"

root_dir = Path.cwd().parent.resolve()
json_file = root_dir / "configs" / "esrnn_configs.json"
esrnn_dir = root_dir / "models" / "esrnn"

with json_file.open("r", encoding="utf-8") as f:
    configs = json.load(f)

for freq in frequencies:
    params = configs[freq]

    X_train, y_train, _, _ = prepare_m4_data(dataset_name=freq, directory=str(DATA_DIR), num_obs=None)

    model = ESRNN(
        max_epochs=params["max_epochs"],
        freq_of_test=5,
        batch_size=params["batch_size"],
        learning_rate=params["learning_rate"],
        per_series_lr_multip=params["per_series_lr_multip"],
        lr_scheduler_step_size=params["lr_scheduler_step_size"],
        lr_decay=params["lr_decay"],
        gradient_clipping_threshold=params["gradient_clipping_threshold"],
        rnn_weight_decay=params["rnn_weight_decay"],
        level_variability_penalty=params["level_variability_penalty"],
        testing_percentile=params["testing_percentile"],
        training_percentile=params["training_percentile"],
        ensemble=params["ensemble"],
        max_periods=params["max_periods"],
        seasonality=params["seasonality"],
        input_size=params["input_size"],
        output_size=params["output_size"],
        cell_type=params["cell_type"],
        state_hsize=params["state_hsize"],
        dilations=params["dilations"],
        add_nl_layer=params["add_nl_layer"],
        random_seed=params["random_seed"],
        device=device
    )

    model.fit(X_train, y_train)

    torch.save(model, esrnn_dir / f"esrnn_{freq}.pt")