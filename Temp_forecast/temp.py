import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.metrics import mean_squared_error
import torch
import os

# Limit CPU threads for better GPU usage
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

file_path = "setpoint19.csv"
df = pd.read_csv(file_path)

df.columns = df.columns.str.strip()

column_mapping = {
    "Var1": "Evap_Out",
    "Var2": "Comp_Out",
    "Var3": "Cond_Out",
    "Var4": "Evap_In",
    "Var5": "COP"
}

df.rename(columns=column_mapping, inplace=True)

expected_features = ["Evap_Out", "Comp_Out", "Cond_Out", "Evap_In"]
target_column = "COP"

missing_features = [col for col in expected_features if col not in df.columns]
if missing_features:
    raise ValueError(f"Missing columns in dataset: {missing_features}")

X = df[expected_features].values
y = df[target_column].values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert data to tensors and move to GPU
X_train, y_train = torch.tensor(X_train, dtype=torch.float32).to(device), torch.tensor(y_train, dtype=torch.float32).to(device)
X_test, y_test = torch.tensor(X_test, dtype=torch.float32).to(device), torch.tensor(y_test, dtype=torch.float32).to(device)

tabnet_model = TabNetRegressor(
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
    scheduler_params={"step_size": 50, "gamma": 0.9},
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    mask_type="entmax",
    device_name=device  # Ensure it runs on CUDA
)

tabnet_model.fit(
    X_train=X_train.cpu().numpy(),  # TabNet requires numpy arrays
    y_train=y_train.cpu().numpy(),
    eval_set=[(X_test.cpu().numpy(), y_test.cpu().numpy())],
    eval_metric=["rmse"],
    max_epochs=20,
    patience=20,
    batch_size=1024,
    virtual_batch_size=64
)

y_pred = tabnet_model.predict(X_test.cpu().numpy())
tabnet_mse = mean_squared_error(y_test.cpu().numpy(), y_pred)

print(f"TabNet Model Trained Successfully!")
print(f"Final MSE: {tabnet_mse:.4f}")

tabnet_model.save_model("tabnet_cop_model")
tabnet_model.load_model("tabnet_cop_model.zip")
