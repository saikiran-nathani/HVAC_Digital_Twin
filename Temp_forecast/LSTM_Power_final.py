import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

data = pd.read_csv('setpoint19.csv')

target_column = 'COP'
feature_columns = data.columns.difference([target_column, 'Time'])  # Exclude 'Time' and target


X = data[feature_columns].values
y = data[target_column].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)


class MyModel(nn.Module):
    def __init__(self, input_size):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = MyModel(input_size=X_train.shape[1]).to(device)


criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


epochs = 200
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor.to(device))
    loss = criterion(outputs, y_train_tensor.to(device))
    loss.backward()
    optimizer.step()


model.eval()
with torch.no_grad():
    train_predictions = model(X_train_tensor.to(device)).cpu().numpy()
    test_predictions = model(X_test_tensor.to(device)).cpu().numpy()


train_mae = mean_absolute_error(y_train, train_predictions)
train_mse = mean_squared_error(y_train, train_predictions)
train_rmse = np.sqrt(train_mse)
train_r_squared = r2_score(y_train, train_predictions)


test_mae = mean_absolute_error(y_test, test_predictions)
test_mse = mean_squared_error(y_test, test_predictions)
test_rmse = np.sqrt(test_mse)
test_r_squared = r2_score(y_test, test_predictions)


plt.figure(figsize=(8, 6))
plt.scatter(y_train, train_predictions, color='blue', label='Train Predict')
plt.scatter(y_test, test_predictions, color='red', label='Test Predict')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='black', linestyle='--')


plt.title('Predicted vs Calculated Energy')
plt.xlabel('Calculated Energy')
plt.ylabel('Predicted Energy')
plt.legend()


metrics_text = (f"Train MAE: {train_mae:.3f}\n"
                f"Test MAE: {test_mae:.3f}\n"
                f"Train RMSE: {train_rmse:.3f}\n"
                f"Test RMSE: {test_rmse:.3f}\n"
                f"Train R²: {train_r_squared:.3f}\n"
                f"Test R²: {test_r_squared:.3f}")
print(metrics_text)

plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))


plt.show()


last_known_data = X_test[-1].reshape(1, -1)
future_predictions = []


previous_data = y_test[-50:]


plot_data = np.concatenate((previous_data, future_predictions))


x_actual = np.arange(50)
x_forecast = np.arange(50, 50 + len(future_predictions))


plt.figure(figsize=(10, 6))
plt.plot(x_actual, previous_data, label="Actual", color="blue", linestyle="-", marker="o")  # Actual data
plt.plot(x_forecast, future_predictions, label="Forecast", color="orange", linestyle="-", marker="x")  # Forecasted data


plt.title("Actual Data with Forecasted Continuation")
plt.xlabel("Time Steps")
plt.ylabel("Target Variable (COP)")
plt.legend()


plt.show()

n_future = 10
for _ in range(n_future):
    last_known_tensor = torch.tensor(last_known_data, dtype=torch.float32).to(device)
    next_prediction = model(last_known_tensor).detach().cpu().numpy()
    future_predictions.append(next_prediction[0][0])

    # Generate new input data based on the last prediction
    last_known_data = np.concatenate((last_known_data.flatten()[:-1], next_prediction.flatten())).reshape(1, -1)

print("Future Predictions:", future_predictions)
