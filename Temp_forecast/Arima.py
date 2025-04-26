import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

warnings.filterwarnings("ignore")


hvac_data = pd.read_csv('C:/Users/sai kiran nathani/PycharmProjects/ML/hvac_data.csv')
print(hvac_data.head())


missing_values = hvac_data.isnull().sum()
print("Missing values:\n", missing_values)


hvac_data.fillna(method='ffill', inplace=True)



def check_stationarity_and_difference(series):
    result = adfuller(series)
    if result[1] > 0.05:
        return np.diff(series), False
    return series, True


# Function to fit ARIMA model
def fit_arima(series):
    model = ARIMA(series, order=(5, 1, 0))
    model_fit = model.fit()
    return model_fit


# Function to calculate forecast metrics
def calculate_metrics(actual, forecast):
    mae = mean_absolute_error(actual, forecast)
    mse = mean_squared_error(actual, forecast)
    rmse = np.sqrt(mse)
    r2 = 1 - np.sum((actual - forecast) ** 2) / np.sum((actual - np.mean(actual)) ** 2)
    return mae, mse, rmse, r2


# Initialize a dictionary to store the results
forecast_results = {}

# Create subplots for ACF of residuals
fig_acf, axes_acf = plt.subplots(4, 2, figsize=(15, 20))
axes_acf = axes_acf.flatten()

# Loop through each field and perform the operations
for idx, field in enumerate(hvac_data.columns[2:10]):
    print(f"\nProcessing {field}")

    # Check stationarity and differencing
    series = hvac_data[field]
    series_diff, is_stationary = check_stationarity_and_difference(series)
    if not is_stationary:
        series = series_diff

    # Fit ARIMA model
    model_fit = fit_arima(series)

    # Make forecast
    forecast = model_fit.forecast(steps=5)
    forecast_results[field] = forecast

    # Calculate metrics
    mae, mse, rmse, r2 = calculate_metrics(series[-5:], forecast)

    print(f"MAE: {mae}")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"R-squared: {r2:.4f}")


    pd.plotting.autocorrelation_plot(pd.Series(model_fit.resid), ax=axes_acf[idx])
    axes_acf[idx].set_title(f"ACF of Residuals for {field}")

plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 8))
for i, field in enumerate(hvac_data.columns[2:10], 1):
    plt.subplot(4, 2, i)
    plt.plot(hvac_data[field][-100:], label='Actual')
    plt.plot(pd.Series(np.concatenate([hvac_data[field][-5:].values, forecast_results[field]]),
                       index=np.arange(len(hvac_data[field]) - 5, len(hvac_data[field]) + 5)), label='Forecast')
    plt.title(field)
    plt.legend()
plt.tight_layout()
plt.show()

print(forecast_results)
