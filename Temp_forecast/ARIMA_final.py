import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime
import warnings
import requests
import time
import io

warnings.filterwarnings("ignore")

def fetch_data_from_thingspeak(channel_id, read_api_key, results=8000):
    url = f"https://api.thingspeak.com/channels/{channel_id}/feeds.csv?api_key={read_api_key}&results={results}"
    response = requests.get(url)
    data = pd.read_csv(io.StringIO(response.text))
    data['created_at'] = pd.to_datetime(data['created_at'])
    return data

def clean_data(data):
    if 'created_at' in data.columns:
        data.drop(['entry_id'], axis=1, inplace=True)

    for col in data.columns:
        data[col] = preprocess_field(data[col])

    data.fillna(method='ffill', inplace=True)
    return data

def preprocess_field(series):
    series = series.replace(r'[^0-9.-]', '', regex=True)
    series = pd.to_numeric(series, errors='coerce')
    series.fillna(method='ffill', inplace=True)
    return series

def check_stationarity_and_difference(series):
    series = series.dropna()
    if len(series) < 10:
        return pd.Series(series), True
    result = adfuller(series)
    if result[1] > 0.05:
        return pd.Series(np.diff(series)), False
    return pd.Series(series), True

def fit_arima(series, order):
    model = ARIMA(series, order=order)
    model_fit = model.fit()
    return model_fit

def calculate_metrics(actual, forecast):
    if len(actual) != len(forecast):
        return np.nan, np.nan, np.nan, np.nan  # Handle cases where lengths do not match
    
    mae = mean_absolute_error(actual, forecast)
    mse = mean_squared_error(actual, forecast)
    rmse = np.sqrt(mse)

    mean_actual = np.mean(actual)
    ss_total = np.sum((actual - mean_actual) ** 2)
    ss_residual = np.sum((actual - forecast) ** 2)

    r2 = 1 - (ss_residual / ss_total) if ss_total > 0 else np.nan  # Avoid division by zero
    
    return mae, mse, rmse, r2
def save_metrics_as_image(metrics, filename):
    fig, ax = plt.subplots(figsize=(10, len(metrics) * 0.5))
    ax.axis('tight')
    ax.axis('off')

    table_data = [[field, f"{values['MAE']:.4f}", f"{values['MSE']:.4f}", f"{values['RMSE']:.4f}", f"{values['R2']:.4f}"] 
                  for field, values in metrics.items()]
    column_labels = ["Field", "MAE", "MSE", "RMSE", "R2"]
    table = ax.table(cellText=table_data, colLabels=column_labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    plt.savefig(filename)
    print(f"Metrics saved as {filename}")

def forecast_hvac_data(hvac_data):
    forecast_results = {}
    metrics = {}

    orders = (7, 1, 0)

    for field in hvac_data.columns:
        if field == 'created_at':
            continue

        print(f"\nProcessing {field}")

        series = hvac_data[field]

        series_diff, is_stationary = check_stationarity_and_difference(series)
        if not is_stationary:
            series = series_diff

        if len(series) < 10:
            print(f"Not enough data points for {field} after differencing.")
            continue

        series = series.dropna()

        try:
            
            model_fit = fit_arima(series, orders)

           
            forecast = model_fit.forecast(steps=10)

            
            mae, mse, rmse, r2 = calculate_metrics(series[-10:], forecast[:10])
            metrics[field] = {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}

            forecast_results[field] = forecast

            print(f"MAE: {mae}")
            print(f"MSE: {mse}")
            print(f"RMSE: {rmse}")
            print(f"R-squared: {r2:.4f}")
        except Exception as e:
            print(f"An error occurred while processing {field}: {e}")

    
    num_fields = len([field for field in hvac_data.columns if field != 'created_at'])
    num_cols = 2
    num_rows = (num_fields + num_cols - 1) // num_cols
    plt.figure(figsize=(15, num_rows * 5))

    subplot_index = 1
    for field in hvac_data.columns:
            if field == 'created_at':
                continue
            if field in forecast_results:
                plt.subplot(num_rows, num_cols, subplot_index)
                plt.plot(hvac_data[field][-100:], label='Actual')
                forecast_index = np.arange(len(hvac_data[field]), len(hvac_data[field]) + 15)
                forecast_series = pd.Series(forecast_results[field], index=forecast_index)
                plt.plot(forecast_series, label='Forecast')
                plt.title(field)
                plt.legend()
                subplot_index += 1

    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plot_filename = f"ARIMA_plot_{timestamp}.png"
    plt.savefig(plot_filename)
    print(f"Plot saved as {plot_filename}")

    metrics_filename = f"ARIMA_metrics_{timestamp}.jpg"
    save_metrics_as_image(metrics, metrics_filename)

CHANNEL_ID = '2834542'
READ_API_KEY = 'SR5O5P9FU4Z93RF9'

while True:
    try:
        hvac_data = fetch_data_from_thingspeak(CHANNEL_ID, READ_API_KEY)
        hvac_data = clean_data(hvac_data)
        forecast_hvac_data(hvac_data)

        time.sleep(900)
    except Exception as e:
        print(f"An error occurred: {e}")
        time.sleep(60)
