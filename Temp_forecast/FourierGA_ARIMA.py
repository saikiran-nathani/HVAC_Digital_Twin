import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.fft import fft, ifft
from datetime import datetime
import warnings
import random

warnings.filterwarnings("ignore")


def apply_fft_filter(series, threshold_ratio=0.1):
    fft_vals = fft(series)
    power = np.abs(fft_vals)
    threshold = threshold_ratio * np.max(power)
    fft_filtered = [val if abs(val) > threshold else 0 for val in fft_vals]
    return np.real(ifft(fft_filtered))


def evaluate_arima_model(series, order):
    try:
        model = ARIMA(series, order=order)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=10)
        actual = series[-10:]
        forecast = forecast[:len(actual)]
        return mean_squared_error(actual, forecast)
    except:
        return np.inf

def ga_optimize_arima(series, population_size=10, generations=5):
    def generate_population():
        return [(random.randint(0, 5), 1, random.randint(0, 5)) for _ in range(population_size)]

    def crossover(p1, p2):
        return tuple(random.choice([p1[i], p2[i]]) for i in range(3))

    def mutate(ind):
        idx = random.randint(0, 2)
        values = list(ind)
        values[idx] = random.randint(0, 5)
        return tuple(values)

    population = generate_population()

    for _ in range(generations):
        scores = [(ind, evaluate_arima_model(series, ind)) for ind in population]
        scores.sort(key=lambda x: x[1])
        top = [x[0] for x in scores[:population_size//2]]
        population = top[:]
        while len(population) < population_size:
            p1, p2 = random.sample(top, 2)
            child = crossover(p1, p2)
            if random.random() < 0.3:
                child = mutate(child)
            population.append(child)

    best_individual = min(population, key=lambda x: evaluate_arima_model(series, x))
    return best_individual


def calculate_metrics(actual, forecast):
    mae = mean_absolute_error(actual, forecast)
    mse = mean_squared_error(actual, forecast)
    rmse = np.sqrt(mse)
    r2 = 1 - (np.sum((actual - forecast) ** 2) / np.sum((actual - np.mean(actual)) ** 2))
    return mae, mse, rmse, r2

def forecast_with_hybrid_arima(data):
    results = {}

    for col in data.columns:
        if col == 'created_at':
            continue

        print(f"\nProcessing {col}")
        series = pd.to_numeric(data[col], errors='coerce').fillna(method='ffill')

       
        filtered_series = apply_fft_filter(series)

        
        best_order = ga_optimize_arima(filtered_series)
        print(f"Best ARIMA order for {col}: {best_order}")

        
        model = ARIMA(filtered_series, order=best_order)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=10)

       
        actual = filtered_series[-10:]
        mae, mse, rmse, r2 = calculate_metrics(actual, forecast[:10])
        results[col] = {
            'forecast': forecast,
            'metrics': {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2},
            'order': best_order
        }

    return results


if __name__ == "__main__":
    filepath = "thingspeak_all_data.csv"
    data = pd.read_csv(filepath)
    data['created_at'] = pd.to_datetime(data['created_at'])

    results = forecast_with_hybrid_arima(data)

    
    for col, res in results.items():
        plt.figure(figsize=(10, 4))
        plt.title(f"Forecast for {col} | ARIMA{res['order']}")
        plt.plot(range(10), res['forecast'], label='Forecast')
        plt.legend()
        plt.grid(True)
        plt.show()

    print("\nForecasting Metrics:")
    for col, res in results.items():
        m = res['metrics']
        print(f"{col}: MAE={m['MAE']:.3f}, MSE={m['MSE']:.3f}, RMSE={m['RMSE']:.3f}, R2={m['R2']:.3f}")