# pip install darts matplotlib pandas

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import TFTModel
from darts.dataprocessing.transformers import Scaler
from darts.utils.likelihood_models import QuantileRegression

# 1. Generate synthetic noisy sinusoidal data
np.random.seed(42)
times = pd.date_range("20210101", periods=300, freq="H")
signal = np.sin(np.linspace(0, 30, 300)) + np.random.normal(0, 0.3, 300)
series = TimeSeries.from_times_and_values(times, signal.astype(np.float32).reshape(-1, 1))

# 2. Scale the data
scaler = Scaler()
series_scaled = scaler.fit_transform(series)

# 3. Split into training and validation (last 12 for forecasting)
train, val = series_scaled[:-12], series_scaled[-12:]

# 4. Define TFT model
model = TFTModel(
    input_chunk_length=24,
    output_chunk_length=12,
    hidden_size=16,
    lstm_layers=1,
    num_attention_heads=4,
    dropout=0.1,
    batch_size=32,
    n_epochs=3,
    likelihood=QuantileRegression(quantiles=[0.1, 0.5, 0.9]),
    add_encoders={
        "cyclic": {"future": ["hour", "dayofweek"]},
        "datetime_attribute": {"future": ["hour", "day"]}
    },
    random_state=42
)

# 5. Train
model.fit(train)

# 6. Predict full quantile forecast (probabilistic TimeSeries)
forecast = model.predict(n=12, num_samples=100)  # num_samples > 1 enables quantile extraction

# 7. Extract quantiles from forecast
p10 = forecast.quantile_timeseries(0.1)
p50 = forecast.quantile_timeseries(0.5)
p90 = forecast.quantile_timeseries(0.9)

# 8. Plot forecast with quantile interval
plt.figure(figsize=(12, 6))
plt.plot(p50.time_index, p50.values(), label="P50 (Median)", color='black')
plt.fill_between(p10.time_index, p10.values().flatten(), p90.values().flatten(),
                 color='gray', alpha=0.4, label="P10–P90 Interval")
plt.plot(val.time_index, val.values(), label="Actual", color='blue')
plt.title("TFT Forecast with Quantile Bands")
plt.xlabel("Time")
plt.ylabel("Scaled Value")
plt.legend()
plt.grid(True)
plt.show()

# 9. Custom quantile loss function
def quantile_loss(y_true, y_pred, q):
    delta = y_true.values() - y_pred.values()
    return np.mean(np.maximum(q * delta, (q - 1) * delta))

# 10. Print quantile losses
print(f"P10 Quantile Loss: {quantile_loss(val, p10, 0.1):.4f}")
print(f"P50 Quantile Loss: {quantile_loss(val, p50, 0.5):.4f}")
print(f"P90 Quantile Loss: {quantile_loss(val, p90, 0.9):.4f}")