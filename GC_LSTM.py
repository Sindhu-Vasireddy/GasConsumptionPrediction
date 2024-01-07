import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD, Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read energy consumption data
consumption_data = pd.read_csv("2018_to_2022_GC.csv")
consumption_values = consumption_data.iloc[:, 1:].values.flatten(order='F')

# Data Preprocessing 
scaler = MinMaxScaler(feature_range=(0, 1))
consumption_values_scaled = scaler.fit_transform(consumption_values.reshape(-1, 1)).flatten()

def create_sequences(data, window_size):
    X = np.zeros((len(data) - window_size, window_size, 1))
    gc = np.zeros(len(data) - window_size)

    for i in range(window_size,len(data)):
        X[i - window_size] = data[i - window_size : i]
        gc[i - window_size] = data[i]
    X = X.reshape(-1, window_size, 1)
    gc = gc.reshape(-1, 1)
    return X,gc

window_size = 12 # 3 weeks data as a window
X, consumption_sequences = create_sequences(consumption_values_scaled.reshape(-1,1), window_size)

print("X.shape", X.shape, "consumption_sequences.shape", consumption_sequences.shape)

# Define the LSTM model
def create_lstm_model(input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
    model = Sequential()
    for i in range(n_layers):
        if i == 0:
            model.add(LSTM(hidden_dim, input_shape=(None, input_dim), return_sequences=True))
        elif i == n_layers - 1:
            model.add(LSTM(hidden_dim, return_sequences=False))
        else:
            model.add(LSTM(hidden_dim, return_sequences=True))
        model.add(Dropout(drop_prob))

    model.add(Dense(output_dim))
    return model

# Hyperparameters
input_dim = 1 # (consumption)
output_dim = 1 # (consumption)
hidden_dim = 256
n_layers = 5
dropout = 0.2

lstm_model = create_lstm_model(input_dim, hidden_dim, output_dim, n_layers, dropout)

lstm_model.compile(optimizer='adam', loss='mean_squared_error')

N=X.shape[0]
r=lstm_model.fit(X[:-N//10],consumption_sequences[:-N//10],batch_size=32,epochs=200,validation_data=(X[-9*N//10:], consumption_sequences[-9*N//10:]),verbose=1)

# Train Vs Validation Loss
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.show()

# One-step forecast 
outputs = lstm_model.predict(X)
print(outputs.shape)
predictions = outputs[:,0]

# Actual vs predicted consumption
plt.plot(consumption_sequences, label='Actual consumption')
plt.plot(predictions, label='Predicted consumption')
plt.title("Actual vs Predicted Consumption (One step forecast)")
plt.xlabel('Time')
plt.ylabel('Consumption')
plt.legend()
plt.show()

# Multi-step forecast
forecast = []
input_ = X[-N//2]
while len(forecast) < len(consumption_sequences[-N//2:]):
  f = lstm_model.predict(input_.reshape(1, window_size, 1))[0,0]
  forecast.append(f)

  input_ = np.roll(input_, -1)
  input_[-1] = f

# Actual vs predicted consumption
plt.plot(consumption_sequences[-N//2:], label='Actual consumption')
plt.plot(forecast, label='Predicted consumption')
plt.title("Actual vs Predicted Consumption (Multi-step forecast)")
plt.xlabel('Time')
plt.ylabel('Consumption')
plt.legend()
plt.show()
