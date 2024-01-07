import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Activation
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score,mean_absolute_error

# Read temperature data
temp_data = pd.read_csv("2018_to_2022_Temp.csv")
temp_values = temp_data.iloc[:, :].values.flatten(order='F')
temp_values = temp_values.reshape(-1, 1)

# Read energy consumption data
consumption_data = pd.read_csv("2018_to_2022_GC.csv")
consumption_values = consumption_data.iloc[:, 1:].values.flatten(order='F')
consumption_values = consumption_values.reshape(-1,1)

# Data Preprocessing 
scaler_temp = StandardScaler()
scaler_cons = StandardScaler()

scaler_temp.fit(temp_values[:len(temp_values) * 2 // 3])
X_scaled_temp = scaler_temp.transform(temp_values)

scaler_cons.fit(consumption_values[:len(consumption_values) * 2 // 3])
X_scaled_cons = scaler_cons.transform(consumption_values)

X_scaled = np.concatenate((X_scaled_temp, X_scaled_cons), axis=1)

def create_sequences(data, window_size):
    sequences = np.zeros((len(data) - window_size, window_size, 2))
    gc = np.zeros(len(data) - window_size)

    for i in range(window_size,len(data)):
        sequences[i - window_size] = data[i - window_size : i,0:2]
        gc[i - window_size] = data[i, 1]
    sequences = sequences.reshape(-1, window_size, 2)
    gc = gc.reshape(-1, 1)
    return sequences,gc

window_size = 12 # 3 weeks data as a window
in_t_gc, out_gc = create_sequences(X_scaled, window_size)

N = len(in_t_gc)
X_train = in_t_gc[:N*2//3]
X_test = in_t_gc[-N//3:]
y_train = out_gc[:N*2//3]
y_test = out_gc[-N//3:]

X_train = X_train.reshape(-1, window_size, 2)
X_test = X_test.reshape(-1, window_size, 2)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Define the LSTM model
def create_lstm_model(input_dim, hidden_dim, output_dim, n_layers, window_size, drop_prob=0.2):
    model = Sequential()
    for i in range(n_layers):
        if i == 0:
            model.add(LSTM(hidden_dim, input_shape=(window_size, input_dim), return_sequences=True))
        elif i == n_layers - 1:
            model.add(LSTM(hidden_dim, return_sequences=False))
        else:
            model.add(LSTM(hidden_dim, return_sequences=True))
        model.add(Dropout(drop_prob))
    model.add(Dense(output_dim))
    return model

# Hyperparameters
input_dim = 2  # (temperature,consumption)
output_dim = 1  # (consumption)
hidden_dim = 256
n_layers = 3
dropout = 0.2

lstm_model = create_lstm_model(input_dim, hidden_dim, output_dim, n_layers,window_size, dropout)

lstm_model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error')

r=lstm_model.fit(X_train, y_train, epochs=300, batch_size=32, validation_data=(X_test,y_test), verbose=1)

# Train Vs Validation Loss
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.show()

# Print the (loss) mean Absolute error
loss = lstm_model.evaluate(in_t_gc, out_gc)
print(f"Mean Squared Error (MSE): {loss}")

predicted_consumption = lstm_model.predict(X_test)
predicted_consumption = predicted_consumption.reshape(-1,1)

y_test = y_test.reshape(-1, 1)

# Calculate and print the R-squared error
r_squared = r2_score(y_test, predicted_consumption)
print(f"R-squared (R2): {r_squared}")

# Calculate and print the mean Absolute error
mae = mean_absolute_error(y_test, predicted_consumption)
print(f"Mean Absolute Error (MAE): {mae}")

# Actual vs One step-predicted consumption
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual Consumption', color='blue')
plt.plot(predicted_consumption, label='Predicted Consumption', color='orange')
plt.title('Actual vs Predicted Consumption')
plt.xlabel('Time')
plt.ylabel('Consumption')
plt.legend()
plt.show()