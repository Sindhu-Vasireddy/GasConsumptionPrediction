import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Read temperature data
temp_data = pd.read_csv("2018_to_2022_Temp.csv")
temp_values = temp_data.iloc[:, :].values.flatten(order='F')
temp_values = temp_values.reshape(-1, 1)

# Read energy consumption data
consumption_data = pd.read_csv("2018_to_2022_GC.csv")
consumption_values = consumption_data.iloc[:, 1:].values.flatten(order='F')
consumption_values = consumption_values.reshape(-1, 1)

# Data Processing 
N = len(temp_values)
X_train = temp_values[:N*2//3]
X_test = temp_values[-N//3:]
y_train = consumption_values[:N*2//3]
y_test = consumption_values[-N//3:]

# Define the model
model = Sequential()
model.add(Dense(256, input_dim=1, activation='relu'))
model.add(Dense(1)) 

optimizer = Adam(learning_rate=0.01)
model.compile(optimizer=optimizer, loss='mean_squared_error')

history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=0)

# Train Vs Validation Loss
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.title('Training vs Validation MSE for Neural Network')
plt.show()

train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Calculate and print the mean squared error
mse = mean_squared_error(y_test, test_predictions)
print(f"Mean Squared Error: {mse}")
# Calculate and print the R-squared error
r_squared = r2_score(y_test, test_predictions)
print(f"R-squared (R2): {r_squared}")
# Calculate and print the mean Absolute error
mae = mean_absolute_error(y_test, test_predictions)
print(f"Mean Absolute Error (MAE): {mae}")

# Actual vs predicted consumption
plt.figure(figsize=(10, 6))
plt.plot(y_test[1:], label='Actual Consumption', color='blue')
plt.plot(test_predictions[1:], label='Predicted Consumption', color='orange')
plt.title('Actual vs Predicted Consumption')
plt.xlabel('Time')
plt.ylabel('Consumption')
plt.legend()
plt.show()
