import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Read temperature data
temp_data = pd.read_csv("2018_to_2022_Temp.csv")
temp_values = temp_data.iloc[:, :].values.flatten(order='F')
temp_values = temp_values.reshape(-1, 1)

# Read energy consumption data
consumption_data = pd.read_csv("2018_to_2022_GC.csv")
consumption_values = consumption_data.iloc[:, 1:].values.flatten(order='F')
consumption_values = consumption_values.reshape(-1,1)

# Data Processing 
N = len(temp_values)
X_train = temp_values[:N*2//3]
X_test = temp_values[-N//3:]
y_train = consumption_values[:N*2//3]
y_test = consumption_values[-N//3:]

model = LinearRegression()
model.fit(X_train, y_train)

train_predictions = model.predict(X_train)
train_error = mean_squared_error(y_train, train_predictions)

test_predictions = model.predict(X_test)
test_error = mean_squared_error(y_test, test_predictions)

# Train Vs Validation Loss
plt.figure(figsize=(8, 6))
plt.plot([1], [train_error], marker='o', markersize=8, label='Training Loss')
plt.plot([1], [test_error], marker='o', markersize=8, label='Validation Loss')
plt.xlabel('Linear Regression')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.title('Training vs Validation MSE for Linear Regression')
plt.show()

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
