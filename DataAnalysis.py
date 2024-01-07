import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

# Read temperature data
temp_data = pd.read_csv("2018_to_2022_Temp.csv")
temp_values = temp_data.iloc[:, :].values.flatten(order='F')
temp_values = temp_values.reshape(-1, 1)

# Read energy consumption data
consumption_data = pd.read_csv("2018_to_2022_GC.csv")
consumption_values = consumption_data.iloc[:, 1:].values.flatten(order='F')
consumption_values = consumption_values.reshape(-1,1)

# scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(temp_values, consumption_values, color='blue')
plt.title('Scatter Plot of Columns')
plt.xlabel('Temperature')
plt.ylabel('Gas Consumption')
plt.show()

# Calculating correlation
correlation_coefficient, p_value = pearsonr(temp_values.reshape(-1), consumption_values.reshape(-1))
print(f"Correlation Coefficient: {correlation_coefficient}")
