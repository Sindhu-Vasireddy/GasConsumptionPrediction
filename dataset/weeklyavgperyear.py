import pandas as pd

# Read the CSV file into a pandas DataFrame
data = pd.read_csv('merged_data.csv')

# Convert MESS_DATUM column to datetime format
data['MESS_DATUM'] = pd.to_datetime(data['MESS_DATUM'], format='%Y-%m-%d %H:%M:%S')

# Extracting date and hour separately for aggregation
data['Date'] = data['MESS_DATUM'].dt.date
data['Hour'] = data['MESS_DATUM'].dt.hour

# Calculate the weighted averages for TT_TU and RF_TU columns
data['Weighted_TT_TU'] = data['TT_TU'] * data['QN_9']
data['Weighted_RF_TU'] = data['RF_TU'] * data['QN_9']

# Aggregate by Date and Hour to get daily hourly weighted sums
hourly_agg = data.groupby(['Date', 'Hour']).agg({
    'Weighted_TT_TU': 'sum',
    'Weighted_RF_TU': 'sum',
    'QN_9': 'sum'
}).reset_index()

# Calculate the weighted average across stations for each hour of the day
hourly_agg['Hourly_Weighted_Avg_TT_TU'] = hourly_agg['Weighted_TT_TU'] / hourly_agg['QN_9']
hourly_agg['Hourly_Weighted_Avg_RF_TU'] = hourly_agg['Weighted_RF_TU'] / hourly_agg['QN_9']

# Group by year and week to calculate the weekly averages
hourly_agg['Year'] = pd.to_datetime(hourly_agg['Date']).dt.year
hourly_agg['Week_Number'] = pd.to_datetime(hourly_agg['Date']).dt.strftime('%U')

weekly_avg = hourly_agg.groupby(['Year', 'Week_Number']).agg({
    'Hourly_Weighted_Avg_TT_TU': 'mean',
    'Hourly_Weighted_Avg_RF_TU': 'mean'
}).reset_index()

# Renaming columns
weekly_avg = weekly_avg.rename(columns={
    'Hourly_Weighted_Avg_TT_TU': 'Weekly_Weighted_Avg_TT_TU',
    'Hourly_Weighted_Avg_RF_TU': 'Weekly_Weighted_Avg_RF_TU'
})

# Save the resulting DataFrame to a CSV file
weekly_avg.to_csv('Weekly_Temp_Humidity_Avg.csv', index=False)

# Pivot the data
pivot_data = weekly_avg.pivot_table(index='Week_Number', columns='Year', values=['Weekly_Weighted_Avg_TT_TU']).droplevel(0, axis=1)

# Remove the last row
pivot_data = pivot_data.iloc[:-1]

pivot_data.to_csv("../2018_to_2022_Temp.csv",index=False)

