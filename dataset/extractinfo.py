import requests
import re
from io import BytesIO
import zipfile
import pandas as pd
import os
from pathlib import Path

# Function to download and extract zip files
def download_and_extract(url, destination):
    response = requests.get(url)
    with zipfile.ZipFile(BytesIO(response.content)) as z:
        for file in z.namelist():
            if file.endswith('.txt') and file.startswith("produkt_"):
                z.extract(file, destination)
                # Rename the extracted .txt file to .csv
                os.rename(os.path.join(destination, file), os.path.join(destination, file[:-4] + '.csv'))
# URL base and end date
base_url = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/air_temperature/historical/"
end_date = "20221231"

# List to store downloaded file paths
downloaded_files = []

# Loop through files and download relevant ones
for file in requests.get(base_url).text.split('\n'):
    if end_date in file:
 # Extract filename without HTML tags
        file_name_match = re.search(r'>(.*?)</a>', file)
        if file_name_match:
            file_name = file_name_match.group(1)
            file_url = base_url + file_name  # Construct complete file URL
            destination = "extracted_dataset"  # Folder name to extract files
            download_and_extract(file_url, destination)

# Read and filter data from CSV files
data_frames = []
# Process the CSV files in the destination folder
for file_name in os.listdir(destination):
    file_path = os.path.join(destination, file_name)
    df = pd.read_csv(file_path, delimiter=';')
    print("df",df)
    # Convert MESS_DATUM to datetime
    df['MESS_DATUM'] = pd.to_datetime(df['MESS_DATUM'], format='%Y%m%d%H')
    # Filter data between 2018010100 to 2022123123
    print("timestamp",pd.Timestamp('2018-01-01'))
    mask = (df['MESS_DATUM'] >= pd.Timestamp('2018-01-01')) & (df['MESS_DATUM'] <= pd.Timestamp('2022-12-31 23:00:00'))
    print("mask",mask)
    df_filtered = df.loc[mask]
    print("mask",df_filtered)
    data_frames.append(df_filtered)

# Merge data frames
merged_df = pd.concat(data_frames)

# Sort data by STATIONS_ID and MESS_DATUM
merged_df.sort_values(by=['STATIONS_ID', 'MESS_DATUM'], inplace=True)

# Save merged data to a single CSV file
merged_df.to_csv('merged_data.csv', index=False)
