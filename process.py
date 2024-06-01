import os
import pandas as pd
from pathlib import Path
# Define the path to the folder containing the CSV files
folder_path = 'raw'

min_year = 1966
# Initialize a dictionary to store aggregated data
data_dict = {year: [] for year in range(min_year, 2024)}

def extract_year(date_str):
    try:
        date = pd.to_datetime(date_str, errors='coerce')
        if pd.notnull(date):
            return date.year
    except Exception as e:
        print(f"Error converting date: {date_str}, Error: {e}")
    return None
rows = []
for filename in os.listdir(folder_path):
    name = Path(filename).stem
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)
        df['Year'] = df.iloc[:, 0].apply(extract_year)

        df_filtered = df[(df['Year'] >= min_year) & (df['Year'] <= 2023)]

        for year in range(min_year, 2024):
            value = df_filtered[df_filtered['Year'] == year].iloc[:, 1].mean()
            data_dict[year].append(value)
    rows.append(name)


results_df = pd.DataFrame(data_dict)
results_df.index = rows
results_df = results_df.T

results_df.to_csv('values.csv', index = True)

# Display the results
print(results_df)