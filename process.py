import os
import pandas as pd
from pathlib import Path
folder_path = 'raw'

min_year = 1987
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
            try:
                value = df_filtered[df_filtered['Year'] == year].iloc[:, 1].mean()
            except:
                print(name)
            data_dict[year].append(value)
    rows.append(name)


results_df = pd.DataFrame(data_dict)
results_df.index = rows
results_df = results_df.T

results_df.to_csv('values.csv', index = True)

# Display the results
#print(results_df)

not_used = ['New Housing Units', 'GDP Deflator', 'Gini Coefficient', 'Industrial Production Index']
results_df.drop(not_used, axis = 1, inplace = True)
results_df.to_csv('filtered.csv', index = True)
