import pandas as pd
from scipy.stats import pearsonr
import numpy as np

df = pd.read_csv('values.csv', index_col = 0)

columns = df.columns
#print(columns)

correlation_matrix = pd.DataFrame(index=columns, columns=columns)

for col1 in columns:
    for col2 in columns:
        if col1 == col2:
            correlation_matrix.loc[col1, col2] = 1.0  
        else:
            correlation, _ = pearsonr(df[col1], df[col2])
            correlation_matrix.loc[col1, col2] = correlation

correlation_matrix.to_csv('correlation_matrix.csv')
