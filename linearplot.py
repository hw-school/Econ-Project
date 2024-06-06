import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
df = pd.read_csv('filtered.csv', index_col = 0)
columns = df.columns.tolist()

predictions = {}
for column in columns:
    years = [2012, 2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023]
    yearsLinear = [2012, 2013,2014,2015,2016,2017,2018,2019,2021,2022,2023]
    

    print(column)
    df_feature = df[column].loc[years]
    features = df[column].loc[yearsLinear].values
    
    yearsLinear = np.array(yearsLinear).reshape(-1,1)
    clf = LinearRegression()
    clf.fit(yearsLinear, features)

    prediction_2024 = clf.predict([[2024]])[0]
    predictions[column] = prediction_2024

    y_pred = clf.predict(yearsLinear)



    plt.plot(yearsLinear, y_pred, linestyle='--', label=f'{column} Best-Fit Line')
    plt.plot(2024, prediction_2024, 'ro', label = f'{column} 2024 Prediction')
    df_feature.plot(figsize = (10,6), marker = 'o', label = f'{column} Actual Data')
    
    plt.xlabel('Year')
    plt.ylabel(f'{column}')
    plt.title(f'{column}, 2012-2023')
    plt.legend()
    plt.savefig(f'graphs/{column}.jpg')
    plt.close()
pred_df = pd.DataFrame(predictions, index=[2024])
pred_df.to_csv('2024.csv')