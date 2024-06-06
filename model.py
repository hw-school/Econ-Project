import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

years = [2012, 2015, 2016, 2017, 2018,2019,2020,2021,2022,2023]

df = pd.read_csv('filtered.csv', index_col = 0)
columns = df.columns
happiness_df = pd.read_csv('Happiness Index.csv', index_col = 0)

from sklearn import linear_model
clf = linear_model.LinearRegression()

X = []
y = []
for year in years:
    X.append(df.loc[year, :].values.tolist())
    happiness = happiness_df.loc[year, :].values.tolist()[0]
    happiness = (7 - happiness) * 100
    y.append(happiness)

clf = linear_model.LinearRegression()
clf.fit(X,y)


from sklearn.metrics import mean_squared_error, r2_score
predictions = clf.predict(X)
mse = mean_squared_error(y, predictions)
r2 = r2_score(y, predictions)
print(df.columns.tolist())
print("Coefficients:", clf.coef_)
print("Intercept:", clf.intercept_)
print("Mean Squared Error:", mse)
print("R² Score:", r2)
print("Predictions:", predictions)

df_2024 = pd.read_csv('2024.csv', index_col=0)
data = df_2024.loc[2024].values
prediction = clf.predict([data])[0]
predicted_index = 7-prediction/100
print(predicted_index)

happiness_indices = happiness_df.loc[years].values
plt.plot(years, happiness_indices, marker = 'o', label = 'Actual Happiness Index')
plt.plot(2024, predicted_index, 'ro', label = 'Predicted Index in 2024')
plt.legend()
plt.savefig('figures/happiness index.jpg')
plt.close()



