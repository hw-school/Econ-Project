import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

df_total = pd.read_csv('filtered.csv', index_col=0)
df = df_total.loc[[2014,2015,2016,2017,2018,2019,2021,2022,2023]]
# Standardize the data
standardized_data = (df - df.mean()) / df.std()

# Apply PCA
pca = PCA()
pca.fit(standardized_data)

explained_variance = np.cumsum(pca.explained_variance_ratio_)
num_components = np.argmax(explained_variance >= 0.95) + 1


pca = PCA(n_components=num_components)
principal_components = pca.fit_transform(standardized_data)
composite_index = principal_components.sum(axis=1)


components = pca.components_

weights = np.sum(components, axis=0)

weights /= np.sum(np.abs(weights))

feature_weights = pd.Series(weights, index=df.columns)
print("Feature Weights:")
print(feature_weights)
feature_weights.to_csv('weights.csv')


df['Composite Index'] = composite_index
df_2020 = df_total.loc[2020]
df_2020 = (df_2020 - df_2020.mean()) / df_2020.std()


plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Composite Index'], "bo-", label='Composite Index')
plt.title('Economic Composite Index')
plt.xlabel('Time')
plt.ylabel('Composite Index')
plt.legend()
plt.savefig('figures/model.jpg')
