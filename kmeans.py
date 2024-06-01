import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns


file_path = 'values.csv'
data = pd.read_csv(file_path, index_col=0) 
features = data.columns.tolist()
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[features])


inertia = []
K = range(1, 11) 

# test for elbow
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(K, inertia, "bo-", markersize=8)
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal k')
plt.savefig('./figures/elbow.jpg')
plt.close()


optimal_k = 4

#kmeans
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(scaled_data)


data['Cluster'] = clusters

# Display the dataframe with cluster labels
pca = PCA(n_components=2)
components = pca.fit_transform(scaled_data)

# PCA Analysis
pca_df = pd.DataFrame(data=components, columns=['PC1', 'PC2'])
pca_df['Cluster'] = clusters
pca_df['Year'] = data.index

plt.figure(figsize=(14, 10)) 
sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=pca_df, palette='viridis')
plt.title('PCA Clusters')

# Annotate each point
for i in range(pca_df.shape[0]):
    plt.annotate(pca_df['Year'].iloc[i], (pca_df['PC1'].iloc[i], pca_df['PC2'].iloc[i]), 
                 fontsize=6)
plt.savefig('./figures/clusters_annotated.jpg')
plt.close() 
