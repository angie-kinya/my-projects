import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("data/Mall_Customers.csv")
print(data.head())

# DATA PROCESSING
# Select relevant features for clustering
X = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Display the scaled data
print(X_scaled[:5])

# APPLY K-MEANS CLUSTERING
# Determine the optimal number of clusters using Elbow Method
wcss = []   # Within Cluster Sum of Squares - measures how compact the clusters are
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Apply KMeans with the optimal number of clusters 
kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_means = kmeans.fit_predict(X_scaled)

# Add cluster labels to the original dataset
data['Cluster'] = y_means
print(data.head())

# Visualize  the clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=y_means, palette='Set1', s=100)
plt.title('Customer Segments')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
# Save the plot as a file (e.g., PNG format)
plt.savefig('cluster_plot.png')

# ANALYZE THE CLUSTERS
# Descriptive statistics for each cluster
cluster_summary = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].groupby(data['Cluster']).mean()
print(cluster_summary)

# Visualize the clusters in 3D
from  mpl_toolkits.mplot3d import Axes3D
# 3D plot using the 3 features ( Age, Annual Income, Spending Score)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X_scaled[:, 0], X_scaled[:, 1],  X_scaled[:, 2], c=y_means, s=50, cmap='viridis')
ax.set_title('3D View of Customer Segments')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Feature 3')
plt.savefig('3Dcluster_plot.png')

# Silhouette score
from sklearn.metrics import silhouette_score

score = silhouette_score(X_scaled, y_means)
print(f'Silhouette Score: {score:.3f}')