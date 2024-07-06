import pandas as pd

# Load the dataset into a DataFrame
df = pd.read_csv("Mall_Customers.csv")

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(df.head())

# Display summary information about the dataset
print("\nSummary information about the dataset:")
print(df.info())

# Display basic statistics of the dataset
print("\nBasic statistics of the dataset:")
print(df.describe())


from sklearn.preprocessing import StandardScaler

# Selecting relevant features for clustering
# We will use 'Annual Income (k$)' and 'Spending Score (1-100)' for clustering
features = ['Annual Income (k$)', 'Spending Score (1-100)']

# Extract the selected features into a new DataFrame
data = df[features]

# Standardize the features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

print("\nScaled data sample:")
print(scaled_data[:5])


from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# Calculate the silhouette score for different number of clusters
silhouette_scores = []

for i in range(2, 11):  # Starting from 2 clusters as silhouette score is not defined for a single cluster
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    cluster_labels = kmeans.fit_predict(scaled_data)
    silhouette_avg = silhouette_score(scaled_data, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# Plot the silhouette scores
plt.plot(range(2, 11), silhouette_scores)
plt.title('Silhouette Score for Different Number of Clusters')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.grid(True)
plt.show()


# From the Silhouette graph, choose the optimal number of clusters (e.g., K=5)
optimal_clusters = 5

# Apply K-means clustering with the chosen number of clusters
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', max_iter=300, n_init=10, random_state=42)
cluster_labels = kmeans.fit_predict(scaled_data)

# Add the cluster labels to the original dataset
df['Cluster'] = cluster_labels

print("\nData with cluster labels:")
print(df.head())


import seaborn as sns

# Visualize the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df['Annual Income (k$)'], y=df['Spending Score (1-100)'], hue=df['Cluster'], palette='Set1', s=100)
plt.title('Clusters of Customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.show()

# Analyze the mean values of each cluster
print("\nMean values of each cluster:")
print(df.groupby('Cluster').mean())


