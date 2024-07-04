import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load the dataset

data = pd.read_csv("Mall_Customers.csv")

# Assuming 'Annual Income (k$)' and 'Spending Score (1-100)' are relevant features
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Feature scaling (optional)
# You can use Min-Max scaling or StandardScaler from sklearn.preprocessing

# Determine the optimal number of clusters (K)
age_ss = []
for k in range(1, 8):
    kmeans = KMeans(n_clusters=k, init='k-means++')
    kmeans.fit(X)
    age_ss.append(kmeans.inertia_)

# Visualize the elbow point
plt.figure(figsize=(12, 6))
plt.plot(range(1, 8), age_ss, marker='o', linestyle='--')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Within-Cluster Sum of Squares (Inertia)')
plt.title('Elbow Method for Optimal K')
plt.show()

# Fit KMeans with the chosen K
kmeans_final = KMeans(n_clusters=3, init='k-means++')
kmeans_final.fit(X)

# Get cluster labels
data['Cluster'] = kmeans_final.labels_

# Visualize the clusters (scatter plot)
plt.figure(figsize=(10, 6))
plt.scatter(X['Annual Income (k$)'], X['Spending Score (1-100)'], c=data['Cluster'], cmap='rainbow')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Customer Segmentation')
plt.show()
