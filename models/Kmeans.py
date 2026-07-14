import pandas as pd
from sklearn.cluster import KMeans

# Read dataset
data = pd.read_csv("Dataset/Crop_recommendation.csv")

# Features (excluding target label)
X = data.drop("label", axis=1)

# Create K-Means model
kmeans = KMeans(
    n_clusters=4,
    init="k-means++",
    max_iter=300,
    n_init=10,
    random_state=42
)

# Train model
data["cluster"] = kmeans.fit_predict(X)

print("=" * 50)
print("K-MEANS CLUSTERING COMPLETED")
print("=" * 50)

print("\nFirst 10 Records:")
print(data[["label", "cluster"]].head(10))

# Display crops in each cluster
for i in range(4):
    print(f"\nCluster {i} Crops:")
    print(data[data["cluster"] == i]["label"].unique())

print("\nCluster Centers:")
print(kmeans.cluster_centers_)