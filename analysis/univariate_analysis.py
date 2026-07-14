import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("Dataset/Crop_recommendation.csv")

# Numerical columns
columns = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

# Plot histogram for each feature
for col in columns:
    plt.figure(figsize=(6,4))
    plt.hist(data[col], bins=20)
    plt.title(f"{col} Distribution")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()