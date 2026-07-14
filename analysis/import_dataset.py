import pandas as pd

# Read Dataset
data = pd.read_csv("Dataset/Crop_recommendation.csv")

# Dataset Shape
print("Dataset Shape:")
print(data.shape)

# Dataset Information
print("\nDataset Information:")
data.info()

# First 5 Rows
print("\nFirst 5 Rows:")
print(data.head())