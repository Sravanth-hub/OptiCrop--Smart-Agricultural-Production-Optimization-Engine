import pandas as pd

# Read Dataset
data = pd.read_csv("Dataset/Crop_recommendation.csv")

# Check Null Values
print("Null Values:")
print(data.isnull().sum())