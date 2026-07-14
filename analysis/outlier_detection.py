import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read Dataset
data = pd.read_csv("Dataset/Crop_recommendation.csv")

# Boxplot
plt.figure(figsize=(10,6))

plt.boxplot([
    data["N"],
    data["P"],
    data["K"],
    data["temperature"],
    data["humidity"],
    data["ph"],
    data["rainfall"]
],
tick_labels=["N","P","K","Temp","Humidity","pH","Rainfall"])

plt.title("Boxplot of Agricultural Features")
plt.show()

# IQR Method (Potassium)

Q1 = data["K"].quantile(0.25)
Q3 = data["K"].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

print("Q1 =", Q1)
print("Q3 =", Q3)
print("IQR =", IQR)
print("Lower Bound =", lower)
print("Upper Bound =", upper)

outliers = data[(data["K"] < lower) | (data["K"] > upper)]

print("Number of Outliers =", len(outliers))

# Log Transformation
data["K_log"] = np.log1p(data["K"])

print(data[["K","K_log"]].head())