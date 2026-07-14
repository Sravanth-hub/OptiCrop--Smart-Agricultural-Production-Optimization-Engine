import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("Dataset/Crop_recommendation.csv")

# -------------------------------
# Crop Count Plot
# -------------------------------
plt.figure(figsize=(12,6))

data["label"].value_counts().plot(kind="bar")

plt.title("Crop Distribution")
plt.xlabel("Crop")
plt.ylabel("Count")
plt.xticks(rotation=90)

plt.tight_layout()
plt.show()

# -------------------------------
# Descriptive Analysis
# -------------------------------
print("\nDescriptive Analysis\n")
print(data.describe())