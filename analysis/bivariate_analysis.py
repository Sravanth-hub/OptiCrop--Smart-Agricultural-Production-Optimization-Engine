import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("Dataset/Crop_recommendation.csv")

plt.figure(figsize=(14,6))

plt.scatter(
    data["label"],
    data["humidity"],
    alpha=0.7
)

plt.xticks(rotation=90)
plt.xlabel("Crop")
plt.ylabel("Humidity")
plt.title("Humidity vs Crop Label")
plt.tight_layout()

plt.show()