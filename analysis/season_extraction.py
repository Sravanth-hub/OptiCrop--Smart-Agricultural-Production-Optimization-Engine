import pandas as pd

# Read Dataset
data = pd.read_csv("Dataset/Crop_recommendation.csv")

# Season Function
def season(temp, rainfall):
    if rainfall > 200:
        return "Rainy"
    elif temp > 30:
        return "Summer"
    else:
        return "Winter"

# Create Season Column
data["Season"] = data.apply(
    lambda x: season(x["temperature"], x["rainfall"]),
    axis=1
)

print(data[["temperature","rainfall","label","Season"]].head(10))

print("\nSeason Counts:")
print(data["Season"].value_counts())