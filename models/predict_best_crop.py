import pickle
import os
import pandas as pd

current_dir = os.path.dirname(__file__)
model_path = os.path.join(current_dir, "model.pkl")

with open(model_path, "rb") as file:
    model = pickle.load(file)

print("===== Crop Prediction =====")

N = float(input("Nitrogen: "))
P = float(input("Phosphorous: "))
K = float(input("Potassium: "))
temperature = float(input("Temperature: "))
humidity = float(input("Humidity: "))
ph = float(input("pH: "))
rainfall = float(input("Rainfall: "))

input_data = pd.DataFrame({
    "N":[N],
    "P":[P],
    "K":[K],
    "temperature":[temperature],
    "humidity":[humidity],
    "ph":[ph],
    "rainfall":[rainfall]
})

prediction = model.predict(input_data)

print("\n🌱 Recommended Crop:", prediction[0])