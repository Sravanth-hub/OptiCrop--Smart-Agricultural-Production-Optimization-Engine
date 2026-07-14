import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from evaluate_model import evaluate_model

# Read Dataset
data = pd.read_csv("Dataset/Crop_recommendation.csv")

# Features & Target
X = data.drop("label", axis=1)
y = data["label"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

# Evaluate & Save
evaluate_model(model, X_test, y_test)