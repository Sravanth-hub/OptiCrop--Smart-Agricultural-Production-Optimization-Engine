from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("models/model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/findyourcrop")
def findyourcrop():
    return render_template("predict.html")

@app.route("/predict", methods=["POST"])
def predict():

    values = [
        float(request.form["nitrogen"]),
        float(request.form["phosphorous"]),
        float(request.form["potassium"]),
        float(request.form["temperature"]),
        float(request.form["humidity"]),
        float(request.form["ph"]),
        float(request.form["rainfall"])
    ]

    prediction = model.predict([values])

    return render_template(
        "predict.html",
        prediction=prediction[0]
    )

if __name__ == "__main__":
    app.run(debug=True)