from flask import Flask, jsonify, request, render_template, send_from_directory
from flask_cors import CORS
import joblib
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

model = joblib.load("landslide_model.pkl")


# ---------------- HOME ----------------

@app.route("/")
def home():
    return render_template("index.html")


# ---------------- PAGES ----------------

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route("/map")
def map_page():
    return render_template("map.html")


@app.route("/area")
def area():
    return render_template("area.html")


@app.route("/alerts")
def alerts():
    return render_template("alerts.html")


@app.route("/analytics")
def analytics():
    return render_template("analytics.html")


@app.route("/ai-test")
def ai_test():
    return render_template("ai-test.html")


# ---------------- CSS ----------------

@app.route("/style.css")
def style():
    return send_from_directory(".", "style.css")


# ---------------- AI PREDICTION ----------------

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    input_data = pd.DataFrame([[
        data["rainfall"],
        data["soil_moisture"],
        data["slope"],
        data["elevation"],
        data["historical_events"]
    ]], columns=[
        "rainfall",
        "soil_moisture",
        "slope",
        "elevation",
        "historical_events"
    ])

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1] * 100

    risk_level = "HIGH" if prediction == 1 else "LOW"

    return jsonify({
        "risk": round(probability, 2),
        "risk_level": risk_level
    })


# ---------------- START SERVER ----------------

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port
    )