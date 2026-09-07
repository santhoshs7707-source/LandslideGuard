import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = pd.read_csv("data/landslide_data.csv")

# Input features
X = data[
    [
        "rainfall",
        "soil_moisture",
        "slope",
        "elevation",
        "historical_events"
    ]
]

# Target
y = data["landslide_risk"]

# Create AI model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train the model
model.fit(X, y)
joblib.dump(model,"landslide_model.pkl")
print("AI model saved successfully! ✅")
print("AI model trained successfully! ✅ ")

# Example location data
sample = pd.DataFrame(
    [[100, 65, 30, 1500, 4]],
    columns=[
        "rainfall",
        "soil_moisture",
        "slope",
        "elevation",
        "historical_events"
    ]
)

# Predict risk
prediction = model.predict(sample)[0]
probability = model.predict_proba(sample)[0][1] * 100

print("Predicted landslide risk:", round(probability, 2), "%")

if prediction == 1:
    print("Risk Level: HIGH ⚠️")
else:
    print("Risk Level: LOW ✅")