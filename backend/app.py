from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import shap

app = Flask(__name__)
CORS(app)  # allow frontend (different domain) to call the API

# --- Load artifacts ---
model = joblib.load("model/model_binary_nhanes.pkl")
scaler = joblib.load("model/scaler_nhanes_binary.pkl")
features = joblib.load("model/features_nhanes_binary.pkl")
race_map = joblib.load("model/race_map.pkl")

HIGH_CONFIDENCE_CUTOFF = 0.7

xgb_step = model.named_steps["clf"] if hasattr(model, "named_steps") else model
explainer = shap.TreeExplainer(xgb_step)

def get_shap_explanation(X_scaled_row):
    shap_values = explainer.shap_values(X_scaled_row)
    values = shap_values[0] if isinstance(shap_values, np.ndarray) and shap_values.ndim == 2 else shap_values
    pairs = list(zip(features, values))
    pairs.sort(key=lambda p: abs(p[1]), reverse=True)
    top = pairs[:5]
    return [
        {
            "feature": name,
            "value": float(val),
            "direction": "positive" if val > 0 else "negative",
            "magnitude_pct": min(100, abs(float(val)) * 100)
        }
        for name, val in top
    ]

def predict_risk_tier(person: dict):
    hba1c = person["hba1c"]
    glucose = person["fasting_glucose_mgdl"]

    row = {k: person[k] for k in person if k != "race_ethnicity"}
    for code in race_map:
        row[f"race_{code}"] = 1 if person["race_ethnicity"] == code else 0

    X = pd.DataFrame([row])
    X = X.reindex(columns=features, fill_value=0)
    X_scaled = scaler.transform(X.values)
    prob_high = model.predict_proba(X_scaled)[0, 1]

    is_diabetic_hba1c = hba1c >= 6.5
    is_diabetic_glucose = glucose >= 126

    if is_diabetic_hba1c or is_diabetic_glucose:
        trigger = []
        if is_diabetic_hba1c:
            trigger.append(f"HbA1c {hba1c}% is in the diabetic range (>=6.5%)")
        if is_diabetic_glucose:
            trigger.append(f"Fasting glucose {glucose} mg/dL is in the diabetic range (>=126 mg/dL)")
        return "High", prob_high, trigger, None

    if prob_high >= HIGH_CONFIDENCE_CUTOFF:
        shap_explanation = get_shap_explanation(X_scaled)
        return "High", prob_high, None, shap_explanation

    is_moderate_hba1c = 5.7 <= hba1c <= 6.4
    is_moderate_glucose = 100 <= glucose <= 125

    if is_moderate_hba1c or is_moderate_glucose:
        trigger = []
        if is_moderate_hba1c:
            trigger.append(f"HbA1c {hba1c}% in prediabetes range (5.7-6.4%)")
        if is_moderate_glucose:
            trigger.append(f"Fasting glucose {glucose} mg/dL in prediabetes range (100-125 mg/dL)")
        return "Moderate", prob_high, trigger, None

    tier = "High" if prob_high >= 0.5 else "Low"
    shap_explanation = get_shap_explanation(X_scaled)
    return tier, prob_high, None, shap_explanation

@app.route("/health")
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body"}), 400

    try:
        person = {
            "patient_name": str(data.get("patient_name", "")).strip(),
            "patient_address": str(data.get("patient_address", "")).strip(),
            "age": float(data["age"]),
            "sex": int(data["sex"]),
            "bmi": float(data["bmi"]),
            "waist_circumference_cm": float(data["waist_circumference_cm"]),
            "hba1c": float(data["hba1c"]),
            "fasting_glucose_mgdl": float(data["fasting_glucose_mgdl"]),
            "race_ethnicity": int(data["race_ethnicity"]),
        }
    except (KeyError, ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid input: {str(e)}"}), 400

    tier, prob_high, reason, shap_explanation = predict_risk_tier(person)

    return jsonify({
        "tier": tier,
        "prob_high": float(round(float(prob_high) * 100, 1)),
        "reason": reason,
        "shap_explanation": shap_explanation,
        "person": {
            "Patient Name": person["patient_name"],
            "Patient Address": person["patient_address"],
            "Age": person["age"],
            "Sex": "Male" if person["sex"] == 1 else "Female",
            "BMI": person["bmi"],
            "Waist Circumference": f'{person["waist_circumference_cm"]} cm',
            "HbA1c": f'{person["hba1c"]}%',
            "Fasting Glucose": f'{person["fasting_glucose_mgdl"]} mg/dL',
            "Race/Ethnicity": race_map.get(
                person["race_ethnicity"],
                race_map.get(str(person["race_ethnicity"]), person["race_ethnicity"])
            ),
        }
    })

@app.route("/api/race-map")
def get_race_map():
    return jsonify(race_map)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)