from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
import shap

app = Flask(__name__)

# --- Load artifacts ---
model = joblib.load("model/model_binary_nhanes.pkl")
scaler = joblib.load("model/scaler_nhanes_binary.pkl")
features = joblib.load("model/features_nhanes_binary.pkl")
race_map = joblib.load("model/race_map.pkl")

HIGH_CONFIDENCE_CUTOFF = 0.7

# SHAP explainer — built once at startup using the trained XGBoost step from the pipeline
xgb_step = model.named_steps["clf"] if hasattr(model, "named_steps") else model
explainer = shap.TreeExplainer(xgb_step)


def get_shap_explanation(X_scaled_row):
    """
    Returns top contributing features (name, shap_value) for a single row,
    sorted by absolute impact, largest first.
    """
    shap_values = explainer.shap_values(X_scaled_row)
    # shap_values shape: (1, n_features) for binary XGBoost
    values = shap_values[0] if isinstance(shap_values, np.ndarray) and shap_values.ndim == 2 else shap_values

    pairs = list(zip(features, values))
    pairs.sort(key=lambda p: abs(p[1]), reverse=True)
    top = pairs[:5]  # top 5 contributing features

    return [
        {
            "feature": name,
            "value": float(val),
            "direction": "positive" if val > 0 else "negative",
            "magnitude_pct": min(100, abs(float(val)) * 100)  # rough visual scale, not a true %
        }
        for name, val in top
    ]


def predict_risk_tier(person: dict):
    """
    Precedence order:
      1. Diabetic-range labs (HbA1c >= 6.5% or glucose >= 126) -> always HIGH
      2. Model confidently High (prob_high >= 0.7) -> HIGH
      3. Prediabetes-range labs (HbA1c 5.7-6.4% or glucose 100-125) -> MODERATE
      4. Otherwise -> model's own Low/High call

    SHAP explanations are only computed for steps 2 and 4 (model-driven decisions).
    Steps 1 and 3 are explained by the triggering lab values themselves.
    """
    hba1c = person["hba1c"]
    glucose = person["fasting_glucose_mgdl"]

    row = {k: person[k] for k in person if k != "race_ethnicity"}
    for code in race_map:
        row[f"race_{code}"] = 1 if person["race_ethnicity"] == code else 0

    X = pd.DataFrame([row])
    X = X.reindex(columns=features, fill_value=0)
    X_scaled = scaler.transform(X.values)
    prob_high = model.predict_proba(X_scaled)[0, 1]

    # Step 1: diabetic-range labs always win — rule-explained, no SHAP
    is_diabetic_hba1c = hba1c >= 6.5
    is_diabetic_glucose = glucose >= 126

    if is_diabetic_hba1c or is_diabetic_glucose:
        trigger = []
        if is_diabetic_hba1c:
            trigger.append(f"HbA1c {hba1c}% is in the diabetic range (>=6.5%)")
        if is_diabetic_glucose:
            trigger.append(f"Fasting glucose {glucose} mg/dL is in the diabetic range (>=126 mg/dL)")
        return "High", prob_high, trigger, None

    # Step 2: confidently High model call — SHAP-explained
    if prob_high >= HIGH_CONFIDENCE_CUTOFF:
        shap_explanation = get_shap_explanation(X_scaled)
        return "High", prob_high, None, shap_explanation

    # Step 3: prediabetes-range labs — rule-explained, no SHAP
    is_moderate_hba1c = 5.7 <= hba1c <= 6.4
    is_moderate_glucose = 100 <= glucose <= 125

    if is_moderate_hba1c or is_moderate_glucose:
        trigger = []
        if is_moderate_hba1c:
            trigger.append(f"HbA1c {hba1c}% in prediabetes range (5.7-6.4%)")
        if is_moderate_glucose:
            trigger.append(f"Fasting glucose {glucose} mg/dL in prediabetes range (100-125 mg/dL)")
        return "Moderate", prob_high, trigger, None

    # Step 4: model's own Low/High call — SHAP-explained
    tier = "High" if prob_high >= 0.5 else "Low"
    shap_explanation = get_shap_explanation(X_scaled)
    return tier, prob_high, None, shap_explanation


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        person = {
            "patient_name": request.form["patient_name"].strip(),
            "patient_address": request.form["patient_address"].strip(),
            "age": float(request.form["age"]),
            "sex": int(request.form["sex"]),
            "bmi": float(request.form["bmi"]),
            "waist_circumference_cm": float(request.form["waist_circumference_cm"]),
            "hba1c": float(request.form["hba1c"]),
            "fasting_glucose_mgdl": float(request.form["fasting_glucose_mgdl"]),
            "race_ethnicity": int(request.form["race_ethnicity"]),
        }

        tier, prob_high, reason, shap_explanation = predict_risk_tier(person)
        prob_percent = float(round(float(prob_high) * 100, 1))
        person_display = {
            "Patient Name": person["patient_name"],
            "Patient Address": person["patient_address"],
            "Age": float(person["age"]),
            "Sex": "Male" if person["sex"] == 1 else "Female",
            "BMI": float(person["bmi"]),
            "Waist Circumference": f'{person["waist_circumference_cm"]} cm',
            "HbA1c": f'{person["hba1c"]}%',
            "Fasting Glucose": f'{person["fasting_glucose_mgdl"]} mg/dL',
            "Race/Ethnicity": race_map.get(
                person["race_ethnicity"],
                race_map.get(str(person["race_ethnicity"]), person["race_ethnicity"]),
            ),
        }

        return render_template(
            "result.html",
            tier=tier,
            prob_high=prob_percent,
            reason=reason,
            shap_explanation=shap_explanation,
            person=person_display,
            race_map=race_map,
        )

    latest_prob = request.args.get("latest")
    return render_template("index.html", race_map=race_map, latest_prob=latest_prob)


@app.route("/records")
def records():
    return render_template("records.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/about-me")
def about_me():
    return render_template("about_me.html")


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
