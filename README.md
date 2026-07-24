# NHANES Diabetes Risk Prediction Tool

A clinical risk-tiering tool that classifies a person's diabetes risk as **Low**, **Moderate**, or **High**, built on an original dataset constructed from raw NHANES 2017–2020 cycle data.

## Overview

This tool combines a trained machine learning model with explicit clinical thresholds to produce a three-tier risk classification. It does not aim to diagnose diabetes it aims to flag risk level using the same lab values clinicians already rely on (HbA1c, fasting glucose), combined with demographic and body-composition factors (age, sex, BMI, waist circumference, race/ethnicity).

## Dataset

The dataset was built by merging five raw NHANES `.XPT` files (`DEMO`, `BMX`, `P_GHB`, `GLU`, `DIQ`) from the 2017–2020 cycle, joined on `SEQN`. No pre-merged public version of this NHANES-diabetes combination exists; the merge itself is original work.

- **Adults-only filter applied** (under-18 respondents excluded)
- **Final adult dataset:** 2,532 rows
- **Risk tier distribution (self-reported):** Low 2,038 (80.5%) / High 414 (16.4%) / Moderate 80 (3.2%)
- **Features used:** age, sex, BMI, waist circumference, HbA1c, fasting glucose, race/ethnicity (one-hot encoded)

The original label (`diabetes_risk`) is derived from `DIQ010`  a self-reported answer to "has a doctor ever told you that you have diabetes?" This is an important limitation discussed below.

## Modeling Approach

Three models (Logistic Regression, Random Forest, XGBoost) were compared on a 3-class Low/Moderate/High target using a leak-free pipeline (SMOTE applied inside cross-validation folds, never before). XGBoost won on macro F1 (0.561) and Cohen's Kappa (0.510), but **Moderate recall was weak across all three models** (12.5%–37.5%) due to severe data scarcity — only 80 real Moderate examples in the entire dataset.

### Why a binary model was chosen for the core classifier

Rather than force a model to learn a class it only has 80 examples of, the final design trains XGBoost as a **binary classifier on Low vs. High only**, with Moderate rows held out entirely from training. This binary model performs strongly:

- **ROC-AUC: 0.935**
- **Cohen's Kappa: 0.670**
- **Low recall: 94.1%** | **High recall: 73.5%**

### Hybrid logic for the Moderate tier

Moderate risk is not predicted by the model directly. Instead, it is assigned through a layered decision process, applied in this order:

1. **Diabetic-range labs override everything.** If HbA1c ≥ 6.5% or fasting glucose ≥ 126 mg/dL, the result is **High**, regardless of what the model or self-report label says. This catches cases the model might otherwise miss due to rare or unseen feature combinations (e.g. isolated high HbA1c with normal glucose).
2. **Confidently High model predictions win next.** If the model's probability of High risk is ≥ 0.7, the result is **High**. This ensures a person with multiple strong combined risk factors (age, BMI, waist circumference) is not pulled into Moderate just because their labs happen to sit at the edge of the prediabetes range.
3. **Prediabetes-range labs trigger Moderate.** If HbA1c is 5.7–6.4% or fasting glucose is 100–125 mg/dL (and steps 1–2 did not already resolve the case), the result is **Moderate** — matching the standard clinical definition of prediabetes.
4. **Otherwise, the model's own Low/High call is used.**

This design was chosen deliberately after testing showed that no probability threshold derived purely from the binary model's output could cleanly isolate Moderate cases — held-out Moderate respondents' predicted probabilities span almost the entire 0–1 range, because prediabetic profiles genuinely overlap with both Low and High profiles. Using the established clinical thresholds for prediabetes and diabetes, rather than a statistically tuned probability band, produces a more defensible and explainable system.

## Honest Performance: The Moderate Tier

This is the system's known weak point, and it is documented here rather than hidden.

Evaluated against the full held-out test set (491 Low/High rows) and the held-out Moderate set (80 rows):

| True label | Correctly classified | Reclassified | Notes |
|---|---|---|---|
| **High** (n=83) | 66 High (79.5%) | 16 → Moderate, 1 → Low | Most misses are model uncertainty combined with borderline labs |
| **Low** (n=408) | 140 Low (34.3%) | 239 → Moderate, 29 → High | Self-report label does not capture undiagnosed prediabetes/diabetes; many "rescues" reflect real lab values the self-report missed |
| **Moderate** (n=80) | 46 Moderate (57.5%) | 26 → High, 8 → Low | Limited by only 80 real examples available for any kind of validation |

**Why so many "Low" people get reclassified:** the original label reflects whether someone has been *told* they have diabetes, not their current lab values. A large share of self-reported "Low" individuals have HbA1c or glucose readings in the prediabetic or diabetic range — meaning the hybrid system is surfacing undiagnosed risk the self-report label was never designed to capture. This is treated as a feature of the design (the explicit goal of a risk tool), not a flaw, but it does mean tier counts will differ substantially from the original self-reported distribution.

**Why the Moderate tier still under-catches by about 42%:** with only 80 real Moderate examples in the entire dataset, there is a hard ceiling on how reliably any model — or any rule-based threshold — can validate performance on this class. This is a data scarcity limitation, not a flaw in the logic, and is unlikely to improve without a meaningfully larger prediabetic sample.

## Project Structure

```
diabetes-system/
│
├── app.py              # Flask app + hybrid prediction logic
├── README.md
│
├── model/
│   ├── model_binary_nhanes.pkl     # Trained XGBoost pipeline (SMOTE + classifier)
│   ├── scaler_nhanes_binary.pkl    # Fitted MinMaxScaler
│   ├── features_nhanes_binary.pkl  # Ordered feature column list
│   ├── race_map.pkl                # Race code -> human-readable label
│   └── shap_values_binary.pkl      # SHAP reference values
│
├── templates/
│   ├── index.html
│   └── result.html
│
└── static/
    ├── style.css
    └── gauge.js
```

## Running Locally

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate        # Windows
   ```
2. Install dependencies:
   ```bash
   pip install flask joblib pandas numpy scikit-learn xgboost imbalanced-learn shap
   ```
3. Run the app:
   ```bash
   python app.py
   ```
4. Open `http://127.0.0.1:5000` in your browser.

## Known Limitations

- **Moderate-class reliability** is constrained by data scarcity (80 examples), as detailed above.
- **Self-reported labels** mean some "Low"/"High" ground truth does not reflect current lab status, only prior diagnosis history.
- **scikit-learn version sensitivity**: artifacts were trained under scikit-learn 1.6.1; loading under a different version may produce `InconsistentVersionWarning` messages. Results should be spot-checked if the environment's sklearn version changes.
- **SHAP explainability** is implemented for the binary model's predictions only; Moderate/High classifications triggered by the clinical rule (rather than the model) are explained by the lab values that triggered the rule, not by SHAP.
