"""
generate_and_train.py
----------------------
Generates a synthetic loan-approval dataset and trains a Logistic Regression
model on it, then saves the trained model as model.pkl using pickle.

This script is run ONCE, offline, to produce model.pkl.
The Flask app (app.py) only LOADS model.pkl -- it never retrains.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

np.random.seed(42)

N = 3000

age = np.random.randint(21, 65, N)
gender = np.random.choice(["Male", "Female"], N)
income = np.random.randint(150000, 2000000, N)          # annual income
employment = np.random.choice(["Salaried", "Self-Employed", "Unemployed"], N, p=[0.55, 0.35, 0.10])
education = np.random.choice(["Graduate", "Not Graduate"], N, p=[0.7, 0.3])
marital = np.random.choice(["Single", "Married"], N)
dependents = np.random.randint(0, 5, N)
loan_amount = np.random.randint(50000, 2000000, N)
loan_term = np.random.choice([12, 24, 36, 60, 120, 180, 240, 360], N)
credit_score = np.random.randint(300, 900, N)
existing_loan = np.random.choice(["Yes", "No"], N, p=[0.3, 0.7])
property_area = np.random.choice(["Urban", "Semiurban", "Rural"], N)

# Build a "logical" approval score so the model learns sensible patterns
score = (
    (credit_score - 300) / 600 * 3.5
    + (income / 2000000) * 2.0
    - (loan_amount / 2000000) * 2.5
    - (dependents * 0.15)
    + (education == "Graduate") * 0.5
    + (employment == "Salaried") * 0.4
    - (employment == "Unemployed") * 1.5
    - (existing_loan == "Yes") * 0.6
    + np.random.normal(0, 0.6, N)
)
approved = (score > 1.2).astype(int)

df = pd.DataFrame({
    "Age": age,
    "Gender": gender,
    "Annual_Income": income,
    "Employment_Status": employment,
    "Education_Level": education,
    "Marital_Status": marital,
    "Dependents": dependents,
    "Loan_Amount": loan_amount,
    "Loan_Term": loan_term,
    "Credit_Score": credit_score,
    "Existing_Loan": existing_loan,
    "Property_Area": property_area,
    "Loan_Status": approved
})

df.to_csv("dataset/loan_data.csv", index=False)

# ---- Encode categorical columns (must match encoding used in app.py) ----
gender_map = {"Male": 0, "Female": 1}
employment_map = {"Salaried": 0, "Self-Employed": 1, "Unemployed": 2}
education_map = {"Graduate": 0, "Not Graduate": 1}
marital_map = {"Single": 0, "Married": 1}
existing_loan_map = {"No": 0, "Yes": 1}
property_map = {"Rural": 0, "Semiurban": 1, "Urban": 2}

df_enc = df.copy()
df_enc["Gender"] = df_enc["Gender"].map(gender_map)
df_enc["Employment_Status"] = df_enc["Employment_Status"].map(employment_map)
df_enc["Education_Level"] = df_enc["Education_Level"].map(education_map)
df_enc["Marital_Status"] = df_enc["Marital_Status"].map(marital_map)
df_enc["Existing_Loan"] = df_enc["Existing_Loan"].map(existing_loan_map)
df_enc["Property_Area"] = df_enc["Property_Area"].map(property_map)

# EXACT feature order used everywhere (training + prediction)
FEATURE_ORDER = [
    "Age", "Gender", "Annual_Income", "Employment_Status", "Education_Level",
    "Marital_Status", "Dependents", "Loan_Amount", "Loan_Term",
    "Credit_Score", "Existing_Loan", "Property_Area"
]

X = df_enc[FEATURE_ORDER].values
y = df_enc["Loan_Status"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

acc = model.score(X_test_scaled, y_test)
print(f"Model trained. Test accuracy: {acc:.3f}")

# Save model + scaler + feature order + encoding maps together
bundle = {
    "model": model,
    "scaler": scaler,
    "feature_order": FEATURE_ORDER,
    "encodings": {
        "Gender": gender_map,
        "Employment_Status": employment_map,
        "Education_Level": education_map,
        "Marital_Status": marital_map,
        "Existing_Loan": existing_loan_map,
        "Property_Area": property_map,
    }
}

with open("model.pkl", "wb") as f:
    pickle.dump(bundle, f)

print("Saved model.pkl")
