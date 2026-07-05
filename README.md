# LoanWise — Loan Prediction Web Application

A complete, self-contained web app that predicts whether a loan application
is likely to be **Approved** or **Rejected**, using a pre-trained
Logistic Regression model served through a Flask backend.

---

## Features

- Modern, responsive banking-themed UI (gradient hero, cards, animations)
- Full loan application form covering 12 input features
- Client-side (JavaScript) **and** server-side (Flask) validation
- Loads a pre-trained `model.pkl` — **never retrains on request**
- Clear result page with approval/rejection message and model confidence
- Reset Form / Predict Again buttons
- Graceful error handling (missing model, bad input, prediction failure)
- About & Contact sections, sticky navbar, footer

---

## Folder Structure

```
LoanPrediction/
│
├── app.py                     # Flask backend
├── generate_and_train.py      # One-time script: builds dataset + trains model.pkl
├── model.pkl                  # Pre-trained model bundle (model + scaler + encodings)
├── requirements.txt
│
├── templates/
│   └── index.html             # Form + result page (single template)
│
├── static/
│   ├── css/style.css
│   ├── js/script.js
│   └── favicon.svg
│
├── dataset/
│   └── loan_data.csv          # Synthetic training data (for reference/reproducibility)
│
└── README.md
```

---

## Installation

1. **Clone / unzip** the project folder.
2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Required Packages

- Flask
- NumPy
- Pandas
- scikit-learn
- pickle (part of the Python standard library)

---

## Running the Application

```bash
python app.py
```

Then open **http://127.0.0.1:5000/** in your browser.

> `model.pkl` is already included and ready to use. You do **not** need to
> re-run `generate_and_train.py` unless you want to regenerate the dataset
> or retrain the model.

### (Optional) Regenerating the model

```bash
python generate_and_train.py
```

This recreates `dataset/loan_data.csv` and overwrites `model.pkl` with a
freshly trained model — useful only if you want to experiment with the
training data or algorithm.

---

## Input Features

| Field              | Type      | Notes                              |
|--------------------|-----------|-------------------------------------|
| Age                | Number    | 18–100                              |
| Gender             | Radio     | Male / Female                       |
| Annual Income      | Number    | > 0                                  |
| Employment Status  | Dropdown  | Salaried / Self-Employed / Unemployed |
| Education Level    | Dropdown  | Graduate / Not Graduate             |
| Marital Status     | Radio     | Single / Married                    |
| Dependents         | Number    | 0–10                                 |
| Loan Amount        | Number    | > 0                                  |
| Loan Term          | Dropdown  | 12–360 months                       |
| Credit Score       | Number    | 300–900                             |
| Existing Loan      | Radio     | Yes / No                            |
| Property Area      | Dropdown  | Urban / Semiurban / Rural            |

---

## Screenshots

_Add screenshots of the home page, form, and result page here after running
the app locally._

---

## Future Improvements

- Swap the synthetic dataset for a real, licensed loan-approval dataset
- Add user authentication and saved application history
- Add a REST API endpoint (JSON in/out) alongside the HTML form
- Add model explainability (e.g., feature-importance breakdown per prediction)
- Deploy to a cloud platform (Render, Railway, Azure, etc.) with HTTPS

---

## Disclaimer

This project uses a **synthetic** dataset and is intended for educational
and demonstration purposes only. It should not be used to make real
financial or lending decisions.
