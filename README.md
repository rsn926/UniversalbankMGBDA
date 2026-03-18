# 🏦 Universal Bank — Personal Loan Campaign Intelligence

An end-to-end **Streamlit dashboard** for analysing customer data and predicting personal loan acceptance using AI classification models. Built for the Universal Bank marketing team to maximise campaign conversion rates with a constrained budget.

---

## 🚀 Live Demo
Deploy instantly on [Streamlit Community Cloud](https://streamlit.io/cloud):
1. Fork this repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo → select `app.py` → Deploy

---

## 📂 Project Structure

```
universal_bank_app/
├── app.py               ← Main Streamlit application
├── UniversalBank.csv    ← Training dataset (5,000 customer records)
├── test_sample.csv      ← Sample test file for the Predict & Download page
├── requirements.txt     ← Python dependencies
└── README.md            ← You are here
```

---

## 📊 Dashboard Sections

| Section | Description |
|---|---|
| 🏠 Executive Summary | KPIs, overall acceptance rate, top segment comparison |
| 📊 Descriptive Analytics | Customer demographics, income, spending, product adoption |
| 🔍 Diagnostic Analytics | What drives loan acceptance? Correlations, segment analysis |
| 🤖 Predictive Analytics | Model metrics, ROC curve, confusion matrices, feature importance |
| 🎯 Prescriptive Analytics | Target segments, budget allocation, campaign playbook |
| 📤 Predict & Download | Upload new customer CSV → get predictions → download results |

---

## 🤖 Models Implemented

- **Decision Tree** — Interpretable, fast, handles class imbalance via `class_weight='balanced'`
- **Random Forest** — Ensemble of 150 trees, robust against overfitting
- **Gradient Boosting** — Best overall AUC-ROC, recommended for customer scoring

All models evaluated on:  
`Train Accuracy · Test Accuracy · Precision · Recall · F1 Score · AUC-ROC`

---

## ⚙️ Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/universal-bank-loan-dashboard.git
cd universal-bank-loan-dashboard

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

Python 3.8+ required.

---

## 📋 Dataset — Column Reference

| Column | Description |
|---|---|
| `ID` | Customer ID (dropped during preprocessing) |
| `Age` | Customer age in completed years |
| `Experience` | Years of professional experience |
| `Income` | Annual income ($000) |
| `ZIP Code` | Home address ZIP code (dropped during preprocessing) |
| `Family` | Family size (1–4) |
| `CCAvg` | Avg. monthly credit card spending ($000) |
| `Education` | 1: Undergrad, 2: Graduate, 3: Advanced/Professional |
| `Mortgage` | Mortgage value ($000), 0 if none |
| `Personal Loan` | **Target variable** — 1: Accepted, 0: Did Not Accept |
| `Securities Account` | 1 if customer holds a securities account |
| `CD Account` | 1 if customer holds a certificate of deposit |
| `Online` | 1 if customer uses online banking |
| `CreditCard` | 1 if customer uses a UB-issued credit card |

---

## 📤 Using the Predict & Download Feature

1. Navigate to **📤 Predict & Download** in the sidebar
2. Upload `test_sample.csv` (or your own customer file)
3. The file should have all columns **except** `Personal Loan`
4. Select a model and classification threshold
5. Download the results CSV with predictions and propensity tiers

---

## 📦 Dependencies

```
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
plotly>=5.15.0
```

---

## 🔑 Key Findings

- Only **9.6%** of customers historically accepted a personal loan
- **CD Account holders** accept at ~30% — 3× the baseline
- **Income** is the single strongest predictor across all three models
- Targeting the top propensity decile yields **4–6× more conversions per dollar**

---

*Built with ❤️ using Streamlit, scikit-learn, and Plotly*
