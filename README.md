# Customer Churn Classifier

> End-to-end ML pipeline predicting telecom customer churn using XGBoost — with SHAP explainability, 5-fold cross-validation, and a live risk prediction interface.

**Live Demo → [churn-risk-engine.streamlit.app](https://churn-risk-engine.streamlit.app)**

<img width="2726" height="1598" alt="image" src="https://github.com/user-attachments/assets/828afb33-53a9-4a32-b902-19216a896ac1" />


---

## What It Does

Telecom companies lose customers every month to competitors. Acquiring a new customer costs 5-7x more than retaining an existing one. This system predicts which customers are at risk of churning — before they leave — so retention teams can intervene in time.

Input a customer's account details and get:
- Churn probability score (0–100%)
- Risk level: High / Medium / Low
- Retention recommendation tailored to their risk profile
- SHAP-based explanation of which features drove the prediction

---

## Model Performance

| Metric | Value |
|---|---|
| ROC-AUC | 0.8114 |
| 5-Fold CV AUC | 0.8228 ± 0.004 |
| F1-Score (churn class) | 0.58 |
| Accuracy | 0.78 |
| Precision (churn) | 0.60 |
| Recall (churn) | 0.55 |
| True Positives | 206 |
| False Negatives | 168 |

---

## Key Findings

**1. Contract type is the strongest business lever**
Month-to-month customers churn at 42.7% vs 11.3% for one-year and just 2.8% for two-year contracts. The single most impactful retention action is converting customers to longer contracts.

**2. Tenure is the #1 predictive feature (SHAP)**
Customers in their first 12 months churn at 47.7%. By 49–72 months, that drops to 9.5%. New customer onboarding is the highest-leverage intervention point.

**3. Fiber optic customers churn more despite paying more**
Internet service type is a top-10 churn driver. Fiber customers pay higher monthly charges but churn at a higher rate — suggesting a service quality or value perception issue.

**4. Engineered feature ranked #3 in importance**
`ChargePerService` (monthly charge divided by number of active services) was created during feature engineering and ranked 3rd in SHAP importance — outranking raw features like MonthlyCharges.

---

## ML Pipeline

### 1. Data Cleaning
- Fixed `TotalCharges` column stored as string — converted to float
- Dropped 11 rows with null TotalCharges (new customers with zero tenure)
- Final dataset: 7,032 customers, 20 features

### 2. Feature Engineering
```python
df['AvgMonthlyCharge'] = df['TotalCharges'] / (df['tenure'] + 1)
df['ChargePerService'] = df['MonthlyCharges'] / (num_services + 1)
df['TenureGroup'] = pd.cut(df['tenure'], bins=[0,12,24,48,72], labels=['New','Growing','Mature','Loyal'])
```
Encoded all categorical variables using binary mapping and one-hot encoding. Final feature count: 29.

### 3. Model Comparison
Four models trained and compared on ROC-AUC:

| Model | AUC | F1 |
|---|---|---|
| XGBoost | 0.8114 | 0.58 |
| LightGBM | ~0.81 | ~0.57 |
| Random Forest | ~0.80 | ~0.56 |
| Logistic Regression | ~0.78 | ~0.54 |

XGBoost selected as best performer.

### 4. Evaluation
- 80/20 stratified train/test split (stratified to preserve 73/27 churn ratio)
- 5-fold cross-validation: mean AUC 0.8228, std 0.004
- SHAP TreeExplainer for feature importance and prediction explanation
- Confusion matrix analysis for business impact assessment

### 5. Class Imbalance
Dataset is 73% non-churn / 27% churn. Handled via:
- Stratified splitting to preserve ratio in both train and test sets
- Evaluation using AUC and F1 instead of accuracy
- XGBoost's built-in `scale_pos_weight` parameter available for tuning

---

## Dashboard Pages

**Dashboard** — churn rate by contract type (risk-coded), top SHAP features, churn rate by tenure group.

**Predict** — live customer risk predictor with 18 input features, probability gauge, and retention recommendation.

**Model Info** — confusion matrix heatmap, SHAP feature importance chart, pipeline steps.

**About** — project overview, key findings, tech stack.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data Processing | Python, Pandas, NumPy |
| ML Models | Scikit-learn, XGBoost, LightGBM |
| Explainability | SHAP |
| Visualisation | Plotly |
| Dashboard | Streamlit |
| Deployment | Streamlit Cloud |
| Version Control | Git, GitHub |

---

## Project Structure

```
churn-classifier/
├── data/
│   ├── raw/                        ← Telco churn dataset (Kaggle)
│   └── processed/
│       ├── churn_clean.csv         ← cleaned + encoded dataset
│       ├── feature_importance.csv  ← SHAP values per feature
│       └── model_results.csv       ← all model comparison scores
├── notebooks/
│   ├── 01_eda.ipynb                ← exploratory data analysis
│   ├── 02_cleaning.ipynb           ← cleaning + feature engineering
│   └── 03_modelling.ipynb          ← model training + evaluation
├── outputs/
│   ├── xgb_model.pkl               ← saved XGBoost model
│   └── scaler.pkl                  ← saved StandardScaler
├── src/
│   └── app.py                      ← Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## Run Locally

```bash
git clone https://github.com/whoj1ngmu77/churn-classifier.git
cd churn-classifier
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd src
streamlit run app.py
```

---

## Dataset

**IBM Telco Customer Churn** — 7,043 telecom customers with account details, service subscriptions, and churn label.

→ [Download from Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

**Key columns:**
- `tenure` — months as a customer
- `Contract` — month-to-month, one year, two year
- `MonthlyCharges` — monthly bill amount
- `Churn` — target variable (Yes/No)

---

## Future Improvements

- Hyperparameter tuning with GridSearchCV or Optuna
- SMOTE oversampling for class imbalance
- Threshold optimization for precision/recall tradeoff
- Real-time prediction API with FastAPI
- MLflow experiment tracking
- Model drift monitoring

---

Built by **Gayathri Menon**
