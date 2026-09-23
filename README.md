# Customer Churn Prediction using Data Analytics and AI

**IBM SkillsBuild Academic Internship | Big Data & Business Management / Data Analytics track**

This is a student project about finding patterns linked with telecom customer churn and using those patterns to suggest retention actions. The analysis code is written to use the commonly shared IBM Telco Customer Churn CSV. The CSV currently included here is a synthetic fallback with the same 21 columns, because downloading IBM's file failed in this workspace. Please keep that distinction in your presentation: the included rows are simulated and the model scores are not real-world evidence.

## Folder structure

```text
.
├── data/
│   ├── raw/Telco-Customer-Churn.csv        # Included synthetic fallback (7,043 rows)
│   └── cleaned/telco_churn_cleaned.csv      # Cleaned version for analysis
├── notebooks/01_customer_churn_analysis.ipynb
├── src/
│   ├── make_synthetic_data.py               # Rebuild the fallback data
│   └── modeling.py                          # Cleaning, features, training and metrics
├── dashboard/app.py                         # Streamlit overview and prediction form
├── reports/
│   ├── project_report.md
│   └── model_results.json                   # Written after training
└── requirements.txt
```

## Dataset

IBM's fictional Telco Customer Churn sample is a common educational dataset with account, service, billing and churn fields. IBM describes the sample [on its Cognos documentation page](https://www.ibm.com/docs/en/cognos-analytics/12.0.x?topic=samples-telco-customer-churn). IBM's archived example repository also has the CSV: [Telco-Customer-Churn.csv](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv).

The local raw CSV in this project is synthetic, made with the same requested column names and broad telecom-style patterns. It has 7,043 generated records. To reproduce it, run `python src/make_synthetic_data.py`. If you replace it with IBM's original CSV, save it under the same filename in `data/raw/`; the notebook and training code will then use that file.

## Setup

From the project root, create and activate an environment, then install packages:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run the notebook in Jupyter:

```bash
jupyter notebook
```

Open `notebooks/01_customer_churn_analysis.ipynb` and run the cells from top to bottom. It cleans the data, saves `data/cleaned/telco_churn_cleaned.csv`, creates eight EDA charts, compares models and writes `src/churn_model.joblib` plus `reports/model_results.json`.

Launch the dashboard from the project root:

```bash
streamlit run dashboard/app.py
```

The app reads the cleaned CSV and saved model. If the model file is not there yet, it trains the candidates the first time the app runs. Training needs the packages in `requirements.txt`.

## Tech stack

Python, pandas, NumPy, scikit-learn, Matplotlib, Seaborn, Jupyter, joblib and Streamlit. Gradient Boosting is used as the third baseline so XGBoost is not required.

## A couple of project limits

The synthetic data has designed patterns, so its metrics only check that the code pipeline works; they do not prove a business outcome. The holdout split is a first baseline, not a deployment review. A next version should use the original source data, check fairness and leakage more carefully, tune the decision threshold around retention-team capacity, and test whether the model remains useful over time.
