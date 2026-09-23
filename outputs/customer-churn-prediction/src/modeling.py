from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
CLEAN_PATH = ROOT / "data" / "cleaned" / "telco_churn_cleaned.csv"
MODEL_PATH = ROOT / "src" / "churn_model.joblib"
RESULT_PATH = ROOT / "reports" / "model_results.json"

SERVICE_COLUMNS = ["PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
CATEGORICAL_COLUMNS = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod", "tenure_bucket"]
NUMERIC_COLUMNS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges", "service_count", "avg_monthly_spend"]
FEATURES = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS


def prepare_customer_data(df):
    df = df.copy()
    df = df.drop_duplicates()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].replace(r"^\s*$", np.nan, regex=True), errors="coerce")
    df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
    df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce")
    df["SeniorCitizen"] = pd.to_numeric(df["SeniorCitizen"], errors="coerce").fillna(0).astype(int)
    # Blank totals belong to brand-new accounts in this data, so zero is a reasonable starting value.
    df["TotalCharges"] = df["TotalCharges"].fillna(0)
    df["tenure_bucket"] = pd.cut(df["tenure"], bins=[-1, 12, 24, 48, 72], labels=["0-12 months", "13-24 months", "25-48 months", "49-72 months"])
    df["service_count"] = df[SERVICE_COLUMNS].apply(lambda col: col.eq("Yes")).sum(axis=1)
    df["avg_monthly_spend"] = np.where(df["tenure"] > 0, df["TotalCharges"] / df["tenure"], df["MonthlyCharges"])
    return df


def build_preprocessor():
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("numbers", numeric_pipe, NUMERIC_COLUMNS), ("categories", categorical_pipe, CATEGORICAL_COLUMNS)])


def train_models():
    data = prepare_customer_data(pd.read_csv(RAW_PATH))
    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(CLEAN_PATH, index=False)
    X = data[FEATURES]
    y = data["Churn"].map({"No": 0, "Yes": 1})
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=250, min_samples_leaf=3, class_weight="balanced", random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=120, learning_rate=0.06, max_depth=2, random_state=42),
    }
    results, fitted = {}, {}
    for name, estimator in candidates.items():
        pipe = Pipeline([("prep", build_preprocessor()), ("model", estimator)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        score = pipe.predict_proba(X_test)[:, 1]
        results[name] = {
            "accuracy": float(accuracy_score(y_test, pred)), "precision": float(precision_score(y_test, pred)),
            "recall": float(recall_score(y_test, pred)), "f1": float(f1_score(y_test, pred)),
            "roc_auc": float(roc_auc_score(y_test, score)),
            "confusion_matrix": confusion_matrix(y_test, pred).tolist()
        }
        fitted[name] = pipe
    best_name = max(results, key=lambda name: results[name]["roc_auc"])
    joblib.dump({"model": fitted[best_name], "model_name": best_name, "features": FEATURES}, MODEL_PATH)
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps({"best_model": best_name, "metrics": results}, indent=2), encoding="utf-8")
    return data, results, fitted[best_name], best_name, X_test, y_test


if __name__ == "__main__":
    _, results, _, best, _, _ = train_models()
    print("Selected by ROC-AUC:", best)
    print(json.dumps(results, indent=2))
