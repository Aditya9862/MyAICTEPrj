from pathlib import Path
import sys
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from modeling import prepare_customer_data, train_models

DATA_PATH = ROOT / "data" / "cleaned" / "telco_churn_cleaned.csv"
MODEL_PATH = ROOT / "src" / "churn_model.joblib"
RAW_PATH = ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"

st.set_page_config(page_title="Customer Churn | Student Project", page_icon="📡", layout="wide")
st.title("Customer Churn Prediction")
st.caption("A small analytics project using IBM-style Telco customer data. Synthetic fallback data is clearly marked in the README.")

if not DATA_PATH.exists():
    st.info("Cleaned file is missing, so I'm preparing it from the raw CSV now.")
    data = prepare_customer_data(pd.read_csv(RAW_PATH))
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(DATA_PATH, index=False)
else:
    data = pd.read_csv(DATA_PATH)

if not MODEL_PATH.exists():
    st.warning("The model hasn't been saved yet. Training the three baseline models for the prediction form.")
    train_models()

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]

overview, predict = st.tabs(["Overview", "Predict a customer"])
with overview:
    churn_rate = (data["Churn"] == "Yes").mean()
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Customers", f"{len(data):,}")
    col_b.metric("Churn rate", f"{churn_rate:.1%}")
    col_c.metric("Selected model", bundle["model_name"])

    left, right = st.columns(2)
    with left:
        by_contract = data.groupby("Contract", observed=True)["Churn"].apply(lambda values: (values == "Yes").mean()).sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(7, 4))
        by_contract.plot(kind="bar", color="#e8895b", ax=ax)
        ax.set_ylabel("Churn rate")
        ax.set_xlabel("")
        ax.set_title("Churn rate by contract")
        ax.tick_params(axis="x", rotation=15)
        st.pyplot(fig)
    with right:
        by_payment = data.groupby("PaymentMethod", observed=True)["Churn"].apply(lambda values: (values == "Yes").mean()).sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(7, 4))
        by_payment.plot(kind="bar", color="#517b9b", ax=ax)
        ax.set_ylabel("Churn rate")
        ax.set_xlabel("")
        ax.set_title("Churn rate by payment method")
        ax.tick_params(axis="x", rotation=20)
        st.pyplot(fig)

    st.subheader("Patterns to look at")
    st.write("This dataset tends to show more churn among month-to-month customers and customers early in their tenure. These group-level patterns are useful leads, but they don't prove that contract type or tenure causes churn.")
    st.dataframe(by_contract.rename("churn_rate").to_frame().style.format({"churn_rate": "{:.1%}"}), use_container_width=True)

with predict:
    st.write("Enter a profile to get a probability from the saved model. This is a screening score, not a guarantee that someone will leave.")
    with st.form("customer_form"):
        c1, c2, c3 = st.columns(3)
        gender = c1.selectbox("Gender", ["Female", "Male"])
        senior = c2.selectbox("Senior citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
        partner = c3.selectbox("Has partner", ["Yes", "No"])
        dependents = c1.selectbox("Dependents", ["No", "Yes"])
        tenure = c2.slider("Tenure (months)", 0, 72, 12)
        phone = c3.selectbox("Phone service", ["Yes", "No"])
        internet = c1.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
        contract = c2.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        payment = c3.selectbox("Payment method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        monthly = c1.number_input("Monthly charges ($)", 18.0, 120.0, 70.0)
        paperless = c2.selectbox("Paperless billing", ["Yes", "No"])
        multi = c3.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
        service_cols = {}
        for label, choices in [("OnlineSecurity", ["No", "Yes", "No internet service"]), ("OnlineBackup", ["No", "Yes", "No internet service"]), ("DeviceProtection", ["No", "Yes", "No internet service"]), ("TechSupport", ["No", "Yes", "No internet service"]), ("StreamingTV", ["No", "Yes", "No internet service"]), ("StreamingMovies", ["No", "Yes", "No internet service"])]:
            service_cols[label] = st.selectbox(label.replace("([A-Z])", r" \1"), choices, key=label)
        submitted = st.form_submit_button("Estimate churn probability")

    if submitted:
        total_charges = max(0, monthly * tenure)
        customer = pd.DataFrame([{
            "gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents, "tenure": tenure,
            "PhoneService": phone, "MultipleLines": multi, "InternetService": internet,
            **service_cols, "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment,
            "MonthlyCharges": monthly, "TotalCharges": total_charges, "Churn": "No"
        }])
        prepared = prepare_customer_data(customer)
        probability = model.predict_proba(prepared[bundle["features"]])[0, 1]
        st.metric("Estimated churn probability", f"{probability:.1%}")
        st.progress(float(probability))
        st.caption("The estimate comes from a model trained on a synthetic teaching dataset in this version. Use it to explore the workflow, not to make customer decisions.")
