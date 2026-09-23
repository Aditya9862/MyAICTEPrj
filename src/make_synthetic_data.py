from pathlib import Path
import numpy as np
import pandas as pd

# This is a fallback teaching dataset, not the original IBM customer file.
rng = np.random.default_rng(42)
n = 7043
customer_ids = [f"C{idx:04d}-{rng.integers(1000, 9999)}" for idx in range(n)]
gender = rng.choice(["Female", "Male"], n)
senior = rng.binomial(1, 0.16, n)
partner = rng.choice(["Yes", "No"], n, p=[0.48, 0.52])
dependents = np.where(np.array(partner) == "Yes", rng.choice(["Yes", "No"], n, p=[0.48, 0.52]), rng.choice(["Yes", "No"], n, p=[0.12, 0.88]))
tenure = np.clip(np.rint(rng.gamma(1.7, 19, n)), 0, 72).astype(int)
contract = []
for months in tenure:
    if months <= 4:
        contract.append(rng.choice(["Month-to-month", "One year", "Two year"], p=[.88, .08, .04]))
    else:
        contract.append(rng.choice(["Month-to-month", "One year", "Two year"], p=[.55, .25, .20]))
contract = np.array(contract)
internet = rng.choice(["DSL", "Fiber optic", "No"], n, p=[.34, .44, .22])
phone = rng.choice(["Yes", "No"], n, p=[.90, .10])

def service_flag(has_internet, p_yes=.45):
    return np.where(has_internet, rng.choice(["Yes", "No"], n, p=[p_yes, 1-p_yes]), "No internet service")

multiple_lines = np.where(np.array(phone) == "Yes", rng.choice(["Yes", "No"], n, p=[.48, .52]), "No phone service")
online_security = service_flag(internet != "No", .29)
online_backup = service_flag(internet != "No", .44)
device_protection = service_flag(internet != "No", .44)
tech_support = service_flag(internet != "No", .30)
streaming_tv = service_flag(internet != "No", .49)
streaming_movies = service_flag(internet != "No", .49)
paperless = rng.choice(["Yes", "No"], n, p=[.59, .41])
payment = rng.choice(["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], n, p=[.34, .23, .22, .21])
monthly = 18 + (internet != "No") * 27 + (internet == "Fiber optic") * 23 + (phone == "Yes") * 20 + (multiple_lines == "Yes") * 8
for flag in [online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies]:
    monthly += (flag == "Yes") * 7
monthly = np.round(np.maximum(18.25, monthly + rng.normal(0, 8, n)), 2)
total = np.round(monthly * tenure + rng.normal(0, 17, n), 2)
total = np.maximum(0, total)

# Month-to-month, short tenure, electronic checks, higher bills and missing support raise the synthetic churn odds.
logit = (-2.70 + (contract == "Month-to-month") * 1.05 + (tenure < 12) * .85 + (tenure < 4) * .35
         + (payment == "Electronic check") * .45 + (monthly > 80) * .40 + (tech_support == "No") * .30
         + (online_security == "No") * .25 + senior * .15 - (contract == "Two year") * .85)
prob = 1 / (1 + np.exp(-logit))
churn = np.where(rng.random(n) < prob, "Yes", "No")

frame = pd.DataFrame({
    "customerID": customer_ids, "gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
    "tenure": tenure, "PhoneService": phone, "MultipleLines": multiple_lines, "InternetService": internet,
    "OnlineSecurity": online_security, "OnlineBackup": online_backup, "DeviceProtection": device_protection,
    "TechSupport": tech_support, "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies,
    "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment,
    "MonthlyCharges": monthly, "TotalCharges": total, "Churn": churn
})
# Like the source file, zero-month customers have blank TotalCharges; we'll clean this explicitly later.
frame.loc[frame["tenure"] == 0, "TotalCharges"] = np.nan
out = Path(__file__).resolve().parents[1] / "data" / "raw" / "Telco-Customer-Churn.csv"
out.parent.mkdir(parents=True, exist_ok=True)
frame.to_csv(out, index=False)
print(f"Wrote {len(frame):,} synthetic rows to {out}")

