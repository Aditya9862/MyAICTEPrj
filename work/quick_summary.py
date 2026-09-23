from pathlib import Path
import pandas as pd, numpy as np
root=Path.cwd()
df=pd.read_csv(root/'data/raw/Telco-Customer-Churn.csv')
df=df.drop_duplicates()
df['TotalCharges']=pd.to_numeric(df['TotalCharges'].replace(r'^\s*$',np.nan,regex=True),errors='coerce').fillna(0)
df['tenure']=pd.to_numeric(df['tenure'],errors='coerce')
df['MonthlyCharges']=pd.to_numeric(df['MonthlyCharges'],errors='coerce')
df['SeniorCitizen']=pd.to_numeric(df['SeniorCitizen'],errors='coerce').fillna(0).astype(int)
df['tenure_bucket']=pd.cut(df['tenure'],bins=[-1,12,24,48,72],labels=['0-12 months','13-24 months','25-48 months','49-72 months'])
services=['PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
df['service_count']=df[services].apply(lambda c:c.eq('Yes')).sum(axis=1)
df['avg_monthly_spend']=np.where(df.tenure>0,df.TotalCharges/df.tenure,df.MonthlyCharges)
(root/'data/cleaned').mkdir(exist_ok=True)
df.to_csv(root/'data/cleaned/telco_churn_cleaned.csv',index=False)
print('overall churn', (df.Churn=='Yes').mean())
for c in ['Contract','tenure_bucket','InternetService','PaymentMethod','SeniorCitizen','OnlineSecurity','TechSupport']:
 print(c, df.groupby(c,observed=True).Churn.apply(lambda s:(s=='Yes').mean()).sort_values(ascending=False).round(3).to_dict())
print('monthly churn mean',df.groupby('Churn').MonthlyCharges.mean().round(2).to_dict())
print('synthetic data saved',df.shape)
