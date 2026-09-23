# Project report: Customer Churn Prediction using Data Analytics and AI

**IBM SkillsBuild Academic Internship | Data Analytics track**

## Problem statement

When customers stop using a telecom service, the company loses recurring revenue and may miss a chance to fix a problem early. This project looks at account and service details that are associated with churn, then builds a simple model that could help a retention team decide whom to contact first. A prediction is a lead for follow-up, not a reason to treat a customer differently without context.

## Dataset

The project is set up for the public IBM Telco Customer Churn sample, a fictional telecom dataset with customer, service, contract, billing and churn fields. IBM describes the sample in its [Cognos documentation](https://www.ibm.com/docs/en/cognos-analytics/12.0.x?topic=samples-telco-customer-churn). Since downloading the original CSV failed in this workspace, the included 7,043-row file is a reproducible synthetic fallback with the same requested column names. It must not be presented as original IBM customer data. The churn label in the fallback was generated from broad patterns such as contract, tenure, payment method, monthly bill and support services.

## Approach

I kept a raw CSV and made a separate cleaned copy. Blank `TotalCharges` values are read as missing and filled with zero, which fits brand-new accounts that have not accumulated charges. I converted charge and tenure fields to numbers, removed duplicate rows, and created tenure buckets, a service count and average monthly spend. Categorical fields are one-hot encoded inside a scikit-learn pipeline so the same transformation is used when a customer is scored in the dashboard.

The notebook compares Logistic Regression, Random Forest and Gradient Boosting with a stratified 80/20 holdout split. Accuracy, precision, recall, F1, ROC-AUC and confusion matrices are included. The selected model is chosen by ROC-AUC as a starting point, but recall and precision both matter: contacting every customer creates wasted effort, while missing likely churners also has a cost.

## EDA findings

The included synthetic fallback has an overall churn rate of **23.9%**. Churn is **33.4%** for month-to-month contracts, **15.1%** for one-year contracts and **6.2%** for two-year contracts. The first-year tenure group has a **37.7%** churn rate, compared with **17.6%** for customers with 49–72 months of tenure. Electronic-check customers show **28.2%** churn, while automatic credit-card customers show **20.1%**. Customers who churn have an average monthly charge of about **$88.73**, compared with **$82.49** for those who stay.

These are patterns in a synthetic file whose churn label was deliberately shaped by some of these same attributes. They are useful for exercising analysis code, but they are not independent findings about IBM's source data or a real company. The senior-citizen split is nearly identical in this fallback, so I would not make a recommendation based on that feature.

## Model results

The notebook trains and scores all three models and writes the results to `reports/model_results.json`; the saved model is `src/churn_model.joblib`. I could not execute model training in the project-building environment because it blocks loading the downloaded scikit-learn/SciPy native components. I have therefore not filled in scores or claimed a winning model here. After installing the listed requirements, run all notebook cells; copy the printed holdout metrics and selected-model name into this section before submitting. That way the final report uses actual results from the included synthetic file rather than made-up numbers.

## Business recommendations

1. Test a modest renewal offer for month-to-month customers. In the included data their churn rate is 33.4%, but a pilot with a comparison group is needed to see if an offer helps.
2. Add a check-in during the first three months, when the first-year group shows the highest churn. Ask about installation and billing friction before suggesting a discount.
3. Review higher monthly bills alongside the exact services on the account. Customers who churn have higher average monthly charges in this file, but the relationship could reflect service mix as much as price.
4. Make automatic payments easy to set up and explain the options clearly. The electronic-check group has higher churn here, but payment method may be a signal rather than the cause.
5. Make tech support and online security help visible during onboarding and after service issues. Check whether customers use the help and whether that is followed by lower churn.

## What I learned

Cleaning choices matter even in a small dataset: blank total charges have to be interpreted in context instead of dropped automatically. I also learned that a model score does not tell a business what action to take, and high accuracy can hide missed churners when churn is the smaller class. The charts were easier to explain after grouping tenure into a few ranges.

## Future improvements

The biggest gap is the synthetic data. I would replace it with IBM's source CSV or an approved real customer dataset before using the findings in a pitch as evidence. I would also test calibrated probabilities, threshold choices, subgroup performance and a time-based validation split. A cost estimate for retention offers would help connect the predictions to an actual decision.
