# Credit Risk Probability Model for Alternative Data

## Project Overview

This project develops an end-to-end credit risk scoring system for Bati Bank using alternative transaction data from an eCommerce platform. Since the dataset does not contain a direct default label, a proxy target variable will be engineered using customer behavioral patterns derived from transaction history.

The project covers data processing, feature engineering, model development, experiment tracking, deployment through FastAPI, containerization using Docker, and CI/CD automation.

---

## Credit Scoring Business Understanding

### 1. How does the Basel II Accord's emphasis on risk measurement influence the need for an interpretable and well-documented model?

The Basel II Accord requires financial institutions to measure and manage credit risk using transparent and reliable methodologies. Credit decisions must be explainable and supported by evidence. Therefore, credit scoring models should be interpretable and well documented.

An interpretable model allows analysts, auditors, regulators, and business stakeholders to understand how predictions are generated and which factors influence customer risk. Proper documentation ensures transparency regarding data sources, feature engineering, assumptions, model selection, validation procedures, and monitoring strategies.

This improves regulatory compliance, supports model governance, and builds confidence in automated lending decisions.

### 2. Without a direct default label, why is a proxy variable necessary, and what business risks does proxy-based prediction introduce?

The provided dataset does not include a direct indicator showing whether a customer eventually defaulted on a loan. Since supervised machine learning models require a target variable, a proxy variable must be created to represent customer risk.

A practical approach is to use customer behavior patterns based on Recency, Frequency, and Monetary (RFM) metrics. Customers with low activity, low transaction frequency, and long inactivity periods may be considered higher risk, while highly active customers may be considered lower risk.

However, proxy variables are imperfect substitutes for actual default data. Incorrect proxy design may introduce misclassification errors, resulting in good customers being rejected or risky customers being approved. These errors may affect profitability, fairness, customer satisfaction, and overall lending performance. Therefore, the proxy target must be carefully justified, validated, and monitored.

### 3. What are the key trade-offs between a simple, interpretable model and a high-performance model in a regulated financial context?

Simple models such as Logistic Regression combined with Weight of Evidence (WoE) transformations provide strong interpretability. Their predictions are easier to explain, validate, monitor, and document, making them attractive in regulated financial environments.

More advanced models such as Gradient Boosting often achieve better predictive performance because they can capture complex nonlinear relationships in the data. However, they are more difficult to interpret and explain.

The primary trade-off is between transparency and predictive performance. Interpretable models simplify regulatory compliance and model governance, while complex models may improve predictive accuracy but require additional explainability methods and stronger documentation. Financial institutions must balance these factors when selecting a credit risk model.
