# Waze User Churn Analysis & Machine Learning Pipeline

## Project Overview
This portfolio project implements an end-to-end machine learning pipeline to predict user churn for Waze. By leveraging exploratory data analysis (EDA), advanced feature engineering, and ensemble classification models (Random Forest and XGBoost), this project identifies key behavioral drivers of user attrition to assist business leadership in optimizing retention strategies.

*Note: This project was completed as part of the Google Advanced Data Analytics Professional Certificate, which contributed academic transfer credit toward my Data Science degree at University of Maryland Global Campus (UMGC).*

## Business Problem & Objective
Waze relies on active user engagement to maintain crowd-sourced traffic and mapping accuracy. User churn directly impacts the quality of the platform's community-driven data. 
* **The Goal:** Build and evaluate robust classification models to predict whether a user will churn or remain active.
* **Evaluation Focus:** Because failing to catch a churning user (False Negative) carries a higher strategic cost than targeting a loyal user (False Positive), model performance was optimized around **Recall** alongside Precision and F1-score.

## Dataset & Features
The dataset contains 14,999 user observations across 13 behavioral metrics, including app sessions, total drives, kilometers driven, driving days, and device type. 

**Engineered Features Include:**
* `km_per_driving_day`: Average kilometers driven per active driving day.
* `professional_driver`: Binary flag separating high-frequency professional drivers from casual users.
* `total_sessions_per_day`: Average daily sessions since onboarding.
* `percent_sessions_in_month`: Proportion of total user sessions logged in the most recent month.

## Methodology
1. **Data Cleaning:** Handled missing target labels and imputed infinite values resulting from zero-division in driving ratios.
2. **Data Splitting:** Partitioned data into a rigorous 60% Training, 20% Validation, and 20% Test split using stratification to preserve class balance (~18% churn rate).
3. **Model Training & Tuning:** Tuned **Random Forest** and **XGBoost** classifiers using 4-fold cross-validation (`GridSearchCV`).
4. **Evaluation:** Evaluated models using accuracy, precision, recall, and F1 metrics, identifying **XGBoost** as the champion model.

## Key Findings & Results
* **Champion Model:** The tuned XGBoost classifier successfully captured core nonlinear relationships in user behavior.
* **Top Churn Drivers:** Engineered features (such as `total_sessions_per_day` and `km_per_driving_day`) dominated the top predictive feature importances, validating the power of domain-specific feature engineering.
* **Recommendation:** Waze should deploy proactive in-app retention campaigns targeting users whose daily session frequency and driving days drop below critical thresholds, rather than relying solely on total mileage.

## Tools & Technologies
* **Language:** Python 3.x
* **Libraries:** `pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`