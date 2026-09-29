# Waze User Churn Prediction Project
# Machine Learning Models: Random Forest & XGBoost

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import pickle

pd.set_option('display.max_columns', None)

# 1. Load dataset
df = pd.read_csv('waze_dataset.csv')

# 2. Feature Engineering & Data Preparation
df = df.dropna(subset=['label'])

df['km_per_driving_day'] = df['driven_km_drives'] / df['driving_days']
df['km_per_driving_day'] = df['km_per_driving_day'].replace(np.inf, 0)

df['professional_driver'] = np.where((df['drives'] >= 60) & (df['driving_days'] >= 15), 1, 0)
df['total_sessions_per_day'] = df['total_sessions'] / df['n_days_after_onboarding']
df['km_per_hour'] = df['driven_km_drives'] / (df['duration_minutes_drives'] / 60)
df['km_per_drive'] = df['driven_km_drives'] / df['drives']
df['km_per_drive'] = df['km_per_drive'].replace(np.inf, 0)
df['percent_sessions_in_month'] = df['sessions'] / df['total_sessions']

df['label'] = df['label'].map({'retained': 0, 'churned': 1})
df['device'] = df['device'].map({'Android': 0, 'iPhone': 1})

X = df.drop(columns=['ID', 'label'])
y = df['label']

X_tr, X_test, y_tr, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_tr, y_tr, test_size=0.25, stratify=y_tr, random_state=42)

# 3. Model Building & Hyperparameter Tuning
rf = RandomForestClassifier(random_state=42)
cv_params_rf = {
    'max_depth': [None, 10],
    'max_features': [1.0, 'sqrt'],
    'max_samples': [0.7, 1.0],
    'min_samples_leaf': [1, 2],
    'min_samples_split': [2, 4],
    'n_estimators': [100, 200]
}
scoring = ['accuracy', 'precision', 'recall', 'f1']

rf_cv = GridSearchCV(rf, cv_params_rf, scoring=scoring, refit='recall', cv=4, n_jobs=-1)
rf_cv.fit(X_train, y_train)

xgb = XGBClassifier(objective='binary:logistic', random_state=42)
cv_params_xgb = {
    'max_depth': [4, 6, 8],
    'min_child_weight': [1, 3],
    'learning_rate': [0.01, 0.1],
    'n_estimators': [100, 200]
}

xgb_cv = GridSearchCV(xgb, cv_params_xgb, scoring=scoring, refit='recall', cv=4, n_jobs=-1)
xgb_cv.fit(X_train, y_train)

# 4. Model Evaluation
def evaluate_model(name, model, X_v, y_v):
    preds = model.predict(X_v)
    return pd.DataFrame({
        'Model': [name],
        'Precision': [precision_score(y_v, preds)],
        'Recall': [recall_score(y_v, preds)],
        'F1 Score': [f1_score(y_v, preds)],
        'Accuracy': [accuracy_score(y_v, preds)]
    })

val_rf = evaluate_model('Random Forest Val', rf_cv, X_val, y_val)
val_xgb = evaluate_model('XGBoost Val', xgb_cv, X_val, y_val)
print(pd.concat([val_rf, val_xgb], ignore_index=True))

test_xgb = evaluate_model('XGBoost Test (Champion)', xgb_cv, X_test, y_test)
print(test_xgb)

# 5. Feature Importance
importances = xgb_cv.best_estimator_.feature_importances_
feature_importances = pd.Series(importances, index=X_train.columns).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=feature_importances.index)
plt.title('XGBoost Feature Importance for Churn Prediction')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()
plt.show()