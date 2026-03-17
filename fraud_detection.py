#PROJECT: Credit Card Fraud Detection (Anomaly Detection Study)
#APPLICATION TO ROBOTICS: This project demonstrates the implementation of -
#-high-recall models (XGBoost/SMOTE) to detect rare events in imbalanced - 
#-datasets—a critical skill for robot sensor fault detection and -
#-predictive maintenance.






import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, precision_score, recall_score, f1_score, roc_curve, auc
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns


# 1. Load dataset
data = pd.read_csv("creditcard.csv")
X = data.drop('Class', axis=1)
y = data['Class']

# 2. Preprocess
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Logistic Regression with SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

lr = LogisticRegression(max_iter=1000, class_weight='balanced')
lr.fit(X_train_res, y_train_res)
y_pred_lr = lr.predict(X_test)

print("=== Logistic Regression Evaluation ===")
print(confusion_matrix(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr, target_names=['Normal','Fraud']))
print(f"ROC-AUC score: {roc_auc_score(y_test, y_pred_lr):.4f}\n")

# 4. Random Forest on original training set
print("=== Random Forest Evaluation ===")
rf = RandomForestClassifier(n_estimators=50, class_weight='balanced', random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf, target_names=['Normal','Fraud']))
print(f"ROC-AUC score: {roc_auc_score(y_test, y_pred_rf):.4f}\n")

# 5. XGBoost on original training set
print("=== XGBoost Evaluation ===")
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
xgb = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss',
    scale_pos_weight=scale_pos_weight
)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

print(confusion_matrix(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_xgb, target_names=['Normal','Fraud']))
print(f"ROC-AUC score: {roc_auc_score(y_test, y_pred_xgb):.4f}\n")

# 6. Comparison Table
models_preds = {
    "Logistic Regression": y_pred_lr,
    "Random Forest": y_pred_rf,
    "XGBoost": y_pred_xgb
}

comparison_metrics = []

for name, y_pred in models_preds.items():
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_score = roc_auc_score(y_test, y_pred)
    comparison_metrics.append({
        "Model": name,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "ROC-AUC": roc_score
    })

comparison_df = pd.DataFrame(comparison_metrics)
print("=== Comparison Table ===")
print(comparison_df)

# 7. Figures

# Comparison Heatmap
plt.figure(1)
sns.heatmap(comparison_df.set_index('Model'), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Comparison of Models")
plt.show()

# ROC Curves
plt.figure(2)
for model, label in zip([lr, rf, xgb], ["Logistic Regression", "Random Forest", "XGBoost"]):
    y_pred_prob = model.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc_val = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{label} (AUC = {roc_auc_val:.4f})")

plt.plot([0,1], [0,1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves Comparison")
plt.legend()
plt.show()
