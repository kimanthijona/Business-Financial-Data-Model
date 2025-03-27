import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score, classification_report
)
from sklearn.inspection import permutation_importance

# Load the trained model
model_path = "D:/PesaJoy/Business_Financial_data_Model/models/model.pkl"
model = joblib.load(model_path)

# Load test data
X_test = pd.read_csv("D:/PesaJoy/Business_Financial_data_Model/data/X_test.csv")
y_test = pd.read_csv("D:/PesaJoy/Business_Financial_data_Model/data/y_test.csv")

# Ensure y_test is a 1D array
y_test = np.ravel(y_test)

# Make predictions
y_pred = model.predict(X_test)

# Check if model is classification or regression
if hasattr(model, "predict_proba"):  # Classification models usually have predict_proba
    print("🔹 Model Type: Classification")
    
    # Get prediction probabilities for AUC
    y_prob = model.predict_proba(X_test)[:, 1]

    # Classification metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    roc_auc = roc_auc_score(y_test, y_prob)

    # Print report
    print("📊 Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print(f"✅ Accuracy: {accuracy:.4f}")
    print(f"✅ Precision: {precision:.4f}")
    print(f"✅ Recall: {recall:.4f}")
    print(f"✅ F1 Score: {f1:.4f}")
    print(f"✅ ROC-AUC: {roc_auc:.4f}")

else:
    print("🔹 Model Type: Regression")
    
    # Regression metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"✅ Mean Absolute Error (MAE): {mae:.4f}")
    print(f"✅ Mean Squared Error (MSE): {mse:.4f}")
    print(f"✅ Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"✅ R² Score: {r2:.4f}")

# Feature Importance Plot (SHAP)
explainer = shap.Explainer(model, X_test)
shap_values = explainer(X_test)

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test)
plt.title("Feature Importance (SHAP)")
plt.show()

# Feature Importance (Permutation Importance)
perm_importance = permutation_importance(model, X_test, y_test, scoring='accuracy' if hasattr(model, "predict_proba") else 'r2')
feature_importance_df = pd.DataFrame({"Feature": X_test.columns, "Importance": perm_importance.importances_mean})
feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=feature_importance_df, palette="viridis")
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance (Permutation)")
plt.show()
