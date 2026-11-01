"""
Breast Cancer Prediction - Training Script
Kya karta hai:
1. Dataset load karta hai (sklearn built-in)
2. 4 alag ML models train + compare karta hai
3. Best model ko model.pkl me save karta hai
4. Confusion matrix aur comparison chart save karta hai
"""
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             ConfusionMatrixDisplay, f1_score)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# 1. Data load
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target  # 0 = malignant (cancer), 1 = benign (safe)
print(f"Dataset shape: {X.shape}")
print(f"Malignant: {(y == 0).sum()} | Benign: {(y == 1).sum()}")

# 2. Train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(kernel="rbf", probability=True),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
}

results, fitted = [], {}
for name, clf in models.items():
    pipe = Pipeline([("scaler", StandardScaler()), ("model", clf)])
    cv = cross_val_score(pipe, X_train, y_train, cv=5).mean()
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    results.append({
        "Model": name,
        "CV Accuracy": round(cv, 4),
        "Test Accuracy": round(accuracy_score(y_test, pred), 4),
        "F1 Score": round(f1_score(y_test, pred), 4),
    })
    fitted[name] = pipe

res = pd.DataFrame(results).sort_values("Test Accuracy", ascending=False)
print("\n=== Model Comparison ===")
print(res.to_string(index=False))

# 4. Best model
best_name = res.iloc[0]["Model"]
best = fitted[best_name]
print(f"\nBest model: {best_name}")
print(classification_report(y_test, best.predict(X_test),
                            target_names=data.target_names))

# 5. Save model + feature info for the app
joblib.dump({"model": best, "features": list(data.feature_names),
             "means": X.mean().to_dict(), "mins": X.min().to_dict(),
             "maxs": X.max().to_dict(), "name": best_name}, "model.pkl")

# 6. Charts
ConfusionMatrixDisplay.from_estimator(
    best, X_test, y_test, display_labels=data.target_names, cmap="Blues")
plt.title(f"Confusion Matrix - {best_name}")
plt.savefig("confusion_matrix.png", dpi=120, bbox_inches="tight")
plt.close()

res.set_index("Model")[["CV Accuracy", "Test Accuracy", "F1 Score"]].plot.bar(
    figsize=(8, 5), ylim=(0.9, 1.0))
plt.title("Model Comparison")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=120)
print("\nSaved: model.pkl, confusion_matrix.png, model_comparison.png")
