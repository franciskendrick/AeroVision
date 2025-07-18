import pandas as pd
import os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# === CONFIG ===
csv_path = "D:/Programming Projects/Repositories/AeroVision/CNN 4/straight-ahead_1 - turn-left_1 - turn-right_1.csv"
output_model_path = "CNN 4/body_language.pkl"

# === LOAD AND CLEAN DATA ===
df = pd.read_csv(csv_path, on_bad_lines="skip")

# Log basic info
print(f"[INFO] Loaded {len(df)} rows and {df.shape[1]} columns from {csv_path}")

# Check for missing values
if df.isnull().values.any():
    print("[WARN] Missing values found. Filling with zeros.")
    df = df.fillna(0)

# Separate features and target
X = df.drop('class', axis=1)
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

y = df['class']

# === SPLIT DATA ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)

# === DEFINE PIPELINES ===
pipelines = {
    'lr': make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    'rc': make_pipeline(StandardScaler(), RidgeClassifier()),
    'rf': RandomForestClassifier(),  # Tree-based models don't need scaling
    'gb': GradientBoostingClassifier(),
}

# === TRAIN MODELS ===
fit_models = {}
accuracies = {}

for name, pipeline in pipelines.items():
    print(f"[INFO] Training {name}...")
    model = pipeline.fit(X_train, y_train)
    fit_models[name] = model
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies[name] = acc
    print(f"{name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

# === CONFUSION MATRIX FOR BEST MODEL ===
best_model_name = max(accuracies, key=accuracies.get)
best_model = fit_models[best_model_name]
print(f"[INFO] Best model: {best_model_name} (Accuracy: {accuracies[best_model_name]:.4f})")

print("[INFO] Displaying confusion matrix...")
ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test)
plt.title(f"Confusion Matrix: {best_model_name}")
plt.show()

# === SAVE MODEL ===
with open(output_model_path, 'wb') as f:
    pickle.dump(best_model, f)
print(f"[INFO] Saved best model '{best_model_name}' to {output_model_path}")
