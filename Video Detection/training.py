import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

# --- Step 1: Feature Extraction (reuse from Step 3) ---
def extract_features(image_path, size=(64, 64)):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"[WARNING] Failed to load {image_path}")
        return None
    img_resized = cv2.resize(img, size)
    img_normalized = img_resized / 255.0
    return img_normalized.flatten()

# --- Step 2: Load and preprocess all training images ---
data_dir = "Video Detection/data"
label_map = {'Straight_ahead': 2, 'Turn_left': 3, 'Turn_right': 4}
X = []
y = []

for label_name, label_val in label_map.items():
    folder = os.path.join(data_dir, label_name)
    if not os.path.exists(folder):
        print(f"[WARNING] Directory not found: {folder}")
        continue
    for filename in os.listdir(folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            path = os.path.join(folder, filename)
            features = extract_features(path)
            if features is not None:
                X.append(features)
                y.append(label_val)

print(f"[INFO] Loaded {len(X)} samples.")

# --- Step 3: Train the model ---
print("[INFO] Training Random Forest Classifier...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)
print("[INFO] Training complete.")

# --- Step 4: Save the model ---
dump(clf, 'classifier_model.pkl')
print("[INFO] Model saved as 'classifier_model.pkl'")
