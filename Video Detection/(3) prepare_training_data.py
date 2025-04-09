import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from preprocess_data import extract_features
import pickle

# Define paths and labels
base_dir = 'Video Detection/data'
label_map = {'Straight_ahead': 2, 'Turn_left': 3, 'Turn_right': 4}

# Lists for features and labels
X = []
y = []

# Loop through each label folder and extract features
for label_name, label_val in label_map.items():
    folder = os.path.join(base_dir, label_name)
    for filename in os.listdir(folder):
        if filename.endswith('.jpg'):
            file_path = os.path.join(folder, filename)
            features = extract_features(file_path)
            X.append(features)
            y.append(label_val)

X = np.array(X)
y = np.array(y)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)