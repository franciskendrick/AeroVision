import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras._tf_keras.keras.utils import to_categorical


# 1) Load your single CSV
df = pd.read_csv("D:\Programming Projects\Repositories\AeroVision\LSTM 2\data.csv")


# 2) Check total rows = 2700 (90 sequences × 30 frames)
n_rows = len(df)
print("Total rows in CSV:", n_rows)  # should print 2700

# 3) Extract labels and features
labels_series = df["class"]            # length = 2700
features_df = df.drop(columns=["class"])  # shape = (2700, 28)

# 4) Convert string labels to integers
label_map = {"straight_ahead": 0, "turn_left": 1, "turn_right": 2}
labels_int = labels_series.map(label_map).to_numpy()  # e.g. array of length 2700

# 5) Figure out how many sequences
#    You said: 30 videos per class × 3 classes = 90 total videos
#    Each video has 30 frames => 30 rows per sequence.
sequence_length = 30
num_sequences = int(n_rows / sequence_length)  # 2700/30 = 90

# 6) Reshape features into (90, 30, num_features)
num_features = features_df.shape[1]  # should be 28
X = features_df.to_numpy().reshape(num_sequences, sequence_length, num_features)
print("X.shape:", X.shape)  # (90, 30, 28)

# 7) Build a 90-element array of video-level labels:
#    We can reshape labels_int into (90, 30), then take the first frame’s label for each group of 30.
y_int = labels_int.reshape(num_sequences, sequence_length)[:, 0]
print("y_int.shape:", y_int.shape)  # (90,)

# 8) One-hot encode y to get shape (90, 3)
y = to_categorical(y_int, num_classes=3).astype(int)
print("y.shape:", y.shape)  # (90, 3)

# 9) Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.05, random_state=42
)

print("x_train.shape, y_train.shape:", x_train.shape, y_train.shape)
print("x_test.shape,  y_test.shape: ", x_test.shape, y_test.shape)

# Now you’re ready to build and train your LSTM on (x_train, y_train).
# Save arrays if you want:
np.save("x_train.npy", x_train)
np.save("x_test.npy", x_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)