import os
import numpy as np
from sklearn.model_selection import train_test_split
from keras._tf_keras.keras.utils import to_categorical

def preprocess_data():
    DATA_PATH = "D:\\Programming Projects\\Repositories\\AeroVision\\LSTM 4\\MP_Data"
    SEQUENCE_LENGTH = 90
    NUM_FEATURES = 99

    # Automatically detect action folders
    actions = [d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))]
    if not actions:
        raise ValueError("No action folders found in MP_Data.")

    label_map = {label: idx for idx, label in enumerate(actions)}
    sequences, labels = [], []

    for action in actions:
        action_path = os.path.join(DATA_PATH, action)
        files = sorted(f for f in os.listdir(action_path) if f.endswith(".npy"))
        n_frames = len(files)
        n_seqs = n_frames // SEQUENCE_LENGTH

        if n_seqs == 0:
            print(f"[WARNING] Not enough frames for '{action}' ({n_frames} < {SEQUENCE_LENGTH}).")
            continue

        if n_frames % SEQUENCE_LENGTH != 0:
            print(f"[INFO] Dropping {n_frames % SEQUENCE_LENGTH} leftover frames for '{action}'.")
            files = files[: n_seqs * SEQUENCE_LENGTH]

        for seq_idx in range(n_seqs):
            start = seq_idx * SEQUENCE_LENGTH
            end = start + SEQUENCE_LENGTH
            window = []

            for fname in files[start:end]:
                path = os.path.join(action_path, fname)
                arr = np.load(path)

                if arr.shape != (NUM_FEATURES,):
                    print(f"[WARNING] Skipping bad frame: {action}/{fname} with shape {arr.shape}")
                    window = []
                    break

                window.append(arr)

            if len(window) == SEQUENCE_LENGTH:
                sequences.append(window)
                labels.append(label_map[action])

    if not sequences:
        raise ValueError("No valid sequences were collected. Check your data.")

    x = np.array(sequences)
    y = to_categorical(labels).astype(int)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.05, stratify=y, random_state=42
    )

    return x_train, x_test, y_train, y_test
