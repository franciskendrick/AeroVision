from sklearn.model_selection import train_test_split
from keras._tf_keras.keras.utils import to_categorical
import numpy as np
import os

def preprocess_data():
    DATA_PATH       = "D:\Programming Projects\Repositories\AeroVision\LSTM 3\MP_Data"
    actions         = ["straight_ahead", "turn_left", "turn_right"]
    sequence_length = 30  # frames per sequence
    label_map       = {label: idx for idx, label in enumerate(actions)}

    sequences, labels = [], []

    for action in actions:
        action_path = os.path.join(DATA_PATH, action)
        if not os.path.isdir(action_path):
            print(f"[WARNING] Missing folder for action '{action}', skipping.")
            continue

        # 1) List and sort all frame files
        files = sorted(f for f in os.listdir(action_path) if f.endswith(".npy"))
        n_frames = len(files)
        n_seqs   = n_frames // sequence_length

        if n_seqs == 0:
            print(f"[WARNING] Not enough frames for '{action}' ({n_frames} < {sequence_length}).")
            continue

        if n_frames % sequence_length != 0:
            print(f"[INFO] Dropping {n_frames % sequence_length} leftover frames for '{action}'.")
            files = files[: n_seqs * sequence_length]

        # 2) Group into sequences
        for seq_idx in range(n_seqs):
            start = seq_idx * sequence_length
            end   = start + sequence_length
            window = []
            for fname in files[start:end]:
                frame_data = np.load(os.path.join(action_path, fname))
                if frame_data.shape != (99,):
                    print(f"[WARNING] Bad shape {frame_data.shape} in {action}/{fname}, skipping this sequence.")
                    window = []
                    break
                window.append(frame_data)

            if len(window) == sequence_length:
                sequences.append(window)
                labels.append(label_map[action])

    # 3) Stack into arrays
    x = np.array(sequences)                # shape (N, 30,  99)
    y = to_categorical(labels).astype(int) # shape (N, 3)

    # 4) Train/test split
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.05, stratify=y, random_state=42
    )

    return x_train, x_test, y_train, y_test
