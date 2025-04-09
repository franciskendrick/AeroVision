from sklearn.model_selection import train_test_split
# from tensorflow.python.keras.utils import to_categorical
# from keras.utils import to_categorical
from keras._tf_keras.keras.utils import to_categorical
import numpy as np
import os

def preprocess_data():
    DATA_PATH = os.path.join("MP_Data")
    actions = np.array(["straight_ahead", "turn_left", "turn_right"])
    no_sequences = 30  # 30 videos worth of data
    sequence_length = 30  # 30 frames in length

    label_map = {label:num for num, label in enumerate(actions)}

    #
    sequences, labels = [], []
    for action in actions:
        for sequence in np.array(os.listdir(os.path.join(DATA_PATH, action))):
            window = []
            for frame_num in range(sequence_length):
                res = np.load(os.path.join(DATA_PATH, action, "{}.npy".format(frame_num)))
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])

    # print(np.array(sequences).shape)  # 90, 30, 132) or 90 videos, 30 frames per video, 132 keypoints per frame
    # print(np.array(labels).shape)  # (90,)

    #
    x = np.array(sequences)
    y = to_categorical(labels).astype(int)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05)

    # print(x_train.shape)  # (85, 30, 132)  -->  (171, 30, 132)
    # print(x_test.shape)  # (5, 30, 132)  -->  (9, 30, 132)
    # print(y_train.shape)  # (85, 3)  -->  (171, 3)
    # print(y_test.shape)  # (5, 3)  -->  (9, 3)

    return [x_train, x_test, y_train, y_test]