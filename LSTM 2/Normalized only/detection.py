import cv2
import numpy as np
import mediapipe as mp
from keras._tf_keras.keras.models import load_model

# ────────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────────────────────────────────────

# 1) The three actions in the order your LSTM expects:
ACTIONS = ["straight_ahead", "turn_left", "turn_right"]

# 2) Path to your trained model (adjust if necessary)
MODEL_PATH = r"D:\Programming Projects\Repositories\AeroVision\LSTM 2\Normalized only\action.h5"

# 3) How many frames to buffer before making one prediction
SEQUENCE_LENGTH = 30

# 4) Minimum confidence threshold to “lock in” a prediction
THRESHOLD = 0.4

# 5) Colors for drawing probability bars (one color per class)
PROB_COLORS = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]
# ────────────────────────────────────────────────────────────────────────────────


# ────────────────────────────────────────────────────────────────────────────────
# PROBABILITY‐BAR VISUALIZATION (OPTIONAL)
# ────────────────────────────────────────────────────────────────────────────────

def prob_viz(probs, actions, input_frame, colors):
    """
    Draw horizontal bars showing each action’s probability.
    probs: 1D array of length len(actions), e.g. [0.1, 0.8, 0.1].
    """
    output = input_frame.copy()
    for idx, prob in enumerate(probs):
        cv2.rectangle(
            output,
            (0, 60 + idx * 40),
            (int(prob * 100), 90 + idx * 40),
            colors[idx],
            -1
        )
        cv2.putText(
            output,
            f"{actions[idx]}",
            (0, 85 + idx * 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
    return output


# ────────────────────────────────────────────────────────────────────────────────
# LANDMARK NORMALIZATION UTILITIES
# ────────────────────────────────────────────────────────────────────────────────

def normalize_landmarks_xy(landmarks, hip_idx1=23, hip_idx2=24, shoulder_idx1=11, shoulder_idx2=12):
    """
    Midpoint‐normalize only x and y coordinates for landmarks 11,12,13,14,15,16,23,24.
    - Uses the midpoint of the two hips as the translation anchor.
    - Uses the distance between the two shoulders as the scale factor.
    Returns a dict: { idx: (x_norm, y_norm) } for each idx in [11,12,13,14,15,16,23,24].
    """
    # Read raw coordinates of the two hips
    hip1 = np.array([landmarks[hip_idx1].x, landmarks[hip_idx1].y])
    hip2 = np.array([landmarks[hip_idx2].x, landmarks[hip_idx2].y])
    midpoint = (hip1 + hip2) / 2

    # Read raw coordinates of the two shoulders to compute scale
    sh1 = np.array([landmarks[shoulder_idx1].x, landmarks[shoulder_idx1].y])
    sh2 = np.array([landmarks[shoulder_idx2].x, landmarks[shoulder_idx2].y])
    shoulder_dist = np.linalg.norm(sh1 - sh2)
    if shoulder_dist < 1e-6:
        shoulder_dist = 1e-6  # avoid division by zero

    normalized = {}
    for idx in [11, 12, 13, 14, 15, 16, 23, 24]:
        raw_xy = np.array([landmarks[idx].x, landmarks[idx].y])
        norm_xy = (raw_xy - midpoint) / shoulder_dist
        normalized[idx] = (float(norm_xy[0]), float(norm_xy[1]))
    return normalized


# ────────────────────────────────────────────────────────────────────────────────
# MAIN DETECTION LOOP
# ────────────────────────────────────────────────────────────────────────────────

def main():
    # 1) Load the trained LSTM model
    model = load_model(MODEL_PATH)

    # 2) Mediapipe Holistic for pose detection
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils

    # 3) Rolling buffer of features (each frame → 16‐dim vector)
    sequence = []

    # 4) Current “signal” (the last “stable” class) we display
    signal = ""

    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Optional mirror:
            # frame = cv2.flip(frame, 1)

            # 5) Run Mediapipe detection
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = holistic.process(image_rgb)
            image_rgb.flags.writeable = True
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            # 6) Draw pose landmarks (visual feedback)
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS
                )

            # 7) EXTRACT exactly the same 16 features per frame:
            #    ‣ 8 landmarks: 11,12,13,14,15,16,23,24
            #    ‣ each gives normalized (x,y) → 8×2=16 values
            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark

                # 7a) Normalize x,y of those 8 keypoints
                norm_dict = normalize_landmarks_xy(lm)

                # 7b) Create feature vector of length 16: [x11, y11, x12, y12, ..., x24, y24]
                feature_vector = []
                for idx in [11, 12, 13, 14, 15, 16, 23, 24]:
                    x_n, y_n = norm_dict[idx]
                    feature_vector.extend([x_n, y_n])

                # 7c) Add this 16‐dim vector to the rolling buffer
                sequence.append(feature_vector)
                if len(sequence) > SEQUENCE_LENGTH:
                    sequence.pop(0)

            # 8) Once we have 30 frames buffered, run prediction
            if len(sequence) == SEQUENCE_LENGTH:
                # (1, 30, 16)
                sequence_array = np.expand_dims(np.array(sequence), axis=0)
                probs = model.predict(sequence_array)[0]  # softmax over 3 classes
                class_idx = np.argmax(probs)
                class_confidence = probs[class_idx]

                # If confidence > threshold, “lock in” that label
                if class_confidence > THRESHOLD:
                    signal = ACTIONS[class_idx]

                # OPTIONAL: draw probability bars
                image = prob_viz(probs, ACTIONS, image, PROB_COLORS)

                # Show predicted label + confidence in the top‐left
                cv2.rectangle(image, (0, 0), (300, 40), (0, 0, 0), -1)
                cv2.putText(
                    image,
                    f"{signal} ({class_confidence:.2f})",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

            # 9) Display the frame
            cv2.imshow("AeroVision Detection", image)

            # Exit on pressing 'q'
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
