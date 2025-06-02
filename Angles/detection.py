import cv2
import numpy as np
import mediapipe as mp
import math
from keras._tf_keras.keras.models import load_model

# ────────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────────────────────────────────────

# 1) The three actions in the order your LSTM expects:
ACTIONS = ["straight_ahead", "turn_left", "turn_right"]

# 2) Path to your trained model (adjust if necessary)
MODEL_PATH = "D:\Programming Projects\Repositories\AeroVision\Angles/action.h5"

# 3) How many frames to buffer before making one prediction
SEQUENCE_LENGTH = 30

# 4) Minimum confidence threshold to “lock in” a prediction
THRESHOLD = 0.4

# 5) Colors for drawing probability bars (one color per class)
PROB_COLORS = [(245,117,16), (117,245,16), (16,117,245)]
# ────────────────────────────────────────────────────────────────────────────────


# ────────────────────────────────────────────────────────────────────────────────
# ANGLE‐COMPUTATION UTILITIES
# ────────────────────────────────────────────────────────────────────────────────

def calculate_angle(a, b, c, direction="right"):
    """
    Compute the signed angle ABC in degrees, 
    then “flip” it so that horizontal arms face 0°-180° consistently.

    a, b, c are (x, y) tuples in *normalized* coords [0..1].
    direction = “right” or “left” flips the sign appropriately.
    """
    # Vector BA and BC
    ba = [a[0] - b[0], a[1] - b[1]]
    bc = [c[0] - b[0], c[1] - b[1]]

    # Raw angle (radians) between BA and BC
    angle_rad = math.atan2(ba[1], ba[0]) - math.atan2(bc[1], bc[0])
    angle_deg = math.degrees(angle_rad) % 360

    if direction == "right":
        return round((180 - angle_deg) % 360)
    else:  # direction == "left"
        return round((angle_deg + 180) % 360)


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
        # Rectangle: from x=0 to x=int(prob*100), vertical: 60+idx*40 → 90+idx*40
        cv2.rectangle(output, (0, 60 + idx * 40), 
                      (int(prob * 100), 90 + idx * 40), colors[idx], -1)
        cv2.putText(output, f"{actions[idx]}", 
                    (0, 85 + idx * 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return output


# ────────────────────────────────────────────────────────────────────────────────
# MAIN DETECTION LOOP
# ────────────────────────────────────────────────────────────────────────────────

def main():
    # 1) Load the trained LSTM model
    model = load_model(MODEL_PATH)

    # 2) Mediapipe Holistic for pose detection
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils

    # 3) Rolling buffer of features (each frame → 28‐dim vector)
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

            # Mirror image if you like (optional)
            # frame = cv2.flip(frame, 1)

            # 5) Run Mediapipe detection
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = holistic.process(image_rgb)
            image_rgb.flags.writeable = True
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            # 6) Draw pose landmarks (so you can see them)
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

            # 7) EXTRACT exactly the same 28 features per frame:
            #    ‣ 8 landmarks: 11,12,13,14,15,16,23,24
            #    ‣ each gives normalized (x,y,z) → 8×3=24 values
            #    ‣ plus 4 angles (lshoulder, rshoulder, lelbow, relbow)
            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                # 7a) Collect normalized xyz for the 8 “keypoints”:
                landmark_indices = [11, 12, 13, 14, 15, 16, 23, 24]
                feature_vector = []
                for idx in landmark_indices:
                    feature_vector.extend([
                        lm[idx].x, 
                        lm[idx].y, 
                        lm[idx].z
                    ])

                # 7b) Compute angles (always using normalized x,y):
                #     LEFT SHOULDER = 11–13–23
                a_ls = (lm[11].x, lm[11].y)
                b_ls = (lm[13].x, lm[13].y)
                c_ls = (lm[23].x, lm[23].y)
                angle_lshoulder = calculate_angle(a_ls, b_ls, c_ls)

                #     RIGHT SHOULDER = 12–14–24 (flip direction="left")
                a_rs = (lm[12].x, lm[12].y)
                b_rs = (lm[14].x, lm[14].y)
                c_rs = (lm[24].x, lm[24].y)
                angle_rshoulder = calculate_angle(a_rs, b_rs, c_rs, direction="left")

                #     LEFT ELBOW = 11–13–15 (flip direction="left")
                a_le = (lm[11].x, lm[11].y)
                b_le = (lm[13].x, lm[13].y)
                c_le = (lm[15].x, lm[15].y)
                angle_lelbow = calculate_angle(a_le, b_le, c_le, direction="left")

                #     RIGHT ELBOW = 12–14–16
                a_re = (lm[12].x, lm[12].y)
                b_re = (lm[14].x, lm[14].y)
                c_re = (lm[16].x, lm[16].y)
                angle_relbow = calculate_angle(a_re, b_re, c_re)

                # 7c) Append the four angles to the feature vector
                feature_vector.extend([
                    angle_lshoulder, 
                    angle_rshoulder, 
                    angle_lelbow, 
                    angle_relbow
                ])

                # 7d) Add this 28‐dim vector to the rolling buffer
                sequence.append(feature_vector)
                if len(sequence) > SEQUENCE_LENGTH:
                    sequence.pop(0)

                # 8) Overlay the angles “below” each joint in pixel coords:
                h, w, _ = image.shape
                def label_below(landmark_idx, text):
                    x_px = int(lm[landmark_idx].x * w)
                    y_px = int(lm[landmark_idx].y * h) + 20
                    cv2.putText(
                        image, text, 
                        (x_px, y_px), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                        (255, 255, 255), 2
                    )

                label_below(13, f"LS: {angle_lshoulder}")
                label_below(14, f"RS: {angle_rshoulder}")
                label_below(15, f"LE: {angle_lelbow}")
                label_below(16, f"RE: {angle_relbow}")

            # 9) Once we have 30 frames buffered, run prediction
            if len(sequence) == SEQUENCE_LENGTH:
                # (1, 30, 28)
                sequence_array = np.expand_dims(np.array(sequence), axis=0)
                probs = model.predict(sequence_array)[0]  # softmax over 3 classes
                # Which index has max probability?
                class_idx = np.argmax(probs)
                class_confidence = probs[class_idx]

                # If confidence > threshold, “lock in” that label
                if class_confidence > THRESHOLD:
                    signal = ACTIONS[class_idx]

                # OPTIONAL: draw probability bars
                image = prob_viz(probs, ACTIONS, image, PROB_COLORS)

                # Show predicted label + confidence in the top‐left
                cv2.rectangle(image, (0,0), (300, 40), (0,0,0), -1)
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

            # 10) Display the frame
            cv2.imshow("AeroVision Detection", image)

            # Exit on pressing 'q'
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
