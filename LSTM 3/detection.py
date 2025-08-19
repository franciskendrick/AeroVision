import os
import cv2
import numpy as np
import mediapipe as mp
from keras._tf_keras.keras.models import load_model

# ─── CONFIG ───────────────────────────────────────────────────────────────────
ACTIONS         = ["straight_ahead", "turn_left", "turn_right"]
MODEL_PATH      = r"LSTM 3/best_action_lstm.h5"  # point to your new saved model
SEQUENCE_LENGTH = 30
THRESHOLD       = 0.4
# ────────────────────────────────────────────────────────────────────────────────

# ─── PROB VIZ ──────────────────────────────────────────────────────────────────
def prob_viz(probs, actions, frame, colors):
    out = frame.copy()
    for i, p in enumerate(probs):
        cv2.rectangle(out, (0, 60 + i*30), (int(p*200), 85 + i*30), colors[i], -1)
        cv2.putText(out, f"{actions[i]}: {p:.2f}", (5, 80 + i*30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)
    return out

PROB_COLORS = [(245,117,16), (117,245,16), (16,117,245)]

# ─── FEATURE EXTRACTOR ────────────────────────────────────────────────────────
def extract_keypoints_full(results):
    if not results.pose_landmarks:
        return np.zeros(33 * 3)
    return np.array([[lm.x, lm.y, lm.z]
                     for lm in results.pose_landmarks.landmark]).flatten()

# ─── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    model       = load_model(MODEL_PATH)
    pose_module = mp.solutions.pose
    pose        = pose_module.Pose(min_detection_confidence=0.5,
                                   min_tracking_confidence=0.5)
    sequence    = []
    signal      = ""

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        if not ret:
            break

        # Mediapipe inference
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res     = pose.process(img_rgb)
        cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR, dst=frame)

        # Draw feedback
        if res.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, res.pose_landmarks, pose_module.POSE_CONNECTIONS)

        # 99‑dim keypoints
        kp = extract_keypoints_full(res)
        sequence.append(kp)
        if len(sequence) > SEQUENCE_LENGTH:
            sequence.pop(0)

        # Predict when buffer full
        if len(sequence) == SEQUENCE_LENGTH:
            seq_arr = np.expand_dims(np.array(sequence), axis=0)  # (1,30,99)
            probs   = model.predict(seq_arr)[0]
            idx     = np.argmax(probs)
            conf    = probs[idx]
            if conf > THRESHOLD:
                signal = ACTIONS[idx]
            frame = prob_viz(probs, ACTIONS, frame, PROB_COLORS)
            cv2.putText(frame, f"{signal} ({conf:.2f})",
                        (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

        cv2.imshow("AeroVision Detection", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    pose.close()

if __name__ == "__main__":
    main()
