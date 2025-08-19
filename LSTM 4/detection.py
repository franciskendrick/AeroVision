import os
import cv2
import numpy as np
import mediapipe as mp
from keras._tf_keras.keras.models import load_model
import time

# ─── CONFIG ───────────────────────────────────────────────────────────────────
ACTIONS         = ["chocks-inserted", "cut-engine", "start-engine", "stop", "straight_ahead", "turn_left", "turn_right"]
MODEL_PATH      = r"LSTM 4/best_action_lstm.h5"
SEQUENCE_LENGTH = 90
THRESHOLD       = 0.2

# ─── COLORS FOR VIZ ────────────────────────────────────────────────────────────
PROB_COLORS = [
    (128, 0, 128), (245, 117, 16), (117, 245, 16), (16, 117, 245),
    (255, 0, 127), (127, 0, 255), (0, 255, 127)
]

# ─── DRAW PROBABILITIES ────────────────────────────────────────────────────────
def prob_viz(probs, actions, frame, colors):
    out = frame.copy()
    for i, p in enumerate(probs):
        cv2.rectangle(out, (0, 60 + i*30), (int(p*200), 85 + i*30), colors[i], -1)
        cv2.putText(out, f"{actions[i]}: {p:.2f}", (5, 80 + i*30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)
    return out

# ─── EXTRACT 99-KEYPOINT VECTOR ────────────────────────────────────────────────
def extract_keypoints_full(results):
    if not results.pose_landmarks:
        return np.zeros(33 * 3)
    return np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]).flatten()

# ─── MAIN LOOP ─────────────────────────────────────────────────────────────────
def main():
    last_detection_time = time.time()

    print("🟢 Starting AeroVision Detection... Press 'q' to exit.")
    model = load_model(MODEL_PATH)

    pose_module = mp.solutions.pose
    pose = pose_module.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    sequence = []
    signal = ""
    confidence = 0.0

    last_signal = None
    last_time = time.time()

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)

        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.pose_landmarks, pose_module.POSE_CONNECTIONS)

        keypoints = extract_keypoints_full(results)
        sequence.append(keypoints)
        if len(sequence) > SEQUENCE_LENGTH:
            sequence.pop(0)

        signal = ""
        confidence = 0.0

        # Manual rule-based: "set-brakes"
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            lw_y = lm[16].y
            rw_y = lm[15].y
            mouth_y = lm[9].y
            right_shoulder_y = lm[12].y
            left_elbow_y = lm[14].y

            if (rw_y < mouth_y and lw_y > left_elbow_y and lw_y > right_shoulder_y):
                signal = "set-brakes"
                confidence = 1.0

        # LSTM model prediction
        if signal == "" and len(sequence) == SEQUENCE_LENGTH:
            input_seq = np.expand_dims(np.array(sequence), axis=0)
            probs = model.predict(input_seq, verbose=0)[0]
            max_idx = np.argmax(probs)
            max_prob = probs[max_idx]

            if max_prob > THRESHOLD:
                signal = ACTIONS[max_idx]
                confidence = max_prob

        # Visualize LSTM probabilities
        if len(sequence) == SEQUENCE_LENGTH and signal in ACTIONS:
            frame = prob_viz(probs, ACTIONS, frame, PROB_COLORS)

        # Time difference tracking
        if signal:
            current_time = time.time()
            time_diff = current_time - last_detection_time
            last_detection_time = current_time
            print(f"[DETECTED] {signal} ({confidence * 100:.0f}%) - Δt: {time_diff:.2f}s")

            cv2.putText(frame, f"{signal} ({confidence * 100:.0f}%)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        cv2.imshow("🛩️ AeroVision Detection", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    pose.close()
    cv2.destroyAllWindows()
    print("🛑 Detection stopped.")


if __name__ == "__main__":
    main()
