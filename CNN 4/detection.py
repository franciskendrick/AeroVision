import pandas as pd
import mediapipe as mp
import cv2
import numpy as np
import pickle

# Load trained model
with open('CNN 4/body_language.pkl', 'rb') as f:
    model = pickle.load(f)

# Get class names in proper order
class_names = model.classes_

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Webcam
cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Recolor to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw pose landmarks
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        try:
            # Extract landmarks
            pose = results.pose_landmarks.landmark
            row = list(np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in pose]).flatten())

            # Prediction
            X = pd.DataFrame([row])
            body_language_class = model.predict(X)[0]
            body_language_probs = model.predict_proba(X)[0]

            # === Drawing ===

            # Status box for class label
            cv2.rectangle(image, (0, 0), (250, 30), (245, 117, 16), -1)
            cv2.putText(image, f'Class: {body_language_class}',
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            # Probabilities (draw each class + prob)
            start_y = 40
            box_height = 20
            for i, (cls, prob) in enumerate(zip(class_names, body_language_probs)):
                y = start_y + i * box_height
                text = f"{cls}: {prob:.2f}"
                cv2.putText(image, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        except Exception as e:
            pass  # Fail silently to avoid crashing if landmarks aren't detected

        cv2.imshow('AeroVision', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
