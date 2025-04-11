import mediapipe as mp
import cv2
import csv
import os
import numpy as np


mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
n = 0


cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Make Detections
        results = holistic.process(image)
        
        # Recolor image back to BGR for rendering
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Pose Detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        # Collect data
        try:
            class_name = "start_engine"

            pose = results.pose_landmarks.landmark
            pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

            pose_row.insert(0, class_name)

            with open("CNN/coords.csv", mode="a", newline="") as f:
                csv_writer = csv.writer(f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(pose_row)
                print(n)
                if n == 600:
                    break

                n+=1
        except:
            pass

        num_coords = len(results.pose_landmarks.landmark)
        landmaarks = ["class"]
        for val in range(1, num_coords+1):
            landmaarks += [f"x{val}", f"y{val}", f"z{val}", f"v{val}"]

        # Show to screen
        cv2.imshow('AeroVision', image)

        # Break
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()