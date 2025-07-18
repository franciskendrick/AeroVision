import mediapipe as mp
import cv2
import csv
import os
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Path to the video file
video_path = "CNN 4\Videos/turn-right_1.mp4"  # Change this as needed

# Create output directory if it doesn't exist
os.makedirs("CNN 4", exist_ok=True)

cap = cv2.VideoCapture(video_path)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    n = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Make Detections
        results = holistic.process(image)
        # Recolor image back to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # Pose Detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        # Data extraction
        try:
            class_name = "turn_right"
            pose = results.pose_landmarks.landmark
            pose_row = list(np.array([[l.x, l.y, l.z, l.visibility] for l in pose]).flatten())
            pose_row.insert(0, class_name)

            with open("CNN 4/turn-right_1_data.csv", mode="a", newline="") as f:
                csv_writer = csv.writer(f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(pose_row)

            print(f"Frame {n} written.")
            n += 1
        except:
            pass

        # Display the video frame
        cv2.imshow('AeroVision', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
