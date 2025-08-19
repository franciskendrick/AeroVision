import os
import cv2
import numpy as np
import mediapipe as mp

# Setup MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Define path to videos
VIDEO_DIR = 'D:\Programming Projects\Repositories\AeroVision\CNN 4\Videos'
OUTPUT_DIR = 'MP_Data'

# Create output directories based on labels
for filename in os.listdir(VIDEO_DIR):
    if filename.endswith(".mp4"):
        label = filename.split('_')[0] + '-' + filename.split('_')[1].split('.')[0]
        label_dir = os.path.join(OUTPUT_DIR, label)
        os.makedirs(label_dir, exist_ok=True)

# Start frame processing
for filename in os.listdir(VIDEO_DIR):
    if not filename.endswith(".mp4"):
        continue

    label = filename.split('_')[0] + '-' + filename.split('_')[1].split('.')[0]
    label_dir = os.path.join(OUTPUT_DIR, label)

    filepath = os.path.join(VIDEO_DIR, filename)
    cap = cv2.VideoCapture(filepath)

    frame_count = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Convert the BGR image to RGB before processing
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(image_rgb)

        if result.pose_landmarks:
            # Flatten landmark data to a 99-dim vector (33 landmarks × (x, y, z))
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in result.pose_landmarks.landmark]).flatten()
            np.save(os.path.join(label_dir, f'{frame_count}.npy'), landmarks)

        frame_count += 1

    cap.release()

pose.close()
print("✅ Data gathering complete.")
