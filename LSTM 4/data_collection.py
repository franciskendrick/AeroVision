import os
import shutil
import cv2
import numpy as np
import mediapipe as mp

# Constants
VIDEO_FILES = [
    r"D:\Programming Projects\Repositories\AeroVision\CNN 4\Videos\chocks-inserted_1.mp4",
    r"D:\Programming Projects\Repositories\AeroVision\CNN 4\Videos\chocks-inserted_2.mp4"
]
OUTPUT_DIR = os.path.join("MP_Data", "chocks-inserted")

# Clear out existing directory for chocks-inserted
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

frame_count = 0

for video_path in VIDEO_FILES:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"⚠ Could not open video: {video_path}")
        continue

    while True:
        success, frame = cap.read()
        if not success:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(image_rgb)

        if result.pose_landmarks:
            landmarks = np.array(
                [[lm.x, lm.y, lm.z] for lm in result.pose_landmarks.landmark]
            ).flatten()
            np.save(os.path.join(OUTPUT_DIR, f"{frame_count}.npy"), landmarks)

        frame_count += 1

    cap.release()

pose.close()
print("✅ Chocks Inserted data collection complete.")
