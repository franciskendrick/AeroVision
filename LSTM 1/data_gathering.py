import cv2
import numpy as np
import os
import mediapipe as mp
from time import sleep

# Setup folder/s for data collection
DATA_PATH = os.path.join("MP_Data")
actions = np.array(["straight_ahead", "turn_left", "turn_right"])
no_sequences = 30  # 30 videos worth of data
sequence_length = 30  # 30 frames per video

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] 
                     for res in results.pose_landmarks.landmark]) if results.pose_landmarks else np.zeros((33, 4))
    return pose.flatten()  # 33 keypoints * 4 = 132

# Setup folders
def setup_datacollection_folder():
    for action in actions:
        os.makedirs(os.path.join(DATA_PATH, action), exist_ok=True)

setup_datacollection_folder()

cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for action in actions:
        for sequence in range(no_sequences):
            print(f"\nStarting collection for {action}, video {sequence}")
            sleep(2)  # pause for user to prepare
            window = []  # buffer to hold 30 frames
            for frame_num in range(sequence_length):
                ret, frame = cap.read()
                image, results = mediapipe_detection(frame, holistic)
                draw_landmarks(image, results)

                # Display status
                cv2.putText(image, f'{action} | Video {sequence} | Frame {frame_num + 1}',
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                keypoints = extract_keypoints(results)
                window.append(keypoints)

                cv2.imshow('AeroVision - Data Collection', image)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

            # Save full 30-frame sequence to one .npy file
            sequence_path = os.path.join(DATA_PATH, action, f"{sequence}.npy")
            np.save(sequence_path, np.array(window))  # shape (30, 132)

cap.release()
cv2.destroyAllWindows()