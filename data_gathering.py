import cv2
import numpy as np
import os
import mediapipe as mp

# Setup folder/s for data collection
DATA_PATH = os.path.join("MP_Data")
actions = np.array(["straight_ahead", "turn_left", "turn_right"])
no_sequences = 30  # 30 videos worth of data
sequence_length = 30  # 30 frames in length

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
    # mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    # mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)


def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    # lefthand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    # righthand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)

    return np.concatenate([pose])
    # return np.concatenate([pose, lefthand, righthand])


def setup_datacollection_folder():
    for action in actions:
        for sequence in range(no_sequences):
            try:
                os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
            except:
                pass


# Setup Folders for Data Collection
setup_datacollection_folder()

cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for action in actions:  # loop through actions
        for sequence in range(no_sequences):  # loop through videos
            for frame_num in range(sequence_length):  # loop through video length
                # Read feed
                ret, frame = cap.read()

                # Make detections
                image, results = mediapipe_detection(frame, holistic)

                # Draw landmarks
                draw_landmarks(image, results)

                # Apply collection logic
                if frame_num == 0:
                    cv2.putText(
                        image, "STARTING COLLECTION", (120, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA
                    )
                    cv2.putText(
                        image, "Collecting frames for {} Video Number {}".format(action, sequence), (15, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA
                    )
                    print("STARTING COLLECTION")
                    # cv2.waitKey(3000)
                else:
                    cv2.putText(
                        image, "Collecting frames for {} Video Number {}".format(action, sequence), (15, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA
                    )
                    print("Collecting frames for {} Video Number {}".format(action, sequence))

                # Export keypoints
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence))
                np.save(npy_path, keypoints)

                # Show to screen
                cv2.imshow("AeroVision", image)

                # Break
                if cv2.waitKey(10) & 0xFF == ord("q"):
                    break

cap.release()
cv2.destroyAllWindows()
