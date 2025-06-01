import cv2
import mediapipe as mp
import math
import os
import csv
import time

# Config
# actions = ["straight_ahead", "turn_left", "turn_right"]
actions = ["turn_left", "turn_right"]
no_sequences = 30
sequence_length = 30
DATA_PATH = "Collected_Data"
all_data = []

# Angle calculation
def calculate_angle(a, b, c, direction="right"):
    ba = [a[0] - b[0], a[1] - b[1]]
    bc = [c[0] - b[0], c[1] - b[1]]
    angle_rad = math.atan2(ba[1], ba[0]) - math.atan2(bc[1], bc[0])
    angle_deg = math.degrees(angle_rad) % 360
    if direction == "right":
        flipped_angle = (180 - angle_deg) % 360
    elif direction == "left":
        flipped_angle = (angle_deg + 180) % 360
    return round(flipped_angle)


# MediaPipe setup
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for action in actions:
        for sequence in range(no_sequences):
            data_rows = []
            print(f"\nStarting collection for {action} | Sequence {sequence}")
            time.sleep(1)

            for frame_num in range(sequence_length):
                ret, frame = cap.read()
                if not ret:
                    continue

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                    lm = results.pose_landmarks.landmark

                    # Extract normalized landmarks
                    keypoints = [
                        lm[11], lm[12], lm[13], lm[14], lm[15], lm[16], lm[23], lm[24]
                    ]
                    row = []
                    for point in keypoints:
                        row.extend([point.x, point.y, point.z])

                    # Compute angles using normalized (x, y)
                    a_ls = (lm[11].x, lm[11].y)
                    b_ls = (lm[13].x, lm[13].y)
                    c_ls = (lm[23].x, lm[23].y)
                    angle_lshoulder = calculate_angle(a_ls, b_ls, c_ls)

                    a_rs = (lm[12].x, lm[12].y)
                    b_rs = (lm[14].x, lm[14].y)
                    c_rs = (lm[24].x, lm[24].y)
                    angle_rshoulder = calculate_angle(a_rs, b_rs, c_rs, "left")

                    a_le = (lm[11].x, lm[11].y)
                    b_le = (lm[13].x, lm[13].y)
                    c_le = (lm[15].x, lm[15].y)
                    angle_lelbow = calculate_angle(a_le, b_le, c_le, "left")

                    a_re = (lm[12].x, lm[12].y)
                    b_re = (lm[14].x, lm[14].y)
                    c_re = (lm[16].x, lm[16].y)
                    angle_relbow = calculate_angle(a_re, b_re, c_re)

                    row.extend([angle_lshoulder, angle_rshoulder, angle_lelbow, angle_relbow])
                    row.insert(0, action)
                    all_data.append(row)

                    # Get image dimensions
                    h, w, _ = image.shape

                    # Convert landmarks to pixel coordinates and add offset
                    def label_below_joint(landmark, label_text):
                        x = int(landmark.x * w)
                        y = int(landmark.y * h) + 20  # 20px below the joint
                        cv2.putText(image, label_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

                    # Add labels below the joints
                    label_below_joint(lm[13], f"LS: {angle_lshoulder}")  # Left Shoulder
                    label_below_joint(lm[14], f"RS: {angle_rshoulder}")  # Right Shoulder
                    label_below_joint(lm[15], f"LE: {angle_lelbow}")     # Left Elbow
                    label_below_joint(lm[16], f"RE: {angle_relbow}")     # Right Elbow

                # Starting frame overlay
                if frame_num == 0:
                    cv2.putText(image, 'STARTING COLLECTION', (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
                    cv2.imshow('AeroVision', image)
                    cv2.waitKey(1000)
                else:
                    cv2.putText(image, f'Collecting {action} | Video {sequence}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                    cv2.imshow('AeroVision', image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

cap.release()
cv2.destroyAllWindows()

# Save single CSV with all data
os.makedirs(DATA_PATH, exist_ok=True)
csv_path = os.path.join(DATA_PATH, "all_data.csv")
with open(csv_path, mode='w', newline='') as f:
    writer = csv.writer(f)
    header = ["class"]
    for i in [11,12,13,14,15,16,23,24]:
        header.extend([f"{i}x", f"{i}y", f"{i}z"])
    header += ["lshoulder", "rshoulder", "lelbow", "relbow"]
    writer.writerow(header)
    writer.writerows(all_data)