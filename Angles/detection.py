import cv2
import mediapipe as mp
import math


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


mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

            lm = results.pose_landmarks.landmark
            h, w, _ = image.shape

            # Left shoulder angle (11, 13, 23)
            a = (lm[11].x * w, lm[11].y * h)
            b = (lm[13].x * w, lm[13].y * h)
            c = (lm[23].x * w, lm[23].y * h)
            angle_leftshoulder = calculate_angle(a, b, c)

            cx, cy = int(a[0]), int(a[1])
            cv2.putText(image, str(angle_leftshoulder), (cx, cy + 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 255, 255), 2, cv2.LINE_AA)

            # Right shoulder angle (12, 14, 24)
            a = (lm[12].x * w, lm[12].y * h)
            b = (lm[14].x * w, lm[14].y * h)
            c = (lm[24].x * w, lm[24].y * h)
            angle_rightshoulder = calculate_angle(a, b, c, "left")

            cx_r, cy_r = int(a[0]), int(a[1])
            cv2.putText(image, str(angle_rightshoulder), (cx_r, cy_r + 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # print(f"LeftShoulder: {angle_leftshoulder}, RightShoulder: {angle_rightshoulder}")

            # ----------------------------------------------------------------------------------- #

            # Left elbow angle (11, 13, 15)
            a = (lm[11].x * w, lm[11].y * h)
            b = (lm[13].x * w, lm[13].y * h)
            c = (lm[15].x * w, lm[15].y * h)
            angle_leftelbow = calculate_angle(a, b, c, "left")

            cx, cy = int(b[0]), int(b[1])
            cv2.putText(image, str(angle_leftelbow), (cx, cy + 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Right shoulder angle (12, 14, 16)
            a = (lm[12].x * w, lm[12].y * h)
            b = (lm[14].x * w, lm[14].y * h)
            c = (lm[16].x * w, lm[16].y * h)
            angle_rightshoulder = calculate_angle(a, b, c)

            cx_r, cy_r = int(b[0]), int(b[1])
            cv2.putText(image, str(angle_rightshoulder), (cx_r, cy_r + 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # print(f"LeftElbow: {angle_leftelbow}, RightElbow: {angle_rightshoulder}")

        cv2.imshow('Pose Landmarks', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
