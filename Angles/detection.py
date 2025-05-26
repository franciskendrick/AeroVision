import cv2
import mediapipe as mp
import math


def calculate_angle(a, b, c):
    # Vectors BA and BC
    ba = [a[0] - b[0], a[1] - b[1]]
    bc = [c[0] - b[0], c[1] - b[1]]

    # Compute the angle between the vectors using atan2
    angle_rad = math.atan2(ba[1], ba[0]) - math.atan2(bc[1], bc[0])
    angle_deg = math.degrees(angle_rad) % 360

    flipped_angle = (180 - angle_deg) % 360

    return round(flipped_angle, 2)


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

            # Get landmarks 11, 13, 23
            lm = results.pose_landmarks.landmark
            h, w, _ = image.shape

            a = (lm[11].x * w, lm[11].y * h)
            b = (lm[13].x * w, lm[13].y * h)
            c = (lm[23].x * w, lm[23].y * h)

            angle = calculate_angle(a, b, c)
            print(angle)

            # Display angle below point 11
            cx, cy = int(a[0]), int(a[1])
            cv2.putText(image, str(angle), (cx, cy + 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Pose Landmarks', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
