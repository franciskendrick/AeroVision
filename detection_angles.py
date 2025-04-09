import cv2
from cvzone.PoseModule import PoseDetector
import math

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    # a, b, c: [x, y]
    # Calculate the angle at point b with a and c as other points
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    dot_product = ba[0]*bc[0] + ba[1]*bc[1]
    magnitude_ba = math.sqrt(ba[0]**2 + ba[1]**2)
    magnitude_bc = math.sqrt(bc[0]**2 + bc[1]**2)
    if magnitude_ba * magnitude_bc == 0:
        return 0
    angle = math.degrees(math.acos(dot_product/(magnitude_ba * magnitude_bc)))
    return angle

# Initialize PoseDetector
detector = PoseDetector()
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1366)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    img = detector.findPose(img)
    lmList, bboxInfo = detector.findPosition(img, bboxWithHands=True)

    signal = "Unknown"
    if lmList:
        # Extract positions for key landmarks by their index (from MediaPipe Pose model):
        # Shoulders: 11 (left) and 12 (right)
        # Elbows: 13 (left) and 14 (right)
        # Wrists: 15 (left) and 16 (right)
        left_shoulder = lmList[11][1:3]
        right_shoulder = lmList[12][1:3]
        left_elbow = lmList[13][1:3]
        right_elbow = lmList[14][1:3]
        left_wrist = lmList[15][1:3]
        right_wrist = lmList[16][1:3]

        # Calculate angles for both arms (using shoulder, elbow, wrist)
        left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_arm_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

        # These thresholds are approximate and may need tuning:
        # - For a "Straight Ahead" signal, we may expect both arms to be somewhat vertical (elbow angle relatively high)
        # - For a "Turn Left" signal (from pilot’s view): right arm is extended horizontally and left arm is making the signal.
        # - For a "Turn Right" signal: left arm is extended horizontally and right arm is making the signal.

        # Example logic for "Straight Ahead":
        if left_arm_angle > 150 and right_arm_angle > 150:
            # Check vertical alignment by comparing y positions:
            # The wrist should be above the elbow and approximately aligned with the shoulder
            if (left_wrist[1] < left_elbow[1] < left_shoulder[1] and 
                right_wrist[1] < right_elbow[1] < right_shoulder[1]):
                signal = "Straight Ahead"

        # Example logic for "Turn Left":
        # For the right arm to be extended horizontally, the y-coordinate of the right wrist should be similar to the right shoulder,
        # and the x-coordinate should be significantly to the right of the shoulder (from the pilot’s perspective).
        # At the same time, the left arm might be making a waving motion.
        elif abs(right_shoulder[1] - right_wrist[1]) < 30 and (right_wrist[0] - right_shoulder[0] > 50):
            signal = "Turn Left"

        # Example logic for "Turn Right":
        # Mirror of "Turn Left": left arm extended horizontally.
        elif abs(left_shoulder[1] - left_wrist[1]) < 30 and (left_shoulder[0] - left_wrist[0] > 50):
            signal = "Turn Right"

        # You might want to display calculated angles for debugging:
        cv2.putText(img, f"L-Arm: {int(left_arm_angle)}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(img, f"R-Arm: {int(right_arm_angle)}", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.putText(img, f'Signal: {signal}', (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if signal == "Turn Right" or signal == "Turn Left":
        print(signal, left_arm_angle, right_arm_angle)
    else:
        print(signal)


    cv2.imshow("Aerovision", img)
    if cv2.waitKey(1) == 27:  # ESC key to exit
        break

cap.release()
cv2.destroyAllWindows()
