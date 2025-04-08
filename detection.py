import cv2
from cvzone.PoseModule import PoseDetector

# Create a PoseDetector object
detector = PoseDetector()

# Start the webcam
cap = cv2.VideoCapture(0)
cv2.setWindowProperty("Result", cv2.WINDOW_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    success, img = cap.read()
    if not success:
        break

    # Find pose
    img = detector.findPose(img)

    # Get landmarks and bounding box
    lmList, bboxInfo = detector.findPosition(img, bboxWithHands=True)

    # Show the result
    cv2.imshow("Result", img)
    if cv2.waitKey(1) == 27:  # ESC key to exit
        break

cap.release()
cv2.destroyAllWindows()
