import cv2
import os

# Create a folder structure for storing images for each signal
labels = {'2': 'Straight_ahead', '3': 'Turn_left', '4': 'Turn_right'}
base_dir = 'Video Detection/data'
for label in labels.values():
    os.makedirs(os.path.join(base_dir, label), exist_ok=True)

cap = cv2.VideoCapture(0)  # Opens the default camera

print("Press 2, 3, or 4 to capture an image for the respective signal.")
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow('Live Capture', frame)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
    elif chr(key) in labels:
        label_name = labels[chr(key)]
        file_path = os.path.join(base_dir, label_name, f"{label_name}_{cv2.getTickCount()}.jpg")
        cv2.imwrite(file_path, frame)
        print(f"Captured and saved image to {file_path}")

cap.release()
cv2.destroyAllWindows()
