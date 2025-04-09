import cv2
import pickle
import numpy as np

# Load the trained classifier
with open('Video Detection/models/classifier_model.pkl', 'rb') as f:
    classifier = pickle.load(f)

def extract_features_from_frame(frame, size=(64, 64)):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray_frame, size)
    normalized = resized / 255.0
    features = normalized.flatten()
    return features

cap = cv2.VideoCapture(0)  # Open the default camera

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    features = extract_features_from_frame(frame)
    features = features.reshape(1, -1)
    
    prediction = classifier.predict(features)[0]
    # Map prediction integer back to signal name
    if prediction == 2:
        signal = "Straight ahead"
    elif prediction == 3:
        signal = "Turn left"
    elif prediction == 4:
        signal = "Turn right"
    else:
        signal = "Unknown"
    
    cv2.putText(frame, f"Signal: {signal}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)
    cv2.imshow("AeroVision", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
