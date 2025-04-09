import cv2
import numpy as np
import pickle

# -----------------------------
# CONFIGURATION AND SETUP
# -----------------------------
# Path to your pre-trained classifier (trained on static images for these 3 signals)
MODEL_PATH = 'classifier_model.pkl'
# Video file containing the marshaling signals
VIDEO_PATH = 'marshaling_video.mp4'

# Mapping classifier output labels to the signal names
# Note: Here we use class numbers 2, 3, and 4 to indicate the following:
# 2: Straight ahead, 3: Turn left, 4: Turn right.
label_dict = {2: "Straight ahead", 3: "Turn left", 4: "Turn right"}

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------

def extract_features(frame):
    """
    Placeholder function to convert a video frame into a feature vector
    expected by the classifier.

    Depending on your pre-processing pipeline, this might involve:
    - Resizing and normalizing the frame.
    - Detecting and cropping a region-of-interest (ROI) (e.g., where the arms are)
    - Extracting landmarks or computing pose angles.
    - Flattening or summarizing the above into a 1D feature vector.
    
    For illustration, we'll simply convert the frame to grayscale,
    resize it to a fixed size, and then flatten it.
    """
    # Convert frame to grayscale (optional, depends on your model input)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Resize frame (adjust size according to your model training)
    resized_frame = cv2.resize(gray_frame, (64, 64))
    
    # Normalize the image
    normalized_frame = resized_frame / 255.0
    
    # Flatten the image to create a feature vector
    feature_vector = normalized_frame.flatten()
    
    return feature_vector

def load_classifier(model_path):
    """Load the pre-trained classifier from disk."""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

# -----------------------------
# MAIN PROCESSING PIPELINE
# -----------------------------
def process_video(video_path, classifier):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Process each frame until the video is completed or interrupted
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Optionally: sample every N frames to reduce processing time,
        # e.g., using a frame counter if your frame rate is very high.

        # Extract features from the current frame
        features = extract_features(frame)
        
        # The classifier expects a 2D array (n_samples, n_features)
        features = features.reshape(1, -1)
        
        # Make prediction with the classifier
        prediction = classifier.predict(features)[0]
        signal_text = label_dict.get(prediction, "Unknown")

        # Display the prediction on the video frame
        cv2.putText(frame, f"Signal: {signal_text}", (30, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Marshaling Signal Detection", frame)

        # Press 'q' key to exit early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up: release the video capture object and close windows.
    cap.release()
    cv2.destroyAllWindows()

# -----------------------------
# SCRIPT EXECUTION
# -----------------------------
if __name__ == '__main__':
    # Load your trained classifier
    clf = load_classifier(MODEL_PATH)
    
    # Start processing the video
    process_video(VIDEO_PATH, clf)
