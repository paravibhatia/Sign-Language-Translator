import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from preproc import train_data  # Import class labels

# Load the trained ASL CNN model
model = load_model("asl_cnn_model.h5")

# Load class labels
class_labels = list(train_data.class_indices.keys())

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                        min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Function to preprocess a cropped hand image
def preprocess_hand(roi):
    img = cv2.resize(roi, (100, 100))  # Resize to model input shape
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Expand dimensions for model
    return img

# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ ERROR: Could not access the webcam.")
    exit()

print("✅ Webcam is running. Show an ASL gesture to predict...")

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("❌ ERROR: Could not read frame from webcam.")
        break

    frame = cv2.flip(frame, 1)  # Flip horizontally for natural mirroring
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB (MediaPipe requirement)

    # Detect hands
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the screen
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get bounding box coordinates
            h, w, _ = frame.shape
            x_min, y_min = w, h
            x_max, y_max = 0, 0

            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * w), int(landmark.y * h)
                x_min = min(x, x_min)
                y_min = min(y, y_min)
                x_max = max(x, x_max)
                y_max = max(y, y_max)

            # Expand bounding box for better cropping
            padding = 30
            x_min, y_min = max(0, x_min - padding), max(0, y_min - padding)
            x_max, y_max = min(w, x_max + padding), min(h, y_max + padding)

            # Crop the hand region
            hand_roi = frame[y_min:y_max, x_min:x_max]

            # Ensure the hand ROI is valid before processing
            if hand_roi.shape[0] > 0 and hand_roi.shape[1] > 0:
                processed_hand = preprocess_hand(hand_roi)

                # Make prediction
                prediction = model.predict(processed_hand)
                predicted_label = class_labels[np.argmax(prediction)]  # Get predicted class

                # Display prediction on the screen
                cv2.putText(frame, f"Prediction: {predicted_label}", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Draw bounding box

    # Show the video feed with prediction
    cv2.imshow("ASL Gesture Recognition (MediaPipe + CNN)", frame)

    # Press 'q' to exit the webcam window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close all windows
cap.release()
cv2.destroyAllWindows()
print("✅ Webcam closed. ASL prediction ended.")
