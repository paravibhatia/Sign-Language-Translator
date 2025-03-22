import cv2
import mediapipe as mp
import os

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Dataset Path
DATASET_PATH = "hand_gestures_dataset"

# List of Gestures
GESTURES = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z", "0", "1", "2", "3",
    "4", "5", "6", "7", "8", "9", "thumbs_up", "hi",
    "bye", "yes", "no", "i_love_you", "thank_you",
    "sorry", "please", "help", "stop", "go", "come",
    "good", "bad", "me", "you", "we", "they"
]

# Create dataset folders
for gesture in GESTURES:
    os.makedirs(os.path.join(DATASET_PATH, gesture), exist_ok=True)

# Initialize Camera
def initialize_camera():
    for i in range(2):  # Try different camera indexes (0 or 1)
        cam = cv2.VideoCapture(i)
        if cam.isOpened():
            return cam
    raise Exception("No working camera found!")

cam = initialize_camera()
gesture_index = 0  # Default gesture: 'A'

# Function to get next available image index
def get_next_image_index(gesture):
    gesture_path = os.path.join(DATASET_PATH, gesture)
    existing_files = os.listdir(gesture_path)
    return len(existing_files)  

while True:
    success, img = cam.read()
    if not success:
        print("Failed to read from camera")
        break

    img = cv2.flip(img, 1)  # Mirror image for natural feel
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process frame with MediaPipe
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get bounding box
            h, w, _ = img.shape
            x_min, y_min = w, h
            x_max, y_max = 0, 0

            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * w), int(landmark.y * h)
                x_min, y_min = min(x_min, x), min(y_min, y)
                x_max, y_max = max(x_max, x), max(y_max, y)

            # Make bounding box square (for better model consistency)
            box_width = x_max - x_min
            box_height = y_max - y_min
            box_size = max(box_width, box_height)  # Ensure square shape

            # Expand bounding box dynamically
            padding = int(box_size * 0.2)  # Add 20% padding
            x_min, y_min = max(0, x_min - padding), max(0, y_min - padding)
            x_max, y_max = min(w, x_min + box_size + padding), min(h, y_min + box_size + padding)

            # Ensure bounding box is valid
            if (x_max - x_min) > 50 and (y_max - y_min) > 50:
                hand_roi = img[y_min:y_max, x_min:x_max]

                # Draw bounding box
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                # Save Image if 's' is pressed
                key = cv2.waitKey(1) & 0xFF
                if key == ord('s'):
                    img_index = get_next_image_index(GESTURES[gesture_index])
                    filename = os.path.join(DATASET_PATH, GESTURES[gesture_index], f"{img_index}.jpg")
                    cv2.imwrite(filename, hand_roi)
                    print(f"✅ Saved: {filename}")

    # Display Current Gesture
    cv2.putText(img, f"Gesture: {GESTURES[gesture_index]} (Press 'n' to next, 'p' to previous, 's' to save)",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Hand Gesture Dataset Collector", img)

    # Switch gestures using 'n' (next) and 'p' (previous)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('n'):  
        gesture_index = (gesture_index + 1) % len(GESTURES)
        print(f"Switched to: {GESTURES[gesture_index]}")
    elif key == ord('p'):  
        gesture_index = (gesture_index - 1) % len(GESTURES)
        print(f"Switched to: {GESTURES[gesture_index]}")
    elif key == ord('q'):  
        break

cam.release()
cv2.destroyAllWindows()
