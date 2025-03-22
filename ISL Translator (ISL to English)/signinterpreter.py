import cv2
import time
import mediapipe as mp

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize text processing variables
buffer = ""
word_list = []
last_char_time = time.time()

# Function to recognize characters (Replace with your actual sign recognition model)
def model(image):
    """
    Replace this function with your actual sign recognition model.
    It should return a character from 'A-Z' or '0-9' or special gestures 'SPACE' or 'DELETE'.
    """
    model = "A"  # Replace this with your model's output logic

    # Simulating model output for testing
    import random
    possible_chars = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789") + ["SPACE", "DELETE"]
    model = random.choice(possible_chars)  

    return model

# Function to process text dynamically
def process_text(predicted_char):
    global buffer, word_list, last_char_time

    current_time = time.time()

    # If no input for 2 seconds, finalize the word
    if current_time - last_char_time > 2 and buffer:
        word_list.append(buffer)
        buffer = ""

    # Handle special gestures
    if predicted_char == "SPACE":
        word_list.append(buffer)  # Store current word
        buffer = ""  # Reset buffer for new word
    elif predicted_char == "DELETE":
        buffer = buffer[:-1]  # Remove last character
    else:
        buffer += predicted_char

    last_char_time = current_time
    return " ".join(word_list) + " " + buffer  # Display live text

# OpenCV Video Capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame for mirror effect
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Convert to RGB for Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    predicted_char = recognize_sign(frame)  # Integrate actual recognition logic
    text_display = process_text(predicted_char)

    # Draw text on the frame
    cv2.putText(frame, text_display, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show the frame
    cv2.imshow("Sign Language Interpreter", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
