import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
from preproc import train_data  # Import class labels

# Load Trained Model
model = load_model("asl_cnn_model.h5")

# Load Class Labels
class_labels = list(train_data.class_indices.keys())  # Get label names

# Function to preprocess an image before passing it to the model
def preprocess_image(img_path):
    # Check if the file exists before loading
    if not os.path.exists(img_path):
        print(f"❌ ERROR: The file '{img_path}' does not exist. Check the path!")
        exit()

    img = cv2.imread(img_path)  # Read the image

    # Check if the image was loaded correctly
    if img is None:
        print(f"❌ ERROR: Could not read image at '{img_path}'. Make sure it's a valid image file.")
        exit()

    img = cv2.resize(img, (100, 100))  # Resize to match model input
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Expand dimensions for model
    return img

# ✅ Use Corrected File Path
image_path = r"C:\Users\HP\OneDrive\Documents\GitHub\hand-gesture-recognition-mediapipe\dataset\2.jpg"

# Preprocess the Image
img = preprocess_image(image_path)

# Make Prediction
prediction = model.predict(img)
predicted_label = class_labels[np.argmax(prediction)]  # Get predicted class

print(f"✅ Predicted Sign: {predicted_label}")

# Show Image with Prediction
img_display = cv2.imread(image_path)
cv2.putText(img_display, f"Prediction: {predicted_label}", (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv2.imshow("ASL Sign Prediction", img_display)
cv2.waitKey(0)
cv2.destroyAllWindows()
