import os
import shutil
import random
import os
import shutil

# Define dataset paths
RAW_DATASET_PATH = "hand_gestures_dataset"  # Change this to where your unsplit dataset is stored
OUTPUT_DIR = "dataset"  # Where train/val/test will be stored

# Set dataset split ratios
TRAIN_RATIO = 0.8  # 80% Training
VAL_RATIO = 0.1    # 10% Validation
TEST_RATIO = 0.1   # 10% Testing

# Ensure output directories exist
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)

# Get all class folders (A-Z, 0-9, etc.)
classes = os.listdir(RAW_DATASET_PATH)

for class_name in classes:
    class_path = os.path.join(RAW_DATASET_PATH, class_name)
    if not os.path.isdir(class_path):
        continue

    # Get all images in the class folder
    images = os.listdir(class_path)
    random.shuffle(images)  # Shuffle dataset to randomize selection

    # Split data into Train, Val, and Test
    train_split = int(len(images) * TRAIN_RATIO)
    val_split = int(len(images) * (TRAIN_RATIO + VAL_RATIO))

    train_images = images[:train_split]
    val_images = images[train_split:val_split]
    test_images = images[val_split:]

    # Create class folders in train/val/test directories
    for split, image_list in zip(["train", "val", "test"], [train_images, val_images, test_images]):
        split_class_dir = os.path.join(OUTPUT_DIR, split, class_name)
        os.makedirs(split_class_dir, exist_ok=True)

        # Move images to their respective directories
        for img_name in image_list:
            src_path = os.path.join(class_path, img_name)
            dst_path = os.path.join(split_class_dir, img_name)
            shutil.copy(src_path, dst_path)

print("✅ Dataset successfully split into train/val/test folders!")
