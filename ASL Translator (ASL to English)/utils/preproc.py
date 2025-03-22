from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import shutil

IMG_SIZE = (100, 100)  # Resize all images to 100x100
BATCH_SIZE = 32

train_dir = "dataset/train"
val_dir = "dataset/val"
test_dir = "dataset/test"

# Load dataset
train_datagen = ImageDataGenerator(rescale=1./255)  # Normalize pixel values
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')

val_data = val_datagen.flow_from_directory(
    val_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')

test_data = test_datagen.flow_from_directory(
    test_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')

print("✅ Dataset successfully loaded!")
print("Number of Classes in Training Data:", len(train_data.class_indices))
print("Class Labels:", train_data.class_indices)



