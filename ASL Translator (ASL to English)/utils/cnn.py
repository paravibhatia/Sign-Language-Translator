import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from preproc import train_data, val_data, test_data

NUM_CLASSES = len(train_data.class_indices)  # Should return 55
INPUT_SHAPE = (100, 100, 3)  # Ensure correct format for RGB images

# Print the class count for verification
print("✅ Number of Classes in Training Data:", NUM_CLASSES)

# ✅ Define the CNN Model
def create_asl_model(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES):
    model = Sequential()

    # 🟢 Convolutional Layer 1
    model.add(Conv2D(32, (3,3), activation='relu', padding='same', input_shape=input_shape))
    model.add(BatchNormalization())  
    model.add(MaxPooling2D(pool_size=(2,2)))  

    # 🟢 Convolutional Layer 2
    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))

    # 🟢 Convolutional Layer 3
    model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))

    # 🟢 Convolutional Layer 4
    model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))

    # 🟢 Flatten Layer
    model.add(Flatten())

    # 🟢 Fully Connected Layers
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))  
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))  

    # 🟢 Output Layer - Softmax for 55 Classes
    model.add(Dense(num_classes, activation='softmax'))  

    return model

# ✅ Create and Compile the Model
model = create_asl_model()

model.compile(optimizer=Adam(learning_rate=0.0005),  # Slightly reduced learning rate for stability
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Print Model Summary
model.summary()

#Model training and saving
history = model.fit(train_data, validation_data=val_data, epochs=30, batch_size=32)
model.save("asl_cnn_model_55_classes.h5")

#Evaluate the Model on Test Data
loss, accuracy = model.evaluate(test_data)
print(f" Test Accuracy: {accuracy * 100:.2f}%")
