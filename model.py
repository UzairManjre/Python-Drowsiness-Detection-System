-# porperty of Uzair Manjre

import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


# Function to load images from a folder
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images

# Assuming your folders are in the same directory as your script
open_eye_images = load_images_from_folder('train/Open_Eyes')
closed_eye_images = load_images_from_folder('train/Closed_Eyes')

# Assuming your images have the same dimensions, resize them if needed
image_size = (64, 64)
open_eye_images = [cv2.resize(img, image_size) for img in open_eye_images]
closed_eye_images = [cv2.resize(img, image_size) for img in closed_eye_images]

# Create labels for your data (1 for open eyes, 0 for closed eyes)
open_eye_labels = np.ones(len(open_eye_images))
closed_eye_labels = np.zeros(len(closed_eye_images))

# Concatenate the data and labels
X = np.array(open_eye_images + closed_eye_images)
y = np.concatenate([open_eye_labels, closed_eye_labels])

# Normalize pixel values to be between 0 and 1
X = X / 255.0

# Shuffle the data
shuffle_indices = np.random.permutation(len(X))
X = X[shuffle_indices]
y = y[shuffle_indices]

# Assuming image_size is the size of your resized images
input_size = image_size[0] * image_size[1] * 3

# Split the data into training and validation sets
split = int(0.8 * len(X))
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

X_test, y_test = X[split:], y[split:]
# Create the model


model = tf.keras.Sequential([
    layers.Flatten(input_shape=(image_size[0], image_size[1], 3)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=60, validation_data=(X_val, y_val))


# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

# Save the trained model
model.save('my_model.keras')

# Load the saved model later
loaded_model = tf.keras.models.load_model('my_model.keras')

# Now you can use loaded_model for making predictions or further training
