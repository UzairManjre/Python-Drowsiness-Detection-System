# porperty of Uzair Manjre
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from collections import deque
from playsound import playsound
import threading

# Load the trained model
model = tf.keras.models.load_model('my_model.keras')

# Load the Haar Cascade classifiers for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Define the function to preprocess the input image
def preprocess_image(img):
    # Assuming your images have the same dimensions as the training images
    image_size = (64, 64)
    img = cv2.resize(img, image_size)
    img = img / 255.0
    return np.expand_dims(img, axis=0)

def play_alert_sound():
    playsound("sounds/beep-warning-6387.mp3")

# Access the camera
cap = cv2.VideoCapture(0)  # Use 0 for default webcam, change the index if you have multiple cameras
# cap.set(3, 640)  # Width
# cap.set(4, 480)  # Height
# test the fps of the camera feedback
fps = cap.get(cv2.CAP_PROP_FPS)
print("Frames per second:", fps)




# Parameters for mean close eye count
mean_close_eye_count_threshold = 10  # Adjust as needed
mean_close_eye_count = deque(maxlen=mean_close_eye_count_threshold)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4)



    for (x, y, w, h) in faces:
        # Extract the region of interest (ROI) around the detected face
        face_roi = frame[y:y + h, x:x + w]

        # Convert the face ROI to grayscale for eye detection
        face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

        # Detect eyes within the face ROI
        eyes = eye_cascade.detectMultiScale(face_gray)

        # Print the detected face coordinates
        print(f"Detected Face: {x}, {y}, {w}, {h}")

        # Detect eyes within the face ROI
        eyes = eye_cascade.detectMultiScale(face_gray)

        # Auto-zoom based on detected eyes or consider closed eyes if no eyes are detected
        if len(eyes) > 0:
            for (ex, ey, ew, eh) in eyes:
                # Define the region of interest (ROI) around the detected eyes
                eye_roi = face_roi[ey:ey + eh, ex:ex + ew]

                # Preprocess the eye ROI for prediction
                preprocessed_eye_roi = preprocess_image(eye_roi)

                # Make a prediction
                prediction = model.predict(preprocessed_eye_roi)

                # Assuming a threshold of 0.6 for binary classification
                if prediction[0][0] > 0.5:
                    cv2.putText(frame, 'Open Eyes', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    mean_close_eye_count.clear()  # Reset the mean close eye count
                else:
                    cv2.putText(frame, 'Closed Eyes', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                                cv2.LINE_AA)
                    mean_close_eye_count.append(1)  # Increment the mean close eye count

                    # Check if the mean close eye count exceeds the threshold
                    if len(mean_close_eye_count) >= mean_close_eye_count_threshold:
                        # Play the alert sound or trigger your alert mechanism
                        cv2.putText(frame, 'ALERT: Drowsiness detected!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                                    cv2.LINE_AA)
                        # Play the alert sound in a separate thread
                        threading.Thread(target=play_alert_sound).start()
                        print("ALERT: Drowsiness detected!")
                        # You can add your alert sound or any other alert mechanism here

        else:
            # No eyes detected within the face ROI, consider as closed eyes
            cv2.putText(frame, 'Closed Eyes', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            mean_close_eye_count.append(1)  # Increment the mean close eye count

            # Check if the mean close eye count exceeds the threshold
            if len(mean_close_eye_count) >= mean_close_eye_count_threshold:
                # Play the alert sound or trigger your alert mechanism
                # Play the alert sound in a separate thread
                threading.Thread(target=play_alert_sound).start()
                print("ALERT: Drowsiness detected!")
                # You can add your alert sound or any other alert mechanism here

    # Display the result
    cv2.imshow('Drowsiness Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera
cap.release()
cv2.destroyAllWindows()
