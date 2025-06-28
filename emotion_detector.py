import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Initialize face detection classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Emotion labels
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def preprocess_face(face_img):
    # Resize to 48x48 and convert to grayscale
    face_img = cv2.resize(face_img, (48, 48))
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_img = img_to_array(face_img)
    face_img = np.expand_dims(face_img, axis=0)
    face_img = face_img / 255.0
    return face_img

def main():
    # Load the trained model
    try:
        model = load_model('emotion_model.h5')
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error: Could not load model. {e}")
        return

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return

    # Check if Haar cascade is loaded
    if face_cascade.empty():
        print("Error: Could not load Haar cascade for face detection.")
        return

    print("Starting emotion detection...")
    print("Press 'q' to quit")

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # Process each face
        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Extract face ROI
            face_roi = frame[y:y+h, x:x+w]

            # Preprocess face
            processed_face = preprocess_face(face_roi)

            # Predict emotion
            try:
                predictions = model.predict(processed_face)
                emotion = emotions[np.argmax(predictions)]
            except Exception as e:
                print(f"Error during prediction: {e}")
                continue

            # Display emotion text
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # Display the resulting frame
        cv2.imshow('Emotion Detection', frame)

        # Break the loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release everything when done
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()