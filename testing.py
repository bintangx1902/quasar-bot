import cv2
import numpy as np
from tensorflow import keras
import keras

# Load the trained model
model = keras.models.load_model('./model/v1.h5')

# Open video capture
cap = cv2.VideoCapture(0)  # Use camera index 0 for the default camera

while True:
    # Capture frame from video
    ret, frame = cap.read()

    # Preprocess frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
    frame = cv2.resize(frame, (192, 255))  # Resize to match input shape of the model
    frame = frame / 127.5 - 1
    frame = np.expand_dims(frame, axis=0)  # Add batch dimension
    frame = np.vstack([frame])

    # Make prediction
    predictions = model.predict(frame)

    # Extract class and confidence from prediction
    class_index = np.argmax(predictions)
    confidence = predictions[0][class_index]

    # Draw bounding box around detected object
    cv2.rectangle(frame, (0, 0), (255, 192), (255, 0, 0), 2)  # Example bounding box, adjust as needed

    # Display frame
    cv2.imshow('Object Detection', frame)

    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
