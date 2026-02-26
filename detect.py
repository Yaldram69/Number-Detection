import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('mnistv2.h5')

def preprocess_image(image, bbox):
    # Crop the region of interest (ROI) from the frame based on the bounding box
    x, y, w, h = bbox
    roi = image[y:y+h, x:x+w]
    
    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Resize the image to 28x28 pixels (the size expected by MNIST)
    resized = cv2.resize(gray, (128, 128))
    
    # Normalize the pixel values to the range [0, 1]
    normalized = resized.astype('float32') / 255.0
    
    # Reshape to add batch dimension (1, 28, 28, 1) for the model
    preprocessed = np.reshape(normalized, (1, 128, 128, 1))
    
    return preprocessed

# Open video capture (webcam)
cap = cv2.VideoCapture(0)

# Bounding box coordinates (where you should place your handwritten digit)
bbox = (100, 100, 200, 200)  # (x, y, width, height)

while True:
    # Capture frame-by-frame from the webcam
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Draw a bounding box rectangle where the digit should be placed
    x, y, w, h = bbox
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)  # Blue rectangle with thickness 2
    
    # Preprocess the region inside the bounding box for prediction
    preprocessed_image = preprocess_image(frame, bbox)
    
    # Make prediction using the model
    predictions = model.predict(preprocessed_image)
    
    # Get the predicted class (digit) from the model's output
    predicted_class = np.argmax(predictions, axis=1)[0]
    
    # Display the predicted digit on the frame
    cv2.putText(frame, f'Predicted Digit: {predicted_class}', (x, y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Show the frame with the bounding box and prediction
    cv2.imshow('Real-Time Digit Recognition', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
