import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model('emotion_recognition_model.h5')

# Define label mapping
label_to_text = {0:'anger', 1:'disgust', 2:'fear', 3:'happiness', 4:'sadness', 5:'surprise', 6:'neutral'}

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
if face_cascade.empty():
    print("Error: Could not load Haar Cascade. Check OpenCV installation.")
    exit()

# Start video capture
cap = cv2.VideoCapture(0)  # 0 for default webcam

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Extract face ROI
        face_roi = gray[y:y+h, x:x+w]
        
        # Resize to 48x48
        face_resized = cv2.resize(face_roi, (48, 48), interpolation=cv2.INTER_AREA)
        
        # Normalize
        face_normalized = face_resized.astype('float32') / 255.0
        
        # Reshape for model prediction
        face_reshaped = np.reshape(face_normalized, (1, 48, 48, 1))
        
        # Predict emotion
        prediction = model.predict(face_reshaped, verbose=0)  # verbose=0 to suppress output
        emotion_label = label_to_text[np.argmax(prediction)]
        
        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Emotion Recognition', frame)

    # Break loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

print("Real-time emotion recognition stopped.")