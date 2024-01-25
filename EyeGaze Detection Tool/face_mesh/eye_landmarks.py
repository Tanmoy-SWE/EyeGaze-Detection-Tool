import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Start video capture (webcam)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    # Convert the image color from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and find facial landmarks
    results = face_mesh.process(image)

    # Convert the image color back to BGR to display it properly
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Draw facial landmarks
    if results.multi_face_landmarks:
        for facial_landmarks in results.multi_face_landmarks:
            # Extract eye coordinates (you can refine this part to be more precise)
            # left_eye = [facial_landmarks.landmark[i] for i in range(133, 373)]
            # right_eye = [facial_landmarks.landmark[i] for i in range(362, 373)]
            
            face = facial_landmarks.landmark

            # Implement gaze estimation logic here
            # This could involve analyzing the positions of the eye landmarks,
            # estimating the head pose, and then deriving the gaze direction.

            # For demonstration, just drawing the eye landmarks
            for point in face:
                x = int(point.x * image.shape[1])
                y = int(point.y * image.shape[0])
                cv2.circle(image, (x, y), 1, (0, 255, 0), -1)

    # Show the image
    cv2.imshow('MediaPipe Eye Coordinates', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
