import cv2
import matplotlib.pyplot as plt
import keyboard  # You'll need to install the 'keyboard' library

# Load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the webcam
cap = cv2.VideoCapture(0)  # 0 corresponds to the default camera (usually your built-in webcam)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # (0, 255, 0) is the color, and 2 is the line thickness

    # Convert the OpenCV BGR frame to RGB for matplotlib
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Display the frame with face detections using matplotlib
    plt.imshow(frame_rgb)
    plt.title("Face Detection")
    plt.axis('off')  # Hide axes for a cleaner display
    plt.pause(0.1)  # Display the image for a short time

    # Exit the loop when the 'q' key is pressed
    if keyboard.is_pressed('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
