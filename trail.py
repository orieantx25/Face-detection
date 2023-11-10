import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from skimage.feature import local_binary_pattern

# Load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('path/to/haarcascade_frontalface_default.xml')

# Function to extract LBP features from an image
def extract_lbp_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    radius = 3
    n_points = 24
    lbp_image = local_binary_pattern(gray, n_points, radius, method='uniform')
    hist, _ = np.histogram(lbp_image.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

# Load the dataset and labels (assuming you have a dataset of face images)
# Here, X contains face images, and y contains corresponding labels (person IDs)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Extract LBP features from training data
X_train_features = [extract_lbp_features(image) for image in X_train]

# Train a K-Nearest Neighbors classifier
knn_classifier = KNeighborsClassifier(n_neighbors=1)
knn_classifier.fit(X_train_features, y_train)

# Initialize the video capture
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Iterate over detected faces and recognize them
    for (x, y, w, h) in faces:
        face_roi = gray[y:y + h, x:x + w]  # Extract the face region
        lbp_features = extract_lbp_features(face_roi)  # Extract LBP features
        prediction = knn_classifier.predict([lbp_features])[0]  # Recognize the face

        # Draw a rectangle around the recognized face and display the person's ID
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f'Person {prediction}', (x, y - 10), font, 0.9, (255, 0, 0), 2)

    # Display the frame
    cv2.imshow('Face Recognition Attendance System', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
