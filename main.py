import cv2
import cvzone
import os
import pickle
import face_recognition
import numpy as np



# Initialize the webcam capture with camera index 1 (or 0 for the default camera)
cap = cv2.VideoCapture(0)

# Set the resolution for the webcam capture
cap.set(3, 640)
cap.set(4, 480)

# Load the 'background.png' image from the 'Resources' folder
imgBackground = cv2.imread('Resources/background.png')

# Importing the mode images into a list
folderModePath = 'Resources/Modes'
modePathList = os.listdir(folderModePath)
imgModeList = []
for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath, path)))
# print(len(imgModeList))

# Load the encoding file
print("Loading Encode File ...")
file = open('EncodeFile.p', 'rb')
encodeListKnownWithIds = pickle.load(file)
file.close()
encodeListKnown, studentIds = encodeListKnownWithIds
# print(studentIds)
print("Encode File Loaded")


while True:
    # Read a frame from the webcam
    success, img = cap.read()

    # Check if the webcam frame capture was successful
    if not success:
        print("Failed to capture frame from the webcam.")
        break

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)


    # Overlay the webcam frame on the background image
    imgBackground[162:162+480, 55:55+640] = img
    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[0]


    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        # print("matches", matches)
        # print("faceDis", faceDis)


        matchIndex = np.argmin(faceDis)
        print("Match Index", matchIndex)

        if matches[matchIndex]:
            print("Known Face Detected")
            print(studentIds[matchIndex])
            # y1, x2, y2, x1 = faceLoc
            # y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            # bbox = 55 + x1, 162 + y1, x2 - x1, y2 - y1
            # imgBackground = cvzone.cornerRect(imgBackground, bbox, rt=0)
            # id = studentIds[matchIndex]

    # Display the combined image
    cv2.imshow("Face Recognition", imgBackground)

    # Exit the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
