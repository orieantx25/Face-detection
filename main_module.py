# face_detection_module.py
import cv2
import mediapipe as mp
import os

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
drawSpec = mpDraw.DrawingSpec(color=(0, 255, 0),thickness=1, circle_radius=1)

face_cascade = cv2.CascadeClassifier(r"C:\Users\faiya\OneDrive\Desktop\Face detection\haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(r"C:\Users\faiya\OneDrive\Desktop\Face detection\haarcascade_eye.xml")

data_folder = r"C:\Users\faiya\OneDrive\Desktop\Face detection\data"
os.makedirs(data_folder, exist_ok=True)

def detect_landmarks(frame, face_id):
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    face_info = []
    if results.multi_face_landmarks:
        for _, faceLms in enumerate(results.multi_face_landmarks):
            mpDraw.draw_landmarks(frame, faceLms, mpFaceMesh.FACEMESH_CONTOURS,
                                  drawSpec, drawSpec)
            landmarks = [(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])) for lm in faceLms.landmark]
            face_info.append({"face_id": face_id, "landmarks": landmarks})
    return frame, face_info

def detect_faces_and_eyes(frame, gray, face_id):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    face_info = []
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
        landmarks = [(x + ex, y + ey) for (ex, ey, ew, eh) in eyes]
        face_info.append({"face_id": face_id, "landmarks": landmarks})
        cv2.imwrite(os.path.join(data_folder, f"detected_face_{face_id}_{len(face_info)}.jpg"), roi_color)
    return frame, face_info

def process_video():
    video = cv2.VideoCapture(0)
    all_faces_info = []
    face_id_counter = 0

    while True:
        _, frame = video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect and draw facial landmarks
        frame, landmarks_info = detect_landmarks(frame, face_id_counter)
        all_faces_info.extend(landmarks_info)

        # Detect and draw faces and eyes
        frame, faces_info = detect_faces_and_eyes(frame, gray, face_id_counter)
        all_faces_info.extend(faces_info)

        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Save faces and landmarks information to a file
    with open(os.path.join(data_folder, "faces_info.txt"), "w") as f:
        for info in all_faces_info:
            f.write(f"Face ID: {info['face_id']}, Landmarks: {info['landmarks']}\n")

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_video()
