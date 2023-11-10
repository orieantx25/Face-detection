import cv2
import mediapipe as mp
import os

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

data_folder = r"C:\Users\faiya\OneDrive\Desktop\Face detection\data"
os.makedirs(data_folder, exist_ok=True)

def register_face(face_id, landmarks):
    with open(os.path.join(data_folder, f"face_{face_id}.txt"), "w") as f:
        for i, (x, y) in enumerate(landmarks):
            f.write(f"{i}\t{x}\t{y}\n")

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for face_id, faceLms in enumerate(results.multi_face_landmarks):
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS,
                                  drawSpec, drawSpec)
            
            # Convert landmarks to a list of tuples (x, y)
            landmarks = [(int(lm.x * img.shape[1]), int(lm.y * img.shape[0])) for lm in faceLms.landmark]
            
            # Register face with ID
            register_face(face_id, landmarks)
            
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
