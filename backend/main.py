from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from datetime import datetime
import face_recognition
import mediapipe as mp
from deepface import DeepFace
import dlib
from keras.models import load_model
from ultralytics import YOLO
import json
from typing import Dict
import base64

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("project\\backend\\shape_predictor_68_face_landmarks.dat")
face_rec_model = dlib.face_recognition_model_v1("project\\backend\\dlib_face_recognition_resnet_model_v1.dat")
emotion_model = load_model("project\\backend\\Emotion_Detection.h5")
yolo_model = YOLO('project\\backend\\yolov8n.pt')
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

known_face_encodings = []
known_face_names = []

image_of_person = face_recognition.load_image_file("project\\backend\\real_71.jpg")
person_encoding = face_recognition.face_encodings(image_of_person)[0]
known_face_encodings.append(person_encoding)
known_face_names.append("Person's Name")

def convert_numpy_types(obj):
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def eye_aspect_ratio(eye):
    eye = [(point.x, point.y) for point in eye]
    A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
    B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
    C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
    return (A + B) / (2.0 * C)

def detect_dark_surroundings(frame, face_box):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    x, y, w, h = face_box
    face_region = gray[y:y+h, x:x+w]
    face_brightness = np.mean(face_region)
    return face_brightness > brightness + 30

def analyze_texture(face_region):
    gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray_face, cv2.CV_64F)
    variance = laplacian.var()
    return variance < 25

def detect_smile(face_region):
    try:
        analysis = DeepFace.analyze(face_region, actions=['emotion'], enforce_detection=False)
        if isinstance(analysis, list):
            analysis = analysis[0]
        return analysis.get('dominant_emotion') == 'happy'
    except:
        return False

def calculate_eyebrow_movement(landmarks, frame_height):
    LEFT_EYEBROW = [65, 66, 70, 63, 105]
    LEFT_EYE = [159, 145]
    eyebrow_y = sum([landmarks[i].y for i in LEFT_EYEBROW]) / len(LEFT_EYEBROW)
    eye_y = sum([landmarks[i].y for i in LEFT_EYE]) / len(LEFT_EYE)
    return (eye_y - eyebrow_y) * frame_height

def mark_attendance(name):
    with open("attendance.txt", "a") as file:
        file.write(f"{name} - {datetime.now()}\n")
    print(f"Attendance marked for {name}")

def compare_faces_and_return_boolean(frame, target_image_path, face_location):
    try:
        face_encoding = face_recognition.face_encodings(frame, [face_location])[0]
    except IndexError:
        return False

    try:
        known_image = face_recognition.load_image_file(target_image_path)
        known_face_encoding = face_recognition.face_encodings(known_image)[0]
    except IndexError:
        return False

    matches = face_recognition.compare_faces([known_face_encoding], face_encoding)
    return True in matches

@app.post("/api/process-frame")
async def process_frame(file: UploadFile = File(...)) -> Dict:
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.COLOR_BGR2RGB)
        
        results = {
            "faces_detected": 0,
            "phone_detected": False,
            "dark_surroundings": False,
            "suspicious_texture": False,
            "smile_detected": False,
            "blink_detected": False,
            "eyebrow_movement": False,
            "emotion": None,
            "identity_verified": False,
            "attedence_Marked":False,
            "timestamp": datetime.now().isoformat()
        }

        yolo_results = yolo_model(frame)
        for detection in yolo_results[0].boxes:
            if int(detection.cls) == 67:
                results["phone_detected"] = True
                break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
        results["faces_detected"] = len(faces)

        for (x, y, w, h) in faces:
            face_region = frame[y:y+h, x:x+w]
            
            results["dark_surroundings"] = detect_dark_surroundings(frame, (x, y, w, h))
            results["suspicious_texture"] = analyze_texture(face_region)
            results["smile_detected"] = detect_smile(face_region)
            
            dlib_rect = dlib.rectangle(x, y, x+w, y+h)
            landmarks = predictor(gray, dlib_rect)
            
            frame_count=0
            left_eye = [landmarks.part(i) for i in range(36, 42)]
            right_eye = [landmarks.part(i) for i in range(42, 48)]
            
            left_eye_ear = eye_aspect_ratio(left_eye)
            right_eye_ear = eye_aspect_ratio(right_eye)

            if left_eye_ear < 0.25 or right_eye_ear < 0.25:
                results["blink_detected"] = True
                frame_count = 0
            else:
                frame_count += 1
                if frame_count > 10:
                    results["blink_detected"] = False

            face_resized = cv2.resize(gray[y:y+h, x:x+w], (48, 48))
            face_resized = face_resized.reshape(1, 48, 48, 1).astype('float32') / 255.0
            emotion_pred = emotion_model.predict(face_resized)
            results["emotion"] = emotion_labels[np.argmax(emotion_pred)]

            face_encoding = face_recognition.face_encodings(frame, [(y, x+w, y+h, x)])[0]
            if len(known_face_encodings) > 0:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                if True in matches:
                    results["identity_verified"] = True
            
            is_match = compare_faces_and_return_boolean(frame, "project\\backend\\real_71.jpg", (y, x+w, y+h, x))
            if is_match:
                results["identity_verified"] = True

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            for face in faces:
                landmarks = predictor(gray, face)
                face_descriptor = face_rec_model.compute_face_descriptor(frame, landmarks)
                distances = [np.linalg.norm(face_descriptor - known_face_encoding) for known_face_encoding in known_face_encodings]
                min_distance_index = np.argmin(distances)

                if distances[min_distance_index] < 0.6:
                    name = known_face_names[min_distance_index]
                else:
                    name = "Unknown"

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mesh_results = mp_face_mesh.process(rgb_frame)
        if mesh_results.multi_face_landmarks:
            for face_landmarks in mesh_results.multi_face_landmarks:
                eyebrow_movement = calculate_eyebrow_movement(face_landmarks.landmark, frame.shape[0])
                results["eyebrow_movement"] = eyebrow_movement > 5.0

        if (results["identity_verified"] and not results["dark_surroundings"] and not results["suspicious_texture"] and results["blink_detected"] and results["eyebrow_movement"] and not results["phone_detected"] and results["smile_detected"]):
                mark_attendance(name)
                results["attedence_Marked"] = True
        else:
             results["attendance_Marked"] = False
            
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        results["processed_frame"] = frame_base64

        results = {k: convert_numpy_types(v) for k, v in results.items()}
        
        return results
    
    except Exception as e:
        print(f"Error processing frame: {str(e)}")
        raise

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
