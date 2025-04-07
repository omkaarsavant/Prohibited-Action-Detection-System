import time
import threading
from collections import deque
import cv2
import os
import torch
import pickle
import pandas as pd
import numpy as np
import cvzone
import xgboost as xgb
from facenet_pytorch import MTCNN, InceptionResnetV1
from scipy.spatial.distance import cosine
from ultralytics import YOLO
from flask import Flask, render_template, Response, request, jsonify
from flask_cors import CORS
import pygame  # For audio handling
import keyboard  # For keyboard input

# ------------------ CONFIGURATION ------------------
SAVE_PATH = "SuspiciousFrames"
ALERT_SOUND = os.path.abspath("buzzer.mp3")  # Use absolute path
ALERT_THRESHOLD = 0.7
TIME_WINDOW = 15

os.makedirs(SAVE_PATH, exist_ok=True)

# ------------------ INITIALIZE PYGAME MIXER ------------------
pygame.mixer.init()
pygame.mixer.music.set_volume(0.5)

# Verify audio file exists
if not os.path.exists(ALERT_SOUND):
    raise FileNotFoundError(f"Alert sound file not found at: {ALERT_SOUND}")

# ------------------ DEVICE SETUP ------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)
face_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
action_model = YOLO('yolo11s-pose.pt')

# Load XGBoost model
xgb_model = xgb.Booster()
xgb_model.load_model('Models/trained_model.json')

# Load known face embeddings
EMBEDDINGS_FILE = 'Models/known_faces_embeddings.pkl'

# ------------------ GLOBAL VARIABLES ------------------
frame_timestamps = deque()
suspicious_count = 0
total_frames = 0
alert_active = False
manual_stop = False  # New flag to track manual stop
stop_alert = threading.Event()

# ------------------ HELPER FUNCTIONS ------------------
def load_known_faces():
    """Load stored embeddings of known faces."""
    if os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE, 'rb') as f:
            known_faces = pickle.load(f)
            for face in known_faces:
                face['embeddings'] = [np.array(embedding) for embedding in face['embeddings']]
            return known_faces
    return []

def get_cosine_similarity(embedding1, embedding2):
    """Compute cosine similarity between two embeddings."""
    if embedding1.size == 0 or embedding2.size == 0:
        return float('inf')
    return 1 - cosine(embedding1.flatten(), embedding2.flatten())

def alert_controller():
    """Control alert sound playback with stop capability."""
    global alert_active, manual_stop
    while not stop_alert.is_set():
        if manual_stop:  # Exit if manually stopped
            break
        alert_active = True
        pygame.mixer.music.load(ALERT_SOUND)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy() and not stop_alert.is_set():
            time.sleep(0.1)
    pygame.mixer.music.stop()  # Ensure the music stops when the alert is cleared
    alert_active = False

def keyboard_listener():
    """Listen for the 'J' key to stop the alert manually."""
    global manual_stop, alert_active, suspicious_count, total_frames
    while True:
        if keyboard.is_pressed('j'):
            stop_alert.set()  # Signal to stop the alert
            manual_stop = True  # Set manual stop flag
            alert_active = False  # Reset alert state
            suspicious_count = 0  # Reset suspicious count
            total_frames = 0  # Reset total frames
            break

# ------------------ FLASK APP ------------------
app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_alert', methods=['POST'])
def stop_alert_route():
    stop_alert.set()
    pygame.mixer.music.stop()
    return jsonify({"status": "Alert stopped"})

# ------------------ VIDEO PROCESSING ------------------
def generate_frames():
    global suspicious_count, total_frames, alert_active, manual_stop
    cap = cv2.VideoCapture(0)
    known_faces = load_known_faces()
    alert_thread = None

    # Start the keyboard listener in a separate thread
    keyboard_thread = threading.Thread(target=keyboard_listener, daemon=True)
    keyboard_thread.start()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ------------------ FACE RECOGNITION ------------------
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, _ = mtcnn.detect(frame_rgb)

        if boxes is not None:
            faces = mtcnn(frame_rgb)  # Detect faces
            for i, face in enumerate(faces):
                embedding = face_model(face.unsqueeze(0))
                matched = False

                for known_face in known_faces:
                    name = known_face['name']
                    stored_embeddings = known_face['embeddings']
                    for stored_embedding in stored_embeddings:
                        similarity = get_cosine_similarity(embedding.detach().cpu().numpy(), stored_embedding)
                        if similarity > 0.7:
                            matched = True
                            cv2.putText(frame, name, (int(boxes[i][0]), int(boxes[i][1] - 10)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                            break
                    if matched:
                        break

                if not matched:
                    cv2.putText(frame, "Unknown", (int(boxes[i][0]), int(boxes[i][1] - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (int(boxes[i][0]), int(boxes[i][1])),
                              (int(boxes[i][2]), int(boxes[i][3])), (0, 255, 0), 2)

        # ------------------ ACTION DETECTION ------------------
        results = action_model(frame, verbose=False)
        annotated_frame = results[0].plot(boxes=False)

        frame_time = time.time()
        frame_timestamps.append(frame_time)
        total_frames += 1

        for r in results:
            bound_box = r.boxes.xyxy
            conf = r.boxes.conf.tolist()
            keypoints = r.keypoints.xyn.tolist()

            for index, box in enumerate(bound_box):
                if conf[index] > 0.55:
                    x1, y1, x2, y2 = map(int, box.tolist())

                    # Prepare data for XGBoost prediction
                    keypoints_flat = {f'x{j}': keypoints[index][j][0] for j in range(len(keypoints[index]))}
                    keypoints_flat.update({f'y{j}': keypoints[index][j][1] for j in range(len(keypoints[index]))})

                    feature_order = ['x0', 'y0', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4', 
                                    'x5', 'y5', 'x6', 'y6', 'x7', 'y7', 'x8', 'y8', 'x9', 'y9', 
                                    'x10', 'y10', 'x11', 'y11', 'x12', 'y12', 'x13', 'y13', 'x14', 'y14', 
                                    'x15', 'y15', 'x16', 'y16']
                    
                    data = {key: keypoints_flat[key] for key in feature_order if key in keypoints_flat}
                    df = pd.DataFrame([data])
                    dmatrix = xgb.DMatrix(df)

                    prediction = xgb_model.predict(dmatrix)
                    binary_prediction = int(prediction > 0.5)

                    if binary_prediction == 0:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cvzone.putTextRect(frame, "Suspicious", (x1, y1), 1, 1)
                        suspicious_count += 1
                    else:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cvzone.putTextRect(frame, "Normal", (x1, y1 + 50), 1, 1)

        # ------------------ ALERT MANAGEMENT ------------------
        while frame_timestamps and frame_timestamps[0] < frame_time - TIME_WINDOW:
            frame_timestamps.popleft()
            total_frames -= 1

        suspicious_ratio = suspicious_count / total_frames if total_frames > 0 else 0

        if suspicious_ratio >= ALERT_THRESHOLD and not alert_active and not manual_stop:
            stop_alert.clear()
            alert_thread = threading.Thread(target=alert_controller)
            alert_thread.start()

        if suspicious_ratio < ALERT_THRESHOLD and alert_active:
            stop_alert.set()


        # Reset manual_stop flag when no suspicious activity is detected
        if suspicious_ratio < ALERT_THRESHOLD:
            manual_stop = False

        # Add alert status to frame only if alert is active
        if alert_active and not manual_stop:
            alert_text = "ALERT: Suspicious Activity Detected!"
            alert_color = (0, 0, 255)
            cv2.putText(frame, alert_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, alert_color, 2)

        # # Add alert status to frame
         alert_text = "ALERT: Suspicious Activity Detected!" if alert_active else "Monitoring..."
         alert_color = (0, 0, 255) if alert_active else (0, 255, 0)
         cv2.putText(frame, alert_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, alert_color, 2)


        # Save frame if alert_active:
        if alert_active:
            timestamp = int(time.time())
            cv2.imwrite(os.path.join(SAVE_PATH, f"suspicious_{timestamp}.jpg"), frame)

        # Stream the frame
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    app.run(debug=False)