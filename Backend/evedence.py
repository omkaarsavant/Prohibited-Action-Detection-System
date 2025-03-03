import time
from collections import deque
import cv2
import os
import cv2
import torch
import pickle
import pandas as pd
import numpy as np
import cvzone
import xgboost as xgb
from facenet_pytorch import MTCNN, InceptionResnetV1
from scipy.spatial.distance import cosine
from ultralytics import YOLO
from flask import Flask, render_template, Response, request, jsonify, send_from_directory
from store_embeddings_multiple import store_face_embeddings_from_images
from flask_cors import CORS
from flask_pymongo import PyMongo
from werkzeug.utils import secure_filename
from PIL import Image as PILImage
import socket
# Store timestamps of frames to maintain a rolling 15-second window
frame_timestamps = deque()
suspicious_count = 0
total_frames = 0
ALERT_THRESHOLD = 0.7  # 70% threshold
TIME_WINDOW = 15  # 15 seconds
SAVE_PATH = "SuspiciousFrames"

os.makedirs(SAVE_PATH, exist_ok=True)  # Ensure directory exists


# Initialize MTCNN for face detection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)
face_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Initialize YOLOv8 for action detection
action_model = YOLO('yolo11s-pose.pt')

# Load XGBoost model for classification
xgb_model = xgb.Booster()
xgb_model.load_model('Models/trained_model.json')

# Path to store the embeddings of known faces
EMBEDDINGS_FILE = 'Models/known_faces_embeddings.pkl'

# Load known faces
def load_known_faces():
    if os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE, 'rb') as f:
            known_faces = pickle.load(f)
            for face in known_faces:
                face['embeddings'] = [np.array(embedding) for embedding in face['embeddings']]
            return known_faces
    return []

# Compare embeddings using cosine similarity
def get_cosine_similarity(embedding1, embedding2):
    if embedding1.size == 0 or embedding2.size == 0:
        return float('inf')
    return 1 - cosine(embedding1.flatten(), embedding2.flatten())

# Process video stream
def generate_frames():
    print("Generating")
    global suspicious_count, total_frames
    cap = cv2.VideoCapture(0)  # Webcam
    known_faces = load_known_faces()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, _ = mtcnn.detect(frame_rgb)

        # Face Recognition
        if boxes is not None:
            faces = mtcnn(frame_rgb)
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

        # Action Detection
        results = action_model(frame, verbose=False)
        annotated_frame = results[0].plot(boxes=False)

        frame_time = time.time()
        frame_timestamps.append(frame_time)
        total_frames += 1  # Count total frames

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

                    # Ensure feature order matches model expectations
                    feature_order = ['x0', 'y0', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4', 
                                     'x5', 'y5', 'x6', 'y6', 'x7', 'y7', 'x8', 'y8', 'x9', 'y9', 
                                     'x10', 'y10', 'x11', 'y11', 'x12', 'y12', 'x13', 'y13', 'x14', 'y14', 
                                     'x15', 'y15', 'x16', 'y16']
                    
                    data = {key: keypoints_flat[key] for key in feature_order if key in keypoints_flat}
                    df = pd.DataFrame([data])

                    dmatrix = xgb.DMatrix(df)

                    # Predict suspicious behavior
                    prediction = xgb_model.predict(dmatrix)
                    binary_prediction = int(prediction > 0.5)

                    if binary_prediction == 1:  # Suspicious
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cvzone.putTextRect(frame, "Suspicious", (x1, y1), 1, 1)
                        suspicious_count += 1  # Increment suspicious frame count
                    else:  # Normal
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cvzone.putTextRect(frame, "Normal", (x1, y1 + 50), 1, 1)

        # Remove old frames from queue (older than 15 seconds)
        while frame_timestamps and frame_timestamps[0] < frame_time - TIME_WINDOW:
            frame_timestamps.popleft()
            total_frames -= 1  # Decrease total frame count accordingly

        # Calculate percentage of suspicious frames
        if total_frames > 0:
            suspicious_ratio = suspicious_count / total_frames
        else:
            suspicious_ratio = 0

        # If 70% of frames are suspicious, raise a flag and save frames
        if suspicious_ratio >= ALERT_THRESHOLD:
            cv2.putText(frame, "ALERT: Suspicious Activity Detected!", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            timestamp = int(time.time())
            frame_filename = os.path.join(SAVE_PATH, f"suspicious_{timestamp}.jpg")
            cv2.imwrite(frame_filename, frame)  # Save frame

        # Convert frame for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()
