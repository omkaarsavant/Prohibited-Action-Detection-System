# import cv2
# import torch
# import pickle
# from facenet_pytorch import MTCNN, InceptionResnetV1
# from scipy.spatial.distance import cosine
# import os
# import numpy as np
# from flask import Response


# # Initialize MTCNN for face detection
# mtcnn = MTCNN(keep_all=True, device='cpu')  # Change 'cpu' to 'cuda' for GPU
# # Initialize Inception ResNet for face recognition
# model = InceptionResnetV1(pretrained='vggface2').eval().to('cpu')  # Change to 'cuda' if using GPU


# # Path to store the embeddings of known faces
# EMBEDDINGS_FILE = 'Models/known_faces_embeddings.pkl'

# # Load known faces and embeddings from file (if exists)
# def load_known_faces():
#     if os.path.exists(EMBEDDINGS_FILE):
#         with open(EMBEDDINGS_FILE, 'rb') as f:
#             known_faces = pickle.load(f)
#             # Ensure embeddings are numpy arrays
#             for face in known_faces:
#                 face['embeddings'] = [np.array(embedding) for embedding in face['embeddings']]
#             return known_faces
#     return []

# # Compare embeddings using cosine similarity
# def get_cosine_similarity(embedding1, embedding2):
#     if embedding1.size == 0 or embedding2.size == 0:
#         print("One of the embeddings is empty. Cannot compute cosine similarity.")
#         return float('inf')  # Return a large value to indicate dissimilarity
#     return 1 - cosine(embedding1.flatten(), embedding2.flatten())  # Flatten both embeddings to 1D


# def generate_frames():
#     # Start video capture (webcam)
#     cap = cv2.VideoCapture(0)

#     # Load known faces and embeddings
#     known_faces = load_known_faces()

#     # Start video stream
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Convert frame to RGB (as OpenCV loads images in BGR format)
#         img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#         # Detect faces
#         boxes, _ = mtcnn.detect(img_rgb)

#         if boxes is not None:
#             faces = mtcnn(img_rgb)  # Extract faces from the image
#             for i, face in enumerate(faces):
#                 embedding = model(face.unsqueeze(0))  # Get embedding for the detected face

#                 # Compare with known faces (from multiple angles)
#                 matched = False
#                 for known_face in known_faces:
#                     name = known_face['name']
#                     stored_embeddings = known_face['embeddings']
#                     for stored_embedding in stored_embeddings:
#                         similarity = get_cosine_similarity(embedding.detach().cpu().numpy(), stored_embedding)
#                         if similarity > 0.7:  # If similarity is above 0.7, consider it a match
#                             matched = True
#                             cv2.putText(frame, name, (int(boxes[i][0]), int(boxes[i][1] - 10)),
#                                         cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
#                             break

#                     if matched:
#                         break

#                 if not matched:
#                     cv2.putText(frame, "Unknown", (int(boxes[i][0]), int(boxes[i][1] - 10)),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

#                 # Draw bounding box around the detected face
#                 cv2.rectangle(frame, (int(boxes[i][0]), int(boxes[i][1])),
#                               (int(boxes[i][2]), int(boxes[i][3])), (0, 255, 0), 2)

#         # Encode the frame in JPEG format
#         ret, buffer = cv2.imencode('.jpg', frame)
#         frame = buffer.tobytes()

#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# def video_feed():
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')




# import cv2
# import torch
# import pickle
# from facenet_pytorch import MTCNN, InceptionResnetV1
# from scipy.spatial.distance import cosine
# import os
# import numpy as np
# import socket

# # Initialize MTCNN for face detection
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# mtcnn = MTCNN(keep_all=True, device=device)  # Use 'device' variable for GPU
# # Initialize Inception ResNet for face recognition
# model = InceptionResnetV1(pretrained='vggface2').eval().to(device)  # Use 'device' variable for GPU
# print(f"Using device: {device}") 



# # Path to store the embeddings of known faces
# EMBEDDINGS_FILE = 'known_faces_embeddings.pkl'

# # Load known faces and embeddings from file (if exists)
# def load_known_faces():
#     if os.path.exists(EMBEDDINGS_FILE):
#         with open(EMBEDDINGS_FILE, 'rb') as f:
#             known_faces = pickle.load(f)
#             # Ensure embeddings are numpy arrays
#             for face in known_faces:
#                 face['embeddings'] = [np.array(embedding) for embedding in face['embeddings']]
#             return known_faces
#     return []

# # Compare embeddings using cosine similarity
# def get_cosine_similarity(embedding1, embedding2):
#     if embedding1.size == 0 or embedding2.size == 0:
#         print("One of the embeddings is empty. Cannot compute cosine similarity.")
#         return float('inf')  # Return a large value to indicate dissimilarity
#     return 1 - cosine(embedding1.flatten(), embedding2.flatten())  # Flatten both embeddings to 1D

# def generate_frames():
#     # Start video capture (webcam)
#     cap = cv2.VideoCapture(0)

#     # Load known faces and embeddings
#     known_faces = load_known_faces()

#     # Start video stream
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Convert frame to RGB (as OpenCV loads images in BGR format)
#         img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#         # Detect faces
#         boxes, _ = mtcnn.detect(img_rgb)

#         if boxes is not None:
#             faces = mtcnn(img_rgb)  # Extract faces from the image
#             for i, face in enumerate(faces):
#                 face = face.to(device)  # Ensure the face tensor is on the same device as the model
#                 embedding = model(face.unsqueeze(0))  # Get embedding for the detected face

#                 # Compare with known faces (from multiple angles)
#                 matched = False
#                 for known_face in known_faces:
#                     name = known_face['name']
#                     stored_embeddings = known_face['embeddings']
#                     for stored_embedding in stored_embeddings:
#                         similarity = get_cosine_similarity(embedding.detach().cpu().numpy(), stored_embedding)
#                         if similarity > 0.7:  # If similarity is above 0.7, consider it a match
#                             matched = True
#                             cv2.putText(frame, name, (int(boxes[i][0]), int(boxes[i][1] - 10)),
#                                         cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
#                             break

#                     if matched:
#                         break

#                 if not matched:
#                     cv2.putText(frame, "Unknown", (int(boxes[i][0]), int(boxes[i][1] - 10)),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

#                 # Draw bounding box around the detected face
#                 cv2.rectangle(frame, (int(boxes[i][0]), int(boxes[i][1])),
#                               (int(boxes[i][2]), int(boxes[i][3])), (0, 255, 0), 2)

#         # Encode the frame in JPEG format
#         ret, buffer = cv2.imencode('.jpg', frame)
#         frame = buffer.tobytes()

#         yield frame

# def start_server(host='0.0.0.0', port=5001):
#     server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     server_socket.bind((host, port))
#     server_socket.listen(1)
#     print(f"Server started at {host}:{port}")

#     while True:
#         client_socket, addr = server_socket.accept()
#         print(f"Connection from {addr}")

#         try:
#             for frame in generate_frames():
#                 # Send frame size
#                 client_socket.sendall(len(frame).to_bytes(4, byteorder='big'))
#                 # Send frame data
#                 client_socket.sendall(frame)
#         except Exception as e:
#             print(f"Error: {e}")
#         finally:
#             client_socket.close()

# if __name__ == '__main__':
#     start_server()
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
from flask import Response

# Initialize MTCNN for face detection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Initialize YOLOv8 for action detection
model_yolo = YOLO('yolo11s-pose.pt')

# Load XGBoost model for classification
model_xgb = xgb.Booster()
model_xgb.load_model('Models/trained_model.json')

# Path to store the embeddings of known faces
EMBEDDINGS_FILE = 'Models/known_faces_embeddings.pkl'

# Load known faces and embeddings
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
        return float('inf')  # Large value indicates dissimilarity
    return 1 - cosine(embedding1.flatten(), embedding2.flatten())

def generate_frames():
    cap = cv2.VideoCapture(0)  # Use webcam
    known_faces = load_known_faces()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, _ = mtcnn.detect(frame_rgb)

        # Face recognition
        if boxes is not None:
            faces = mtcnn(frame_rgb)
            for i, face in enumerate(faces):
                embedding = model(face.unsqueeze(0))  # Get face embedding
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

        # Action detection
        results = model_yolo(frame, verbose=False)
        annotated_frame = results[0].plot(boxes=False)

        for r in results:
            bound_box = r.boxes.xyxy
            conf = r.boxes.conf.tolist()
            keypoints = r.keypoints.xyn.tolist()

            for index, box in enumerate(bound_box):
                if conf[index] > 0.55:
                    x1, y1, x2, y2 = map(int, box.tolist())

                    # Prepare data for XGBoost prediction
                    data = {f'x{j}': keypoints[index][j][0] for j in range(len(keypoints[index]))}
                    data.update({f'y{j}': keypoints[index][j][1] for j in range(len(keypoints[index]))})
                    df = pd.DataFrame(data, index=[0])
                    dmatrix = xgb.DMatrix(df)

                    # Predict suspicious behavior
                    sus = model_xgb.predict(dmatrix)
                    binary_prediction = int(sus > 0.5)

                    if binary_prediction == 0:  # Suspicious
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cvzone.putTextRect(frame, "Suspicious", (x1, y1), 1, 1)
                    else:  # Normal
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cvzone.putTextRect(frame, "Normal", (x1, y1 + 50), 1, 1)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
