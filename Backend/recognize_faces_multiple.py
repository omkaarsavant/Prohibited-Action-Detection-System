from flask import Flask, request, jsonify
import cv2
import torch
import pickle
from facenet_pytorch import MTCNN, InceptionResnetV1
from scipy.spatial.distance import cosine
import os
import numpy as np
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Initialize MTCNN for face detection
mtcnn = MTCNN(keep_all=True, device='cpu')  # Change 'cpu' to 'cuda' for GPU
# Initialize Inception ResNet for face recognition
model = InceptionResnetV1(pretrained='vggface2').eval().to('cpu')  # Change to 'cuda' if using GPU

# Path to store the embeddings of known faces
EMBEDDINGS_FILE = 'Models/known_faces_embeddings.pkl'

# Load known faces and embeddings from file (if exists)
def load_known_faces():
    if os.path.exists(EMBEDDINGS_FILE):
        try:
            with open(EMBEDDINGS_FILE, 'rb') as f:
                known_faces = pickle.load(f)
                # Ensure known_faces is a list of dictionaries
                if isinstance(known_faces, list) and all(isinstance(face, dict) for face in known_faces):
                    # Ensure embeddings are numpy arrays
                    for face in known_faces:
                        face['embeddings'] = [np.array(embedding) for embedding in face['embeddings']]
                    print(f"Loaded {len(known_faces)} known faces.")
                    return known_faces
                else:
                    print("Error: Invalid format in known_faces_embeddings.pkl")
                    return []
        except (pickle.UnpicklingError, EOFError, AttributeError, ImportError, IndexError) as e:
            print(f"Error: Invalid format in known_faces_embeddings.pkl - {e}")
            return []
    print("No known faces embeddings file found.")
    return []

# Compare embeddings using cosine similarity
def get_cosine_similarity(embedding1, embedding2):
    if embedding1.size == 0 or embedding2.size == 0:
        print("One of the embeddings is empty. Cannot compute cosine similarity.")
        return float('inf')  # Return a large value to indicate dissimilarity
    return 1 - cosine(embedding1.flatten(), embedding2.flatten())  # Flatten both embeddings to 1D

# Recognize faces in the frame
def recognize_faces(frame):
    known_faces = load_known_faces()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, _ = mtcnn.detect(img)
    recognized_faces = []
    if boxes is not None:
        print(f"Detected {len(boxes)} faces.")
        for box in boxes:
            face = img[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
            face = cv2.resize(face, (160, 160))
            face = torch.tensor(face).permute(2, 0, 1).unsqueeze(0).float().to('cpu')
            embedding = model(face).detach().cpu().numpy()
            name = "Unknown"
            max_similarity = 0
            for known_face in known_faces:
                for known_embedding in known_face['embeddings']:
                    similarity = get_cosine_similarity(embedding, known_embedding)
                    if similarity > max_similarity:
                        max_similarity = similarity
                        if similarity > 0.6:  # Adjusted threshold
                            name = known_face['name']
            recognized_faces.append({"name": name, "box": box.tolist(), "similarity": max_similarity})
            print(f"Recognized: {name} with similarity: {max_similarity}")  # Print recognized name for debugging
    else:
        print("No faces detected.")
    return recognized_faces

@app.route('/recognize', methods=['GET'])
def recognize():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        img = Image.open(BytesIO(file.read()))
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        recognized_faces = recognize_faces(frame)
        return jsonify(recognized_faces)

if __name__ == '__main__':
    app.run(debug=True)
    