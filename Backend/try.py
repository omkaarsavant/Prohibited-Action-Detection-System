import os
import cv2
import torch
import pickle
import numpy as np
from tqdm import tqdm
from flask import Flask, request, jsonify
from pymongo import MongoClient
from facenet_pytorch import MTCNN, InceptionResnetV1
from flask_cors import CORS
# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})# Enable CORS for all routes

app.config['IMAGE_FOLDER'] = 'stored_images'  # Folder to store images
os.makedirs(app.config['IMAGE_FOLDER'], exist_ok=True)

# MongoDB setup
client = MongoClient("mongodb://localhost:27017/")
db = client["face_recognition_db"]
collection = db["known_faces"]

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize MTCNN for face detection
mtcnn = MTCNN(keep_all=True, device=device)
# Initialize Inception ResNet for face recognition
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Path to store embeddings
EMBEDDINGS_FILE = 'Models/known_faces_embeddings.pkl'
IMAGE_SIZE = (640, 480)

# Load known faces and embeddings from file (if exists)
def load_known_faces():
    if os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE, 'rb') as f:
            return pickle.load(f)
    return []

# Save known faces and embeddings to a file
def save_known_faces(known_faces):
    with open(EMBEDDINGS_FILE, 'wb') as f:
        pickle.dump(known_faces, f)

# Store face embeddings and save images
@app.route('/store_faces', methods=['POST'])
def store_face_embeddings_from_images():
    name = request.form.get('name')
    images = request.files.getlist('images')  # Multiple images can be uploaded
    embeddings = []
    image_paths = []
    
    for image in images:
        # Save the image with a unique filename
        filename = f"{name}_{image.filename}"
        path = os.path.join(app.config['IMAGE_FOLDER'], filename)
        image.save(path)  # Save before processing
        image_paths.append(path)

        # Read and process the saved image
        img = cv2.imread(path)
        if img is None:
            print(f"Error: Could not read {path}")
            continue

        img_resized = cv2.resize(img, IMAGE_SIZE)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

        # Detect faces
        boxes, _ = mtcnn.detect(img_rgb)
        if boxes is not None:
            faces = mtcnn(img_rgb)
            for face in faces:
                embedding = model(face.unsqueeze(0).to(device))
                embeddings.append(embedding.detach().cpu().numpy())
        else:
            print(f"No face detected in {filename}, skipping.")

    known_faces = load_known_faces()
    found = False
    for known_face in known_faces:
        if known_face['name'] == name:
            known_face['embeddings'].extend(embeddings)
            found = True
            break

    if not found:
        known_faces.append({'name': name, 'embeddings': embeddings})
    
    save_known_faces(known_faces)

    # Store data in MongoDB
    prisoner_data = {
        "name": name,
        "image_paths": image_paths,
        "embedding_count": len(embeddings)
    }
    collection.insert_one(prisoner_data)
    
    return jsonify({'message': f'Embeddings and images for {name} stored successfully.'})

if __name__ == '__main__':
    app.run(debug=True)
