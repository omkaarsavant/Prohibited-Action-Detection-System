import os
import cv2
import torch
import pickle
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
from tqdm import tqdm

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize MTCNN for face detection
mtcnn = MTCNN(keep_all=True, device=device)
# Initialize Inception ResNet for face recognition
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Path to store the embeddings of known faces
EMBEDDINGS_FILE = 'Models/known_faces_embeddings.pkl'

# Resize image for faster processing
IMAGE_SIZE = (640, 480)

# Load known faces and embeddings from file (if exists)
def load_known_faces():
    if os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE, 'rb') as f:
            known_faces = pickle.load(f)
            return known_faces
    return []

# Save known faces and embeddings to a file
def save_known_faces(known_faces):
    with open(EMBEDDINGS_FILE, 'wb') as f:
        pickle.dump(known_faces, f)

# Function to store face embeddings from uploaded images
def store_face_embeddings_from_images(images, name):
    embeddings = []

    for image in images:
        # Load and resize the image
        img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print("Could not read the image.")
            continue

        img_resized = cv2.resize(img, IMAGE_SIZE)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

        # Detect faces
        boxes, _ = mtcnn.detect(img_rgb)

        if boxes is not None:
            faces = mtcnn(img_rgb)
            for i, face in enumerate(faces):
                embedding = model(face.unsqueeze(0).to(device))
                embeddings.append(embedding.detach().cpu().numpy())
        else:
            print("No face detected, skipping this image.")

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
    print(f"Embeddings for {name} have been stored.")