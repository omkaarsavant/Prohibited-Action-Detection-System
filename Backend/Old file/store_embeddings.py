import cv2
import torch
import pickle
from facenet_pytorch import MTCNN, InceptionResnetV1
import os

# Initialize MTCNN for face detection
mtcnn = MTCNN(keep_all=True, device='cpu')  # Change 'cpu' to 'cuda' for GPU
# Initialize Inception ResNet for face recognition
model = InceptionResnetV1(pretrained='vggface2').eval().to('cpu')  # Change to 'cuda' if using GPU

# Path to store the embeddings of known faces
EMBEDDINGS_FILE = 'known_faces_embeddings.pkl'

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

# Function to process a single image
def store_face_embedding(image_path, name):
    # Load image from file
    img = cv2.imread(image_path)

    if img is None:
        print(f"Could not read the image: {image_path}")
        return

    # Convert BGR to RGB (as OpenCV loads images in BGR format)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Detect faces
    boxes, _ = mtcnn.detect(img_rgb)

    if boxes is not None:
        faces = mtcnn(img_rgb)  # Extract faces from the image
        for i, face in enumerate(faces):
            # Get the embedding for the detected face
            embedding = model(face.unsqueeze(0))

            # Load known faces and embeddings
            known_faces = load_known_faces()

            # Add the new face and its embedding to the known_faces list
            known_faces.append((name, embedding.detach().cpu().numpy()))

            # Save the updated known faces and embeddings
            save_known_faces(known_faces)

            print(f"Face embedding for {name} has been stored.")

# Main function to prompt for user input and process image
if __name__ == "__main__":
    # Path to the image you want to process
    image_path = input("Enter the path of the image: ")

    # Name of the person
    name = input("Enter the name for this person: ")

    # Store the face embedding for the given image
    store_face_embedding(image_path, name)
