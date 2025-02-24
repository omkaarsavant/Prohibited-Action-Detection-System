# import os
# import cv2
# import torch
# import pickle
# from facenet_pytorch import MTCNN, InceptionResnetV1
# import numpy as np
# from tqdm import tqdm  # Import tqdm for progress bar

# # Check if CUDA is available
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # Initialize MTCNN for face detection
# mtcnn = MTCNN(keep_all=True, device=device)  # Use 'device' variable for GPU
# # Initialize Inception ResNet for face recognition
# model = InceptionResnetV1(pretrained='vggface2').eval().to(device)  # Use 'device' variable for GPU

# # Path to store the embeddings of known faces
# EMBEDDINGS_FILE = 'known_faces_embeddings.pkl'

# # Resize image for faster processing
# IMAGE_SIZE = (640, 480)  # Resize to 640x480, adjust as needed

# # Load known faces and embeddings from file (if exists)
# def load_known_faces():
#     if os.path.exists(EMBEDDINGS_FILE):
#         with open(EMBEDDINGS_FILE, 'rb') as f:
#             known_faces = pickle.load(f)
#             return known_faces
#     return []

# # Save known faces and embeddings to a file
# def save_known_faces(known_faces):
#     with open(EMBEDDINGS_FILE, 'wb') as f:
#         pickle.dump(known_faces, f)

# # Function to store multiple face embeddings from a folder of images
# def store_face_embeddings_from_folder(folder_path, name):
#     embeddings = []

#     # Get all image files in the folder
#     image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('png', 'jpg', 'jpeg'))]

#     if not image_files:
#         print("No images found in the folder.")
#         return

#     print(f"Processing {len(image_files)} images...")

#     # Create a tqdm progress bar
#     for image_file in tqdm(image_files, desc="Processing images", unit="image"):
#         image_path = os.path.join(folder_path, image_file)

#         # Load and resize the image
#         img = cv2.imread(image_path)
#         if img is None:
#             print(f"Could not read the image: {image_path}")
#             continue

#         img_resized = cv2.resize(img, IMAGE_SIZE)
#         img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

#         # Detect faces
#         boxes, _ = mtcnn.detect(img_rgb)

#         if boxes is not None:
#             faces = mtcnn(img_rgb)  # Extract faces from the image
#             for i, face in enumerate(faces):
#                 # Get the embedding for the detected face
#                 embedding = model(face.unsqueeze(0).to(device))

#                 # Add the embedding to the list
#                 embeddings.append(embedding.detach().cpu().numpy())

#                 # Draw bounding box around the detected face
#                 cv2.rectangle(img_resized, (int(boxes[i][0]), int(boxes[i][1])),
#                               (int(boxes[i][2]), int(boxes[i][3])), (0, 255, 0), 2)

#             # Optionally show the processed image (can be skipped for large datasets)
#             # cv2.imshow(f"Processed Image: {image_file}", img_resized)
#             # cv2.waitKey(0)
#         else:
#             print(f"No face detected in {image_file}, skipping this image.")

#     # Release any OpenCV windows
#     cv2.destroyAllWindows()

#     # Save embeddings for the person
#     known_faces = load_known_faces()

#     # Check if the person already exists in the known faces list
#     found = False
#     for known_face in known_faces:
#         if known_face['name'] == name:
#             known_face['embeddings'].extend(embeddings)
#             found = True
#             break

#     # If the person does not exist, create a new entry
#     if not found:
#         known_faces.append({'name': name, 'embeddings': embeddings})

#     # Save the updated embeddings
#     save_known_faces(known_faces)
#     print(f"Embeddings for {name} have been stored.")

# if __name__ == "__main__":
#     folder_path = input("Enter the folder path containing images of the person: ")
#     name = input("Enter the name of the person: ")

#     # Call the function to store embeddings
#     store_face_embeddings_from_folder(folder_path, name)





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