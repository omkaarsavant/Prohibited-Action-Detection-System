import cv2
import torch
import pickle
import time
from facenet_pytorch import MTCNN, InceptionResnetV1
from scipy.spatial.distance import cosine
import os
import numpy as np
from sklearn.metrics import classification_report  # For classification report
import matplotlib.pyplot as plt  # For plotting graphs

# Initialize MTCNN for face detection
mtcnn = MTCNN(keep_all=True, device='cpu')  # Change 'cuda' to 'cpu' if needed

# Initialize Inception ResNet for face recognition
model = InceptionResnetV1(pretrained='vggface2').eval().to('cpu')  # Change to 'cuda' if using GPU

# Path to store the embeddings of known faces
EMBEDDINGS_FILE = 'known_faces_embeddings.pkl'

# Load known faces and embeddings from file (if exists)
def load_known_faces():
    if os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE, 'rb') as f:
            known_faces = pickle.load(f)
            # Ensure embeddings are numpy arrays
            for face in known_faces:
                face['embeddings'] = [np.array(embedding) for embedding in face['embeddings']]
            return known_faces
    return []

# Compare embeddings using cosine similarity
def get_cosine_similarity(embedding1, embedding2):
    if embedding1.size == 0 or embedding2.size == 0:
        print("One of the embeddings is empty. Cannot compute cosine similarity.")
        return float('inf')  # Large value indicating dissimilarity
    return 1 - cosine(embedding1.flatten(), embedding2.flatten())

# Function to recognize faces from video stream using multiple angles
def recognize_faces_from_stream():
    # Start video capture (webcam)
    cap = cv2.VideoCapture(0)

    # Load known faces and embeddings
    known_faces = load_known_faces()

    # Check for label data with name 'Deepak' in known faces
    deepak_exists = any(face.get('name') == "Deepak" for face in known_faces)
    if deepak_exists:
        print("Label data for 'Deepak' found in known faces.")
    else:
        print("Label data for 'Deepak' not found in known faces.")

    # Lists to accumulate frame processing times and predicted labels for classification
    frame_times = []
    predicted_labels = []

    # Process the video stream frame by frame
    while True:
        start_time = time.time()  # Timer starts

        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB (OpenCV loads in BGR)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces
        boxes, _ = mtcnn.detect(img_rgb)

        if boxes is not None:
            # Extract faces from the image
            faces = mtcnn(img_rgb)
            for i, face in enumerate(faces):
                # Get embedding for the detected face
                embedding = model(face.unsqueeze(0))
                label = "Unknown"
                matched = False

                # Compare with known faces
                for known_face in known_faces:
                    name = known_face['name']
                    stored_embeddings = known_face['embeddings']
                    for stored_embedding in stored_embeddings:
                        similarity = get_cosine_similarity(embedding.detach().cpu().numpy(), stored_embedding)
                        if similarity > 0.7:  # If similarity is high enough, it's a match
                            label = name
                            matched = True
                            if name == "Deepak":
                                print("Deepak detected!")
                            # Optionally include the similarity score in the displayed label
                            label = f"{name} ({similarity:.2f})"
                            break
                    if matched:
                        break

                # Append the predicted label (without the similarity score) for classification
                predicted_labels.append(label.split(" (")[0])

                # Draw label and bounding box
                cv2.putText(frame, label, (int(boxes[i][0]), int(boxes[i][1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0) if matched else (0, 0, 255), 2)
                cv2.rectangle(frame, (int(boxes[i][0]), int(boxes[i][1])),
                              (int(boxes[i][2]), int(boxes[i][3])), (0, 255, 0) if matched else (0, 0, 255), 2)

        # Compute elapsed time for the frame (in milliseconds)
        end_time = time.time()
        elapsed_ms = (end_time - start_time) * 1000
        frame_times.append(elapsed_ms)
        cv2.putText(frame, f"Frame Time: {elapsed_ms:.2f} ms", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Display the video stream
        cv2.imshow("Face Recognition - Video Stream", frame)

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

    # Print average execution time per frame
    if frame_times:
        avg_time = np.mean(frame_times)
        print(f"Average execution time per frame: {avg_time:.2f} ms")

    # --- Generate Classification Report ---
    # Assume that the ground truth for every detection is "Deepak"
    if predicted_labels:
        y_true = ["Deepak"] * len(predicted_labels)
        print("\nClassification Report for target label 'Deepak':")
        report = classification_report(y_true, predicted_labels, zero_division=0)
        print(report)

        deepak_count = predicted_labels.count("Deepak")
        print(f"'Deepak' was detected {deepak_count} times out of {len(predicted_labels)} total detections.")
    else:
        print("No faces were detected; classification report not generated.")

    # --- Generate Graph for Frame Processing Time ---
    plt.figure(figsize=(10, 5))
    plt.plot(frame_times, marker='o', linestyle='-', color='r')
    plt.xlabel('Frame Index')
    plt.ylabel('Processing Time (ms)')
    plt.title('Frame Processing Time per Frame')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Main entry point
if __name__ == "__main__":
    recognize_faces_from_stream()
