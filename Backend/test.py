# # # from flask import Flask, request, jsonify, render_template
# # # import os
# # # from flask_cors import CORS
# # # import cv2, pickle
# # # from video_stream import video_feed  # Import the video feed function

# # # app = Flask(__name__)
# # # CORS(app)

# # # app.config['IMAGE_FOLDER'] = 'Images'

# # # # @app.route('/upload', methods=['POST'])
# # # # def upload_file():
# # # #     print("Form Received")

# # # #     images = request.files.getlist('FaceImages')

# # # #     name = request.form.get('Name')
# # # #     prisoner_number = request.form.get('PrisonerNumber')
# # # #     age = request.form.get('Age')
# # # #     height = request.form.get('Height')
# # # #     weight = request.form.get('Weight')
# # # #     gender = request.form.get('Gender')

# # # #     # Save the image files to the Images folder
# # # #     image_paths = []
# # # #     for image in images:
# # # #         if image and image.filename != '':
# # # #             image_path = os.path.join(app.config['IMAGE_FOLDER'], image.filename)
# # # #             image.save(image_path)
# # # #             image_paths.append(image_path)
# # # #         else:
# # # #             return jsonify({'error': 'Invalid image file'}), 400

# # # #     # Use the saved image file path for face recognition
# # # #     img = cv2.cvtColor(cv2.imread(image_paths[0]), cv2.COLOR_BGR2RGB)
# # # #     encodings = face_recognition.face_encodings(img)
# # # #     if len(encodings) > 0:
# # # #         encode = encodings[0]
# # # #     else:
# # # #         return jsonify({'error': f"No face found in image '{image_paths[0]}'"}), 400

# # # #     # Prepare the new encoding entry as a dictionary
# # # #     new_encoding = {
# # # #         'prisoner_id': prisoner_number,
# # # #         'encoding': encode.tolist()  # Convert to list for serialization
# # # #     }

# # # #     # Save the encoding to a pickle file
# # # #     encode_file_path = 'encodefile.p'
# # # #     if os.path.exists(encode_file_path):
# # # #         # Load existing encodings
# # # #         with open(encode_file_path, 'rb') as f:
# # # #             all_encodings = pickle.load(f)
# # # #     else:
# # # #         all_encodings = []

# # # #     # Append the new encoding
# # # #     all_encodings.append(new_encoding)
# # # #     print(len(all_encodings))

# # # #     # Save the updated encodings back to the pickle file
# # # #     with open(encode_file_path, 'wb') as f:
# # # #         pickle.dump(all_encodings, f)

# # # #     print(f"Received Name: {name}")
# # # #     print(f"Received Prisoner Number: {prisoner_number}")
# # # #     print(f"Received Age: {age}")
# # # #     print(f"Received Height: {height}")
# # # #     print(f"Received Weight: {weight}")
# # # #     print(f"Received Gender: {gender}")
# # # #     print(f"Images saved at: {image_paths}")

# # # #     return jsonify({
# # # #         'message': f"Received name: {name}, prisoner number: {prisoner_number}, age: {age}, height: {height}, weight: {weight}, gender: {gender}, and images: {[image.filename for image in images]}"
# # # #     })

# # # @app.route('/video_feed')
# # # def stream_video():
# # #     return video_feed()

# # # @app.route('/')
# # # def index():
# # #     return render_template('index.html')

# # # if __name__ == '__main__':
# # #     if not os.path.exists(app.config['IMAGE_FOLDER']):
# # #         os.makedirs(app.config['IMAGE_FOLDER'])
# # #     app.run(port=5000)



# # from flask import Flask, render_template, Response
# # from flask_cors import CORS
# # import os
# # import socket

# # app = Flask(__name__)
# # CORS(app)

# # app.config['IMAGE_FOLDER'] = 'Images'

# # def receive_frames(host='127.0.0.1', port=5001):
# #     client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# #     client_socket.connect((host, port))

# #     try:
# #         while True:
# #             # Receive frame size
# #             frame_size = int.from_bytes(client_socket.recv(4), byteorder='big')
# #             # Receive frame data
# #             frame_data = b''
# #             while len(frame_data) < frame_size:
# #                 packet = client_socket.recv(frame_size - len(frame_data))
# #                 if not packet:
# #                     break
# #                 frame_data += packet

# #             yield (b'--frame\r\n'
# #                    b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
# #     finally:
# #         client_socket.close()

# # @app.route('/video_feed')
# # def stream_video():
# #     return Response(receive_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# # @app.route('/')
# # def index():
# #     return render_template('index.html')

# # if __name__ == '__main__':
# #     if not os.path.exists(app.config['IMAGE_FOLDER']):
# #         os.makedirs(app.config['IMAGE_FOLDER'])
# #     app.run(port=5000)



# from flask import Flask, render_template, Response, request, jsonify
# from flask_cors import CORS
# import os
# import socket
# from store_embeddings_multiple import store_face_embeddings_from_images
# from action_detection import detect_shoplifting
# from video_stream import video_feed

# app = Flask(__name__)
# CORS(app)

# app.config['IMAGE_FOLDER'] = 'Images'

# @app.route('/upload', methods=['POST'])
# def upload():
#     name = request.form['Name']
#     images = request.files.getlist('FaceImages')
#     store_face_embeddings_from_images(images, name)
#     return jsonify({"message": "Embeddings stored successfully"}), 200

# def receive_frames(host='127.0.0.1', port=5001):
#     client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     client_socket.connect((host, port))

#     try:
#         while True:
#             # Receive frame size
#             frame_size = int.from_bytes(client_socket.recv(4), byteorder='big')
#             # Receive frame data
#             frame_data = b''
#             while len(frame_data) < frame_size:
#                 packet = client_socket.recv(frame_size - len(frame_data))
#                 if not packet:
#                     break
#                 frame_data += packet

#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
#     finally:
#         client_socket.close()

# @app.route('/video_feed')
# def stream_video():
#     return video_feed()
#     # return Response(receive_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/action-detection')
# def action_detection():
#     return Response(detect_shoplifting("Assets/Videos/vid.mp4"), mimetype='multipart/x-mixed-replace; boundary=frame')


# @app.route('/')
# def index():
#     return render_template('index.html')

# if __name__ == '__main__':
#     if not os.path.exists(app.config['IMAGE_FOLDER']):
#         os.makedirs(app.config['IMAGE_FOLDER'])
#     app.run(port=5000)

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

app = Flask(__name__)
CORS(app)

app.config["MONGO_URI"] = "mongodb://localhost:27017/your_database_name"
mongo = PyMongo(app)

# Set the path for the images
app.config['IMAGE_FOLDER'] = os.path.join(os.path.dirname(__file__), 'Images')
os.makedirs(app.config['IMAGE_FOLDER'], exist_ok=True)  # Create image folder if it doesn't exist

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

    cap.release()
    cv2.destroyAllWindows()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload', methods=['POST'])
def upload():
    name = request.form['Name']
    prisoner_number = request.form['PrisonerNumber']
    age = request.form['Age']
    height = request.form['Height']
    weight = request.form['Weight']
    gender = request.form['Gender']
    images = request.files.getlist('FaceImages')

    store_face_embeddings_from_images(images, prisoner_number)
    
    image_paths = []
    for image in images:
        if image:
            # Convert image to JPEG format
            img = PILImage.open(image)
            img = img.convert("RGB")  # Ensure image is in RGB mode
            filename = secure_filename(image.filename)
            jpeg_filename = f"{os.path.splitext(filename)[0]}.jpeg"  # Change extension to .jpeg
            image_path = os.path.join(app.config['IMAGE_FOLDER'], jpeg_filename)

            # Save the image as JPEG
            img.save(image_path, "JPEG")  # Save as JPEG
            image_paths.append(f"Images/{jpeg_filename}")  # Store the path for MongoDB

    user_data = {
        "Name": name,
        "PrisonerNumber": prisoner_number,
        "Age": age,
        "Height": height,
        "Weight": weight,
        "Gender": gender,
        "ImagePaths": image_paths  # Store the list of image paths
    }
    mongo.db.users.insert_one(user_data)  # Insert user data into MongoDB

    return jsonify({"message": "Embeddings stored successfully"}), 200

@app.route('/images/<path:filename>', methods=['GET'])
def serve_image(filename):
    return send_from_directory(app.config['IMAGE_FOLDER'], filename)

@app.route('/user/<prisoner_number>', methods=['GET'])
def get_user(prisoner_number):
    user = mongo.db.users.find_one({"PrisonerNumber": prisoner_number})
    if user:
        return jsonify({
            "Name": user["Name"],
            "PrisonerNumber": user["PrisonerNumber"],
            "Age": user["Age"],
            "Height": user["Height"],
            "Weight": user["Weight"],
            "Gender": user["Gender"],
            "ImagePaths": user["ImagePaths"]  # Return the image paths
        }), 200
    else:
        return jsonify({"message": "User  not found"}), 404

def receive_frames(host='127.0.0.1', port=5001):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))

    try:
        while True:
            # Receive frame size
            frame_size = int.from_bytes(client_socket.recv(4), byteorder='big')
            # Receive frame data
            frame_data = b''
            while len(frame_data) < frame_size:
                packet = client_socket.recv(frame_size - len(frame_data))
                if not packet:
                    break
                frame_data += packet

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
    finally:
        client_socket.close()


@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    if not os.path.exists(app.config['IMAGE_FOLDER']):
        os.makedirs(app.config['IMAGE_FOLDER'])
    app.run(port=5000)