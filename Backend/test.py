from flask import Flask, render_template, Response, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_pymongo import PyMongo
import os
import socket
from store_embeddings_multiple import store_face_embeddings_from_images
from action_detection import detect_shoplifting, video_path
from video_stream import video_feed
from werkzeug.utils import secure_filename
from PIL import Image as PILImage

app = Flask(__name__)
CORS(app)

# MongoDB Configuration
app.config["MONGO_URI"] = "mongodb://localhost:27017/your_database_name"
mongo = PyMongo(app)

# Set the path for the images
app.config['IMAGE_FOLDER'] = os.path.join(os.path.dirname(__file__), 'Images')
os.makedirs(app.config['IMAGE_FOLDER'], exist_ok=True)  # Create image folder if it doesn't exist

@app.route('/upload', methods=['POST'])
def upload():
    name = request.form['Name']
    prisoner_number = request.form['PrisonerNumber']
    age = request.form['Age']
    height = request.form['Height']
    weight = request.form['Weight']
    gender = request.form['Gender']
    images = request.files.getlist('FaceImages')

    # Store face embeddings
    store_face_embeddings_from_images(images, name)

    # Store user data in MongoDB
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

    return jsonify({"message": "Embeddings stored successfully and user data saved."}), 200

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
# Other routes remain unchanged...
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

@app.route('/video_feed')
def stream_video():
    return video_feed()
    # return Response(receive_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/action-detection')
def action_detection():
    return Response(detect_shoplifting("Assets/Videos/vid.mp4"), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    if not os.path.exists(app.config['IMAGE_FOLDER']):
        os.makedirs(app.config['IMAGE_FOLDER'])
    app.run(port=5000)