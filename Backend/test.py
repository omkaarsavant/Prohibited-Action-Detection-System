# # from flask import Flask, request, jsonify, render_template
# # import os
# # from flask_cors import CORS
# # import cv2, pickle
# # from video_stream import video_feed  # Import the video feed function

# # app = Flask(__name__)
# # CORS(app)

# # app.config['IMAGE_FOLDER'] = 'Images'

# # # @app.route('/upload', methods=['POST'])
# # # def upload_file():
# # #     print("Form Received")

# # #     images = request.files.getlist('FaceImages')

# # #     name = request.form.get('Name')
# # #     prisoner_number = request.form.get('PrisonerNumber')
# # #     age = request.form.get('Age')
# # #     height = request.form.get('Height')
# # #     weight = request.form.get('Weight')
# # #     gender = request.form.get('Gender')

# # #     # Save the image files to the Images folder
# # #     image_paths = []
# # #     for image in images:
# # #         if image and image.filename != '':
# # #             image_path = os.path.join(app.config['IMAGE_FOLDER'], image.filename)
# # #             image.save(image_path)
# # #             image_paths.append(image_path)
# # #         else:
# # #             return jsonify({'error': 'Invalid image file'}), 400

# # #     # Use the saved image file path for face recognition
# # #     img = cv2.cvtColor(cv2.imread(image_paths[0]), cv2.COLOR_BGR2RGB)
# # #     encodings = face_recognition.face_encodings(img)
# # #     if len(encodings) > 0:
# # #         encode = encodings[0]
# # #     else:
# # #         return jsonify({'error': f"No face found in image '{image_paths[0]}'"}), 400

# # #     # Prepare the new encoding entry as a dictionary
# # #     new_encoding = {
# # #         'prisoner_id': prisoner_number,
# # #         'encoding': encode.tolist()  # Convert to list for serialization
# # #     }

# # #     # Save the encoding to a pickle file
# # #     encode_file_path = 'encodefile.p'
# # #     if os.path.exists(encode_file_path):
# # #         # Load existing encodings
# # #         with open(encode_file_path, 'rb') as f:
# # #             all_encodings = pickle.load(f)
# # #     else:
# # #         all_encodings = []

# # #     # Append the new encoding
# # #     all_encodings.append(new_encoding)
# # #     print(len(all_encodings))

# # #     # Save the updated encodings back to the pickle file
# # #     with open(encode_file_path, 'wb') as f:
# # #         pickle.dump(all_encodings, f)

# # #     print(f"Received Name: {name}")
# # #     print(f"Received Prisoner Number: {prisoner_number}")
# # #     print(f"Received Age: {age}")
# # #     print(f"Received Height: {height}")
# # #     print(f"Received Weight: {weight}")
# # #     print(f"Received Gender: {gender}")
# # #     print(f"Images saved at: {image_paths}")

# # #     return jsonify({
# # #         'message': f"Received name: {name}, prisoner number: {prisoner_number}, age: {age}, height: {height}, weight: {weight}, gender: {gender}, and images: {[image.filename for image in images]}"
# # #     })

# # @app.route('/video_feed')
# # def stream_video():
# #     return video_feed()

# # @app.route('/')
# # def index():
# #     return render_template('index.html')

# # if __name__ == '__main__':
# #     if not os.path.exists(app.config['IMAGE_FOLDER']):
# #         os.makedirs(app.config['IMAGE_FOLDER'])
# #     app.run(port=5000)



# from flask import Flask, render_template, Response
# from flask_cors import CORS
# import os
# import socket

# app = Flask(__name__)
# CORS(app)

# app.config['IMAGE_FOLDER'] = 'Images'

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
#     return Response(receive_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/')
# def index():
#     return render_template('index.html')

# if __name__ == '__main__':
#     if not os.path.exists(app.config['IMAGE_FOLDER']):
#         os.makedirs(app.config['IMAGE_FOLDER'])
#     app.run(port=5000)



from flask import Flask, render_template, Response, request, jsonify
from flask_cors import CORS
import os
import socket
from store_embeddings_multiple import store_face_embeddings_from_images
from action_detection import detect_shoplifting, video_path
from video_stream import video_feed

app = Flask(__name__)
CORS(app)

app.config['IMAGE_FOLDER'] = 'Images'

@app.route('/upload', methods=['POST'])
def upload():
    name = request.form['Name']
    images = request.files.getlist('FaceImages')
    store_face_embeddings_from_images(images, name)
    return jsonify({"message": "Embeddings stored successfully"}), 200

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