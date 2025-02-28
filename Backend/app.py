from flask import Flask, render_template, Response
import cv2
from recognize_faces_multiple import recognize_faces

from flask import Flask, request, jsonify, send_file
from pymongo import MongoClient
import gridfs
from io import BytesIO


app = Flask(__name__)

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["UserDatabase"]
fs = gridfs.GridFS(db)
users_collection = db["users"]

@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video device")
        return
    while True:
        success, frame = cap.read()
        if not success:
            print("Error: Could not read frame")
            break
        else:
            frame = recognize_faces(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                print("Error: Could not encode frame")
                continue
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
    