from flask import Flask, render_template, request
import cv2
import numpy as np
import os
import base64
import pickle
import subprocess

app = Flask(__name__)
UPLOAD_FOLDER = 'faces'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load DNN face detector
net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")

recognizer = None
labels = {}

def load_model():
    global recognizer, labels
    if os.path.exists("recognizer.yml") and os.path.exists("labels.pkl"):
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read("recognizer.yml")
        with open("labels.pkl", "rb") as f:
            raw_labels = pickle.load(f)
            labels = {v: k for k, v in raw_labels.items()}
        print("✅ Model loaded.")
    else:
        print("⚠️ Model not found. Only registration will work.")

load_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])  # ✅ Allow POST
def upload():
    data_url = request.form['image']
    username = request.form.get('username')
    action = request.form['action']

    if action == "register" and not username:
        return render_template('index.html', result="❌ Username is required for registration.")

    # Decode base64 image from webcam
    encoded = data_url.split(',')[1]
    image_data = base64.b64decode(encoded)
    np_arr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    (h, w) = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # DNN face detection
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            faces.append((x1, y1, x2 - x1, y2 - y1))

    if len(faces) == 0:
        return render_template('index.html', result="❌ No face detected.")

    x, y, w, h = faces[0]
    roi = gray[y:y+h, x:x+w]

    if action == "register":
        user_dir = os.path.join(UPLOAD_FOLDER, username)
        os.makedirs(user_dir, exist_ok=True)
        count = len(os.listdir(user_dir))
        path = os.path.join(user_dir, f"{count+1}.jpg")
        cv2.imwrite(path, roi)

        # Retrain
        subprocess.run(["python", "train.py"])
        load_model()

        return render_template('index.html', result=f"✅ Registered '{username}' and retrained model.")

    elif action == "login":
        if recognizer is None:
            return render_template('index.html', result="⚠️ Model not trained yet. Please register first.")

        id_, conf = recognizer.predict(roi)
        if conf < 65:
            name = labels.get(id_, "Unknown")
            return render_template('index.html', result=f"✅ Login successful: {name} (Confidence: {round(conf, 2)})")
        else:
            return render_template('index.html', result="❌ Face not recognized.")

    return render_template('index.html', result="❌ Unknown action.")
    
if __name__ == '__main__':
    app.run(debug=True)
