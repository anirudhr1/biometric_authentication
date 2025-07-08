import cv2
import pickle
import os
import sys
import logging
import tkinter as tk
from tkinter import messagebox

# 🔍 Setup logging
logging.basicConfig(filename="login_log.txt", level=logging.DEBUG)
logging.debug("login.py started")

# 📦 Support PyInstaller .exe resource access
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS  # used by PyInstaller
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# 🧠 Load face detector and recognizer
face_cascade_path = resource_path("haarcascade_frontalface_default.xml")
recognizer_path = resource_path("recognizer.yml")
labels_path = resource_path("labels.pkl")

face_cascade = cv2.CascadeClassifier(face_cascade_path)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(recognizer_path)

with open(labels_path, "rb") as f:
    labels = pickle.load(f)
    labels = {v: k for k, v in labels.items()}  # id → name

# 🧭 Setup GUI (hidden root)
tk.Tk().withdraw()

# 🎥 Start webcam
cap = cv2.VideoCapture(0)
recognized = False

while True:
    ret, frame = cap.read()
    if not ret:
        logging.debug("Camera read failed.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        id_, confidence = recognizer.predict(roi_gray)
        logging.debug(f"Detected face. ID: {id_}, Confidence: {confidence}")

        if confidence < 65:
            name = labels.get(id_, "Unknown")
            recognized = True
            logging.debug(f"Login success: {name} with confidence {confidence}")
            cap.release()
            cv2.destroyAllWindows()
            messagebox.showinfo("Login Successful", f"✅ Welcome, {name}!\nConfidence: {round(confidence, 2)}")
            break

    if recognized:
        break

    cv2.imshow("Face Login - Press Q to Cancel", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        logging.debug("User quit login window.")
        break

cap.release()
cv2.destroyAllWindows()

if not recognized:
    logging.debug("Login failed. Face not recognized.")
    messagebox.showerror("Login Failed", "❌ Face not recognized.")
