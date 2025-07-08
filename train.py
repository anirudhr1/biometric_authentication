import cv2
import os
import numpy as np

# Path to the dataset
data_path = "faces"
faces = []
labels = []
label_map = {}  # name -> id
label_id = 0

# Traverse all user folders
for user_name in os.listdir(data_path):
    user_folder = os.path.join(data_path, user_name)
    if not os.path.isdir(user_folder):
        continue

    if user_name not in label_map:
        label_map[user_name] = label_id
        label_id += 1

    for img_name in os.listdir(user_folder):
        img_path = os.path.join(user_folder, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        faces.append(img)
        labels.append(label_map[user_name])

# Train the recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(labels))
recognizer.save("recognizer.yml")

# Save label map for reverse lookup
import pickle
with open("labels.pkl", "wb") as f:
    pickle.dump(label_map, f)

print("Training complete. Model saved as recognizer.yml.")
