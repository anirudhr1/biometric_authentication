import cv2
import os
import sys

# ✅ Get username from command-line or input
if len(sys.argv) >= 2:
    username = sys.argv[1]
else:
    username = input("Enter username: ")

save_path = os.path.join("faces", username)
os.makedirs(save_path, exist_ok=True)

cam = cv2.VideoCapture(0)
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

count = 0
while True:
    ret, img = cam.read()
    if not ret:
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        count += 1
        face_img = gray[y:y+h, x:x+w]
        cv2.imwrite(f"{save_path}/{str(count)}.jpg", face_img)
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow("Registering Face - Press Q to Quit", img)
    if cv2.waitKey(1) & 0xFF == ord('q') or count >= 50:
        break

cam.release()
cv2.destroyAllWindows()
