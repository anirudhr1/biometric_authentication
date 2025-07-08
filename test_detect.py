import cv2

# Load the debug image
img = cv2.imread("static/debug.jpg")
if img is None:
    print("❌ debug.jpg not found!")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Detect faces
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(60, 60))
print(f"🔍 Faces detected: {len(faces)}")

# Draw and display
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Detected Faces", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
