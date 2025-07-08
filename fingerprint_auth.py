import cv2
import os

def match_fingerprint(input_fp_path, known_folder="fingerprints/"):
    orb = cv2.ORB_create()

    input_img = cv2.imread(input_fp_path, 0)
    if input_img is None:
        print("Could not load input fingerprint.")
        return None

    kp1, des1 = orb.detectAndCompute(input_img, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    best_match = None
    best_score = float('inf')

    for file in os.listdir(known_folder):
        if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        known_img = cv2.imread(os.path.join(known_folder, file), 0)
        if known_img is None:
            continue

        kp2, des2 = orb.detectAndCompute(known_img, None)
        if des1 is None or des2 is None:
            continue

        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        score = sum(m.distance for m in matches[:30])  # lower = better

        if score < best_score:
            best_score = score
            best_match = file.split('.')[0]

    if best_match and best_score < 1000:
        print(f"✅ Fingerprint matched: {best_match} (score: {round(best_score, 2)})")
        return best_match
    else:
        print("❌ No fingerprint match found.")
        return None
