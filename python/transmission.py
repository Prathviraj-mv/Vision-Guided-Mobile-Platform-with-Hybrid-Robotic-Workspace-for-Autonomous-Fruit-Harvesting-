
import cv2
import requests
import time

# Raspberry Pi IP address
PI_URL = "http://10.179.82.120:8000/upload"

# External USB webcam
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 15)

if not cap.isOpened():
    raise RuntimeError("Webcam not detected")

print("Transmitting video to Raspberry Pi...")

session = requests.Session()

while True:

    ret, frame = cap.read()

    if not ret:
        continue

    # Compress frame
    success, encoded = cv2.imencode(
        ".jpg",
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, 65]
    )

    if not success:
        continue

    try:
        response = session.post(
            PI_URL,
            data=encoded.tobytes(),
            headers={"Content-Type": "image/jpeg"},
            timeout=3
        )

        print("Frame sent:", response.status_code)

    except requests.RequestException as e:
        print("Connection error:", e)
        time.sleep(1)

cap.release()
