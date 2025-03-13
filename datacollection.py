import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os

# Replace this URL with the one shown on your phone's IP Webcam app
stream_url = "http://192.168.36.219:8080/video"  # Update with your phone's IP
cap = cv2.VideoCapture(stream_url)

# Check if the stream opened successfully
if not cap.isOpened():
    print("Error: Could not connect to the mobile camera stream.")
    print(" - Verify the IP Webcam app is running on your phone.")
    print(" - Check the URL matches the one shown in the app (e.g., http://192.168.x.x:8080/video).")
    print(" - Ensure your phone and computer are on the same Wi-Fi network.")
    exit()

detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300
counter = 0

folder = "Data/i love you"  # Directory for saving images
if not os.path.exists(folder):
    os.makedirs(folder)

while True:
    success, img = cap.read()
    if not success:
        print("Error: Could not read frame from the stream.")
        break

    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Crop the hand region with offset
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        imgCropShape = imgCrop.shape

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgWhite)

    cv2.imshow('Image', img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)

cap.release()
cv2.destroyAllWindows()