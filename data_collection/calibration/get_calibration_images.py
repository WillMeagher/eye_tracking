import cv2
import time

import sys
sys.path.insert(0, '.')
from config import *

device = 1
cap = cv2.VideoCapture(device)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1280)

frame_count = 0

while cap.isOpened():

    # Capture frame-by-frame
    ret, frame = cap.read()

    # Check if frame is not empty
    if not ret:
        continue

    frame_count += 1

    if frame_count % 30 == 0:
        cv2.imwrite(config['path'] + 'data_collection/valibration/selfie_calibration_data/' + str(time.time()) + ".jpg", frame)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    cv2.waitKey(1)