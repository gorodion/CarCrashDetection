import cv2
import torch
from config import *
import os

# DETECTOR = torch.hub.load('ultralytics/yolov5', 'custom', path=DETECTOR_PATH)
# detect()


def save_crops(crops, out_dir, frame_number):
    for i, crop in enumerate(crops):
        cv2.imwrite(os.path.join(out_dir, f"{frame_number}_{i}.jpg"), crop)


def detect_cars(cap: cv2.VideoCapture, start_pos, out_dir):
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_pos)
    counter = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if counter % CAR_DET_INTERVAL == 0:
            crops = detect(frame)
            save_crops(crops, out_dir, counter)