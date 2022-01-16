import cv2
import torch
from config import *
import os

# DETECTOR = torch.hub.load('ultralytics/yolov5', 'custom', path=DETECTOR_PATH)
# detect()
import numpy as np

from config import DETECTOR_PATH, DETECTOR_THR, TARGET_CLASSES

DETECTOR = torch.hub.load('ultralytics/yolov5', 'custom', path=DETECTOR_PATH)


def extract_crops(frame, bboxes):
    return [frame[y0:y1, x0:x1] for x0, y0, x1, y1 in bboxes]


def detect(frame):  # TODO resize ?
    results = DETECTOR([frame])
    predicts = results.xyxy[0].cpu().numpy()
    predicts = predicts[predicts[:, 4] > DETECTOR_THR]
    predicts = predicts[np.isin(predicts[:, -1], TARGET_CLASSES)]

    bboxes = predicts[:, :4].astype(int)
    crops = extract_crops(frame, bboxes)
    return crops


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