import cv2
import torch
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

def detect_cars(cap: cv2.VideoCapture, start_pos):
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_pos)

