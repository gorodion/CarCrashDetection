import cv2
import torch

# DETECTOR = torch.hub.load('ultralytics/yolov5', 'custom', path=DETECTOR_PATH)

def detect_cars(cap: cv2.VideoCapture, start_pos):
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_pos)

