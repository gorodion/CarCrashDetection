import torch

NFRAMES = 300
STRIDE = 150
CAR_DET_INTERVAL = 25
ACCIDENT_THR = 0.7
ACCIDENT_CLF_PATH = '/home/gorodion/dtp/logs/checkpoints/best.pth'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TRESHOLD = 0.002
CLF_WEIGHTS = "/home/gorodion/dtp/Densenet169_largeVal.pth"
IMG_SIZE = 224
BATCH_SIZE = 4
L = 3


DETECTOR_PATH = '/home/gorodion/dtp/Yolov5_DeepSort_Pytorch/yolov5/weights/X-704.pt'
DETECTOR_THR = 0.3
TARGET_CLASSES = [0, 3, 4]

PREDICTIONS_CSV = "predictions.csv"