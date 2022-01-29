import torch

ACCIDENT_CLF_PATH = '/home/gorodion/dtp/logs/checkpoints/best.pth'


NFRAMES = 300
STRIDE = 150
CAR_DET_INTERVAL = 25
ACCIDENT_THR = 0.9

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TRESHOLD = 0.002
IMG_SIZE = 224
BATCH_SIZE = 4
L = 3


DETECTOR_THR = 0.3
TARGET_CLASSES = [0, 3, 4]

PREDICTIONS_CSV = "predictions.csv"
