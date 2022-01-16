import torch

ACCIDENT_CLF_PATH = './models/best.pth'
CLF_WEIGHTS = "./models/Densenet169.pth"
DETECTOR_PATH = './models/X-704.pt'

NFRAMES = 300
STRIDE = 150
CAR_DET_INTERVAL = 25
ACCIDENT_THR = 0.5

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TRESHOLD = 0.002
IMG_SIZE = 224
BATCH_SIZE = 4
L = 3


DETECTOR_THR = 0.3
TARGET_CLASSES = [0, 3, 4]

PREDICTIONS_CSV = "predictions.csv"
