import torch

NFRAMES = 300
STRIDE = 150
CAR_DET_INTERVAL = 30
ACCIDENT_THR = 0.7
ACCIDENT_CLF_PATH = './models/accident_clf.pth'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TRESHOLD = 0.002
CLF_WEIGHTS = "./models/Densenet169.pth"
IMG_SIZE = 224
BATCH_SIZE = 4
L = 10


DETECTOR_PATH = './models/X-704.pt'
DETECTOR_THR = 0.3
TARGET_CLASSES = [0, 3, 4]
