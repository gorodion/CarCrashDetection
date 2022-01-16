import torch

NFRAMES = 300
STRIDE = 150
CAR_DET_INTERVAL = 30
ACCIDENT_THR = 0.7
ACCIDENT_CLF_PATH = './accident_clf.pth'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')