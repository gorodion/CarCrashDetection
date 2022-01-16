import torch
import albumentations as albu
from albumentations.pytorch import ToTensorV2

from config import ACCIDENT_CLF_PATH, ACCIDENT_THR, NFRAMES, DEVICE

TRANSFORMS = albu.Compose([
    albu.Resize(224, 224),
    albu.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
    ToTensorV2()
])

ACCIDENT_CLF = torch.load(ACCIDENT_CLF_PATH, map_location=DEVICE).eval()

@torch.no_grad()
def predict_sample(model, X):
    return model(X.cuda()).cpu().sigmoid().item()

def predict_accident(frames: deque):
    assert len(frames) == NFRAMES
    frames = [TRANSFORMS(image=frame)['image'] for frame in frames]
    frames = torch.stack(frames)[None]
    prob = predict_sample(ACCIDENT_CLF, frames)
    print('Уверенность в аварии:', prob)
    return prob > ACCIDENT_THR