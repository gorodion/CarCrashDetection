import glob
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn
import torchvision
from torchvision import transforms, models
from torch.utils.data import Dataset
from typing import List
from PIL import Image
import os
import cv2 as cv
from tqdm import tqdm
import multiprocessing as mp
import logging
from MyFancyLogger import init_logger
from config import *


class CarsDatasetInference(Dataset):
    def __init__(self, root_dir):
        self.files = glob.glob(glob.escape(root_dir) + "/*.jpg")
        self.transform = transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        image = Image.open(path)
        if self.transform:
            image = self.transform(image)
        return image, path


class Densenet169(nn.Module):
    def __init__(self):
        super().__init__()
        self.pretrained_model = models.densenet169(pretrained=False)
        feature_in = self.pretrained_model.classifier.in_features
        self.pretrained_model.classifier = nn.Linear(feature_in, 1)

    def forward(self, x):
        return self.pretrained_model(x)


def notify(predictions, paths):
    logger = init_logger("Cars classification")
    for i in range(len(predictions)):
        if predictions[i] == 1:
            logger.warning(f"{paths[i]}: спец. машина")
        else:
            logger.info(f"{paths[i]}: обыкновенная машина")


def predict_emergency(model, dataset, threshold):
    model.eval()
    indices = list(range(len(dataset)))
    testset = torch.utils.data.Subset(dataset, indices)
    num_workers = 0#mp.cpu_count()

    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                           num_workers=num_workers)
    total_preds = []
    total_paths = []
    for inputs, paths in testloader:
        inputs = inputs.to(DEVICE)

        with torch.no_grad():
            outputs = model(inputs).sigmoid().squeeze().detach().cpu().numpy()
            outputs = (outputs > threshold).astype(np.int8)
            if len(outputs.shape) == 0:
                continue
            total_preds.extend(outputs.tolist())
            total_paths.extend(paths)
    notify(total_preds, total_paths)
    if sum(total_preds) >= L:
        return 1
    return 0
