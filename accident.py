import torch
from torch.nn.utils import weight_norm
from torch import nn
from torchvision.models import resnet
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


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()


class GAP1d(nn.Module):
    'Global Adaptive Pooling + Flatten'

    def __init__(self, output_size=1):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool1d(output_size)
        self.flatten = nn.Flatten()

    def forward(self, x):
        return self.flatten(self.gap(x))


class TemporalBlock(nn.Module):
    def __init__(self, ni, nf, ks, stride, dilation, padding, dropout=0.0):
        super().__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(
                ni, nf, ks, stride=stride, padding=padding, dilation=dilation,
            ),
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = weight_norm(
            nn.Conv1d(
                nf, nf, ks, stride=stride, padding=padding, dilation=dilation,
            ),
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
        )
        self.downsample = nn.Conv1d(ni, nf, 1) if ni != nf else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


def temporal_conv_net(c_in, layers, ks=2, dropout=0.0):
    temp_layers = []
    for i, layer in enumerate(layers):
        dilation_size = 2 ** i
        ni = c_in if i == 0 else layers[i - 1]
        nf = layer
        temp_layers += [
            TemporalBlock(
                ni,
                nf,
                ks,
                stride=1,
                dilation=dilation_size,
                padding=(ks - 1) * dilation_size,
                dropout=dropout,
            ),
        ]
    return nn.Sequential(*temp_layers)


class TCN(nn.Module):
    def __init__(
            self,
            c_in,
            c_out,
            layers=[25] * 8,
            ks=7,
            conv_dropout=0.0,
            fc_dropout=0.0,
    ):
        super().__init__()
        self.norm = nn.BatchNorm1d(c_in)
        self.tcn = temporal_conv_net(c_in, layers, ks=ks, dropout=conv_dropout)
        self.gap = GAP1d()
        self.dropout = nn.Dropout(fc_dropout) if fc_dropout else None
        self.linear = nn.Linear(layers[-1], c_out)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        x = self.norm(x)
        x = self.tcn(x)
        x = self.gap(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.linear(x)


class ResNetTCN(nn.Module):
    def __init__(self):
        super().__init__()
        model = resnet.resnet18(pretrained=True)
        in_feats = model.fc.in_features
        model.fc = Identity()
        self.encoder = model
        self.tcn = TCN(in_feats, 1, layers=[68] * 8, ks=3)

    def forward(self, X):
        '[B, frames, C, H, W]'  # ;print(X.shape)
        B, F, C, H, W = X.shape
        X = X.reshape(-1, C, H, W)
        X = self.encoder(X)
        '[B*frames, in_feats]'  # ;print(X.shape)
        X = X.reshape(B, F, -1)
        '[B, frames, in_feats]'  # ;print(X.shape)
        X = X.transpose(1, 2)
        '[B, in_feats, frames]'  # ;print(X.shape)
        X = self.tcn(X)
        return X


@torch.no_grad()
def predict_sample(model, X):
    return model(X.cuda()).cpu().sigmoid().item()


def predict_accident(model, frames):
    """
    This function predicts if there was an accedent on a video
    :param model: model
    :param frames: video frames
    :return: label, 1 for accedents 0 for ordinary videos
    """
    assert len(frames) == NFRAMES
    frames = [TRANSFORMS(image=frame)['image'] for frame in frames]
    frames = torch.stack(frames)[None]
    prob = predict_sample(model, frames)
    print('accident confidence:', prob)
    return prob > ACCIDENT_THR