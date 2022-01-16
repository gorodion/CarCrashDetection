import argparse
import os
import glob
import cv2
import CarsClassifier
import torch

TRESHOLD = 0.002
CLF_WEIGHTS = "Densenet169.pth"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    args = parser.parse_args()
    assert os.path.isdir(args.path), "Given path is not a directory"
    files = glob.glob(glob.escape(args.path) + "/*.mp4")
    for file in files:
        pass

    # PART 3
    model = CarsClassifier.Densenet169()
    model.load_state_dict(torch.load(CLF_WEIGHTS,  map_location=torch.device('cpu')))
    ds = CarsClassifier.CarsDatasetInference("videos")
    CarsClassifier.predict_emergency(model, ds, TRESHOLD)