import argparse
import os
import glob
import cv2
import pandas as pd
from collections import deque
from config import *
from accident import predict_accident
from detector import detect_cars
from CarsClassifier import predict_emergency, CarsDatasetInference, Densenet169
from MyFancyLogger import init_logger
from accident import ResNetTCN

paths = []
predictions = []
logger = init_logger("Car accident detection")


def detect_accedent(cap: cv2.VideoCapture):
    ACCIDENT_CLF = ResNetTCN()
    ACCIDENT_CLF.load_state_dict(torch.load(ACCIDENT_CLF_PATH, map_location=DEVICE)['model_state_dict'])
    ACCIDENT_CLF.eval()
    ACCIDENT_CLF.to(DEVICE)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frames = deque(maxlen=NFRAMES)
    new_frames = 0
    cur_frame = 0
    is_accident = False
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = frame[..., ::-1]  # BGR -> RGB
        frames.append(frame)  # TODO resize?
        new_frames += 1
        cur_frame += 1
        if len(frames) == NFRAMES and new_frames >= STRIDE:
            logger.info(f'current frame {cur_frame}')  ###
            new_frames = 0
            is_accident = predict_accident(ACCIDENT_CLF, frames)
            if is_accident:
                secs = int(cur_frame / fps)
                mm = secs // 60
                ss = secs % 60
                logger.warn(f'found accident on {mm:02d}:{ss:02d}')
                break

    len_ = len(frames)
    if len_ < NFRAMES:
        logger.info(f'short video {int(len_ / fps)} seconds')
        frames += [frames[-1]] * (NFRAMES - len_)
        is_accident = predict_accident(ACCIDENT_CLF, frames)
    return is_accident, cur_frame


def process_video(vid_path: str, dirname: str)->None:
    """
    This function generates prediction for a single video
    :param vid_path: path to video
    :param dirname: directory to save crops
    :return: prediction
    """
    cap = cv2.VideoCapture(vid_path)
    assert cap.isOpened(), f'Video {vid_path} is not opened'
    is_accident, cur_frame = detect_accedent(cap)
    if is_accident:
        logger.warn('detecting cars')
        start_frame = cur_frame - STRIDE # from the middle of interval
        detect_cars(cap, start_frame, dirname)
        ds = CarsDatasetInference(dirname)
        model = Densenet169()
        model.load_state_dict(torch.load(CLF_WEIGHTS, map_location=torch.device(DEVICE)))
        model = model.to(DEVICE)
        model.eval()
        is_emergency = predict_emergency(model, ds, TRESHOLD)
        if is_emergency:
            logger.warn('accident and emergency found')
            predictions.append(1)
        else:
            predictions.append(0)
    else:
        logger.info('no accident found')
        predictions.append(0)
    cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    args = parser.parse_args()
    assert os.path.isdir(args.path), "Given path is not a directory"
    files = glob.glob(args.path + "/*.mp4")
    for file in files:
        logger.info(f'processing {file}')
        save_to = os.path.basename(file).split('.')[0]
        os.makedirs(save_to, exist_ok=True)
        process_video(file, save_to)
        paths.append(file)
    final_predictions = pd.DataFrame(data=zip(paths, predictions), columns=["path", "prediction"])
    final_predictions.to_csv(PREDICTIONS_CSV, index=False)

