import argparse
import os
import glob
import cv2
import pandas as pd
from collections import deque
from config import *
from accident import predict_accident
from accident import ResNetTCN
from accidents_logger import save_accident

paths = []
predictions = []
log = open('logs.txt', 'w')
ACCIDENT_CLF = ResNetTCN()
ACCIDENT_CLF.load_state_dict(torch.load(ACCIDENT_CLF_PATH, map_location=DEVICE)['model_state_dict'])
ACCIDENT_CLF.eval()
ACCIDENT_CLF.to(DEVICE)


def detect_accident(cap: cv2.VideoCapture):
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
            new_frames = 0
            is_accident = predict_accident(ACCIDENT_CLF, frames)
            if is_accident:
                break
    len_ = len(frames)
    if len_ < NFRAMES:
        frames += [frames[-1]] * (NFRAMES - len_)
        is_accident = predict_accident(ACCIDENT_CLF, frames)
    return is_accident, cur_frame, frames


def process_video(vid_path: str)->None:
    """
    This function generates prediction for a single video
    :param vid_path: path to video
    :param dirname: directory to save crops
    :return: prediction
    """
    cap = cv2.VideoCapture(vid_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    assert cap.isOpened(), f'Video {vid_path} is not opened'
    is_accident, cur_frame, frames = detect_accident(cap)
    if is_accident:
        save_to = os.path.basename(vid_path).split('.')[0]
        start_secs = int(cur_frame / fps)
        start_mm = start_secs // 60
        start_ss = start_secs % 60
        end_secs = int((cur_frame + NFRAMES) / fps)
        end_mm = end_secs // 60
        end_ss = end_secs % 60
        resulting_path = f'accident_{save_to}.mp4'
        print(f'{vid_path}: found accident on {start_mm:02d}:{start_ss:02d}-{end_mm}:{end_ss}\tsaved to: {resulting_path}',
              file=log)
        save_accident(frames, resulting_path, fps)
        print(f'{vid_path}: accident found', file=log)
        predictions.append(1)
    else:
        print(f'{vid_path}: no accident found', file=log)
        predictions.append(0)
    cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    args = parser.parse_args()
    assert os.path.isdir(args.path), "Given path is not a directory"
    files = glob.glob(args.path + "/*.mp4")
    for file in files:
        process_video(file)
        paths.append(file)
    final_predictions = pd.DataFrame(data=zip(paths, predictions), columns=["path", "prediction"])
    final_predictions.to_csv(PREDICTIONS_CSV, index=False)
    log.close()

