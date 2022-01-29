import argparse
import os
import glob
import cv2
import pandas as pd
from collections import deque
from config import *
from accident import predict_accident
from accident import ResNetTCN
import sys
from accidents_logger import save_accident

paths = []
predictions = []
sys.stdout = open('logs.txt', 'w')


def detect_accident(cap: cv2.VideoCapture, video_path: str,
                     save_to: str):
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
            new_frames = 0
            is_accident = predict_accident(ACCIDENT_CLF, frames)
            if is_accident:
                start_secs = int(cur_frame / fps)
                start_mm = start_secs // 60
                start_ss = start_secs % 60
                end_secs = int((cur_frame + NFRAMES) / fps)
                end_mm = end_secs // 60
                end_ss = end_secs % 60
                resulting_path = f'accident_{save_to}.avi'
                print(f'{video_path}: found accident on {start_mm:02d}:{start_ss:02d}-{end_mm}:{end_ss}\tsaved to: {resulting_path}')
                save_accident(frames, resulting_path, fps)
                break
    len_ = len(frames)
    if len_ < NFRAMES:
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
    is_accident, cur_frame = detect_accident(cap, vid_path, dirname)
    if is_accident:
        print(f'{vid_path}: accident found')
        predictions.append(1)
    else:
        print(f'{vid_path}: no accident found')
        predictions.append(0)
    cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    args = parser.parse_args()
    assert os.path.isdir(args.path), "Given path is not a directory"
    files = glob.glob(args.path + "/*.mp4")
    for file in files:
        save_to = os.path.basename(file).split('.')[0]
        os.makedirs(save_to, exist_ok=True)
        process_video(file, save_to)
        paths.append(file)
    final_predictions = pd.DataFrame(data=zip(paths, predictions), columns=["path", "prediction"])
    final_predictions.to_csv(PREDICTIONS_CSV, index=False)

