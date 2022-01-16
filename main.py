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

model = Densenet169()
model.load_state_dict(torch.load(CLF_WEIGHTS, map_location=torch.device(DEVICE)))
model = model.to(DEVICE)
model.eval()
paths = []
predictions = []


def process_video(vid_path: str, dirname: str):
    logger = init_logger("Car accident detection")
    cap = cv2.VideoCapture(vid_path)
    assert cap.isOpened(), f'Video {vid_path} is not opened'
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    frames = deque(maxlen=NFRAMES)
    new_frames = 0
    cur_frame = 0
    is_accident = False
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = frame[..., ::-1] # BGR -> RGB
        frames.append(frame) # TODO resize?
        new_frames += 1
        cur_frame += 1
        if len(frames) == NFRAMES and new_frames >= STRIDE:
            logger.info('current frame', cur_frame) ###
            new_frames = 0
            is_accident = predict_accident(frames)
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
        is_accident = predict_accident(frames)

    if is_accident:
        start_frame = cur_frame - STRIDE # from the middle of interval
        detect_cars(cap, start_frame, dirname)
        ds = CarsDatasetInference(dirname)
        is_emergency = predict_emergency(model, ds, TRESHOLD)
        if is_emergency:
            logger.warn('accident found')
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
    files = glob.glob(glob.escape(args.path) + "/*.mp4")
    # Init model for emergency cars classification
    for file in files:
        save_to = os.path.basename(file).split('.')[0]
        process_video(file, save_to)
        paths.append(file)
    final_predictions = pd.DataFrame(data=zip(paths, predictions), columns=["path", "prediction"])
    final_predictions.to_csv(PREDICTIONS_CSV, index=False)

