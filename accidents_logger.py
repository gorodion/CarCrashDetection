import cv2
from config import *
from typing import Tuple
from collections import deque


def save_accident(frames: deque, out_dir: str, fps: float)->None:
    """
    This function saves the accident to the needed file
    :param frames: a deque of accident frames
    :param out_dir: path to video with the accident
    :param fps: fps in the video
    :return:
    """
    fheight, fwidth = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(out_dir, fourcc, fps, (fwidth, fheight))
    for frame in frames:
        out.write(frame)
    out.release()
