import cv2
import numpy as np
import re

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_index = 0
    frame_skip = 10

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % frame_skip == 0:
            frames.append(frame)

        frame_index += 1

    cap.release()
    return frames

def read_text(file_path):
    with open(file_path, 'r') as file:
        text = file.read().replace('\n', '')
        text = re.sub(r'\s+', ' ', text).strip()
        return text

def preprocess_video(file):
    return extract_frames(file)

def preprocess_text(file):
    return read_text(file)
