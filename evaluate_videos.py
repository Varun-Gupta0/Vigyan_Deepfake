import cv2
import sys
import numpy as np
import logging

# Suppress debug logs from the classifier to keep output clean
logging.getLogger("backend.frame_classifier").setLevel(logging.WARNING)

from backend.video_detector import VideoDeepfakeDetector
from backend.frame_classifier import FrameClassifier

def evaluate_video(video_path: str, max_frames=30):
    print(f"Loading {video_path}...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return []

    classifier = FrameClassifier()
    det = VideoDeepfakeDetector()
    det.classifier = classifier

    probabilities = []
    frames_extracted = 0

    while cap.isOpened() and frames_extracted < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        # The detector handles face extraction.
        # We classify the frame. If it finds a face, it returns a score.
        res = det.classify_frame(frame)
        score = res.get("score", None)
        
        if score is not None:
            probabilities.append(score)
            frames_extracted += 1
            print(f"Frame {frames_extracted}: fake_prob = {score:.4f}")

    cap.release()
    return probabilities

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 evaluate_videos.py <real_video.mp4> <fake_video.mp4>")
        sys.exit(1)

    real_video = sys.argv[1]
    fake_video = sys.argv[2]

    # Evaluate REAL video
    print("\nEvaluating REAL video...")
    real_probs = evaluate_video(real_video, 30)
    
    # Evaluate FAKE video
    print("\nEvaluating DEEPFAKE video...")
    fake_probs = evaluate_video(fake_video, 30)

    print("\n" + "="*40)
    print("FINAL REPORT")
    print("="*40)
    
    if real_probs:
        print("\nREAL video")
        print(f"avg_fake_prob = {np.mean(real_probs):.4f}")
        print(f"std = {np.std(real_probs):.4f}")
    else:
        print("\nREAL video: No faces detected or video could not be opened.")

    if fake_probs:
        print("\nFAKE video")
        print(f"avg_fake_prob = {np.mean(fake_probs):.4f}")
        print(f"std = {np.std(fake_probs):.4f}")
    else:
        print("\nFAKE video: No faces detected or video could not be opened.")

if __name__ == "__main__":
    main()
