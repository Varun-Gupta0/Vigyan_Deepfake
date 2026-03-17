import cv2
import numpy as np
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def detect_deepfake_video(video_path):
    """
    Detects deepfakes in a video using a lightweight pipeline.

    Args:
        video_path (str): Path to the video file.

    Returns:
        dict: A dictionary containing the deepfake score and reason.
    """

    try:
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        if not frames:
            return {"video_score": 0.0, "reason": "No frames extracted"}

        face_detector = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
        face_frames = []
        face_detection_failures = 0

        for frame in frames:
            results = face_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.detections:
                face_frames.extend(results.detections)
            else:
                face_detection_failures += 1

        if not face_frames:
            return {"video_score": 0.0, "reason": "No faces detected"}

        # Compute anomaly score based on frame inconsistencies, face detection failures, and abnormal brightness variation
        frame_consistency_score = len(face_frames) / len(frames)
        brightness_variation_score = np.std([np.mean(frame[:, :, 0]) for frame in frames])

        anomaly_score = (1 - frame_consistency_score) + (face_detection_failures / len(frames)) + brightness_variation_score

        # Determine reason based on score (adjust threshold as needed)
        if anomaly_score > 100:  # Example threshold
            reason = "Detected artifact"
        else:
            reason = "No significant artifacts detected"

        return {"video_score": float(anomaly_score), "reason": reason}

    except Exception as e:
        print(f"Error processing video: {e}")
        return {"video_score": 0.0, "reason": f"Error processing video: {e}"}
