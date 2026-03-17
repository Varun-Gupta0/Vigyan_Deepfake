import cv2
import numpy as np
import mediapipe as mp
from collections import deque

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils


class WebcamDeepfakeDetector:
    def __init__(self, buffer_size=30):
        self.face_detector = mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )
        self.buffer_size = buffer_size
        self.face_history = deque(maxlen=buffer_size)
        self.brightness_history = deque(maxlen=buffer_size)
        self.frame_count = 0
        self.confidence_threshold = 0.65

    def analyze_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detector.process(rgb)

        has_face = results.detections is not None and len(results.detections) > 0

        self.face_history.append(1 if has_face else 0)

        brightness = np.mean(frame)
        self.brightness_history.append(brightness)

        self.frame_count += 1

        return has_face, results.detections

    def compute_anomaly_score(self):
        if self.frame_count < 5:
            return 0.0, "Collecting frames..."

        face_consistency = np.mean(self.face_history)

        brightness_var = (
            np.std(self.brightness_history) if len(self.brightness_history) > 1 else 0
        )

        detection_fail_rate = 1 - face_consistency

        anomaly_score = (detection_fail_rate * 0.4) + (brightness_var * 0.01)

        if anomaly_score > self.confidence_threshold:
            reason = "High artifact likelihood detected"
            is_deepfake = True
        else:
            reason = "No significant artifacts detected"
            is_deepfake = False

        return anomaly_score, reason, is_deepfake

    def draw_results(self, frame, detections, anomaly_score, is_deepfake):
        h, w = frame.shape[:2]

        if detections:
            for detection in detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                bw = int(bbox.width * w)
                bh = int(bbox.height * h)

                color = (0, 0, 255) if is_deepfake else (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)

        label = "DEEPFAKE" if is_deepfake else "AUTHENTIC"
        color = (0, 0, 255) if is_deepfake else (0, 255, 0)

        cv2.putText(
            frame,
            f"{label}: {anomaly_score:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
        )

        cv2.putText(
            frame,
            f"Frames: {self.frame_count}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )

        return frame


def run_live_detection(camera_index=0):
    detector = WebcamDeepfakeDetector(buffer_size=30)

    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}")
        return

    print("Starting live deepfake detection. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame")
            break

        has_face, detections = detector.analyze_frame(frame)

        anomaly_score, reason, is_deepfake = detector.compute_anomaly_score()

        output_frame = detector.draw_results(
            frame, detections, anomaly_score, is_deepfake
        )

        cv2.imshow("Vigyan Deepfake Detector (Live Webcam)", output_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Live detection stopped.")


if __name__ == "__main__":
    run_live_detection()
