import numpy as np
import mediapipe as mp
import cv2
import logging
import collections
from backend.frame_classifier import FrameClassifier

logger = logging.getLogger(__name__)

mp_face_detection = mp.solutions.face_detection


class VideoDeepfakeDetector:
    def __init__(self):
        self.face_detector = mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )
        self.classifier = FrameClassifier()
        # Smooth over the last 15 per-face scores for live webcam
        self.smoothing_window = collections.deque(maxlen=15)

    # ── helpers ──────────────────────────────────────────────────────────────

    def detect_faces(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detector.process(rgb)
        return results.detections if (results and results.detections) else []

    def _crop_face(self, frame, detection, target_size=(224, 224)):
        """
        Crop face with 20% padding on each side, clamped to image bounds.
        Returns a BGR numpy array of shape (target_size, target_size, 3), or None.
        """
        h, w = frame.shape[:2]
        box = detection.location_data.relative_bounding_box
        bx = int(box.xmin * w)
        by = int(box.ymin * h)
        bw = int(box.width * w)
        bh = int(box.height * h)

        # 20 % padding
        pad_x = int(bw * 0.20)
        pad_y = int(bh * 0.20)

        x1 = max(0, bx - pad_x)
        y1 = max(0, by - pad_y)
        x2 = min(w, bx + bw + pad_x)
        y2 = min(h, by + bh + pad_y)

        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            return None
        return cv2.resize(face, target_size)

    def _box_coords(self, frame, detection):
        """Return (x, y, w, h) in pixel coordinates for the UI overlay."""
        h, w = frame.shape[:2]
        box = detection.location_data.relative_bounding_box
        x = int(box.xmin * w)
        y = int(box.ymin * h)
        bw = int(box.width * w)
        bh = int(box.height * h)
        return x, y, bw, bh

    # ── single-frame classification ──────────────────────────────────────────

    def classify_frame(self, frame):
        """
        Detect all faces in frame, classify each, and return aggregate info.

        Returns:
            {
              "score": float | None,   # None when no face detected
              "faces": [{"bbox": [...], "score": float}, ...],
              "face_found": bool,
            }
        """
        detections = self.detect_faces(frame)
        faces = []

        for det in detections:
            crop = self._crop_face(frame, det)
            if crop is None:
                continue
            score = self.classifier.classify(crop)
            x, y, bw, bh = self._box_coords(frame, det)
            faces.append({"bbox": [x, y, bw, bh], "score": float(score)})

        if faces:
            avg = float(np.mean([f["score"] for f in faces]))
        else:
            avg = None

        return {"score": avg, "faces": faces, "face_found": bool(faces)}

    # ── uploaded video analysis ───────────────────────────────────────────────

    def detect(self, video_path: str):
        """
        Analyse an uploaded video: sample up to 40 evenly-spaced frames,
        classify each face crop, and return temporal aggregate metrics.
        """
        logger.info("Step 1/4: Opening video")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video: {video_path}")

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        logger.info(f"Step 2/4: Sampling from {len(frames)} total frames")

        all_scores = []
        all_faces = []

        if frames:
            num_samples = min(40, len(frames))
            indices = np.linspace(0, len(frames) - 1, num_samples, dtype=int)
            for i in indices:
                result = self.classify_frame(frames[i])
                all_faces.extend(result["faces"])
                # Only accumulate when a face was actually detected
                if result["score"] is not None:
                    all_scores.append(result["score"])

        logger.info(f"Step 3/4: Scored {len(all_scores)} frames with detected faces")

        if all_scores:
            mean_score   = float(np.mean(all_scores))
            median_score = float(np.median(all_scores))
            std_score    = float(np.std(all_scores))
            final_score  = mean_score  # mean over all valid frames
        else:
            mean_score = median_score = final_score = 0.5
            std_score = 0.0

        logger.info(
            f"Step 4/4: Aggregated – mean={mean_score:.4f}, "
            f"median={median_score:.4f}, final={final_score:.4f}"
        )

        return {
            "video_score":           final_score,
            "mean_fake_probability": mean_score,
            "median_fake_probability": median_score,
            "std_fake_probability":  std_score,
            "frame_scores":          all_scores,
            "frames_analyzed":       len(frames),
            "faces":                 all_faces,
            "reason": "Temporal facial artefact analysis — EfficientNet-B7 + MediaPipe",
        }

    # ── live-webcam helper ────────────────────────────────────────────────────

    def classify_frame_live(self, frame):
        """
        Classify a single live-webcam frame and maintain temporal smoothing.
        Returns the smoothed fake_probability plus raw face data.
        """
        result = self.classify_frame(frame)

        # Only push confident detections into the window
        if result["face_found"] and result["score"] is not None:
            self.smoothing_window.append(result["score"])

        smoothed = (
            float(np.mean(self.smoothing_window))
            if self.smoothing_window
            else 0.0
        )
        return {
            "score":   smoothed,
            "faces":   result["faces"],
            "face_found": result["face_found"],
        }
