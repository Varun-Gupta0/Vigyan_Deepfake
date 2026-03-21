import numpy as np
import mediapipe as mp
import cv2
import logging
import collections
from backend.frame_classifier import FrameClassifier

logger = logging.getLogger(__name__)

mp_face_detection = mp.solutions.face_detection


class VideoDeepfakeDetector:
    def __init__(self, classifier=None):
        self.face_detector = mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )
        self.smoothing_window = collections.deque(maxlen=15)
        self.classifier = classifier or FrameClassifier()
        # Initialize Xception for Pro mode on demand or pre-load
        self.pro_classifier = FrameClassifier(model_type="xception_ffpp")

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

    def classify_frame(self, frame, mode: str = "lite"):
        """
        Detect all faces in frame, classify each, and return aggregate info.

        Returns:
            {
              "score": float | None,   # Aggregated score (normalized 0-1)
              "faces": [{"bbox": [...], "score": float}, ...],
              "face_found": bool,
            }
        """
        detections = self.detect_faces(frame)

        faces = []
        # Use pro_classifier if mode is pro
        clf = self.pro_classifier if mode == "pro" else self.classifier
        
        for det in detections:
            x, y, bw, bh = self._box_coords(frame, det)
            crop = self._crop_face(frame, det)
            if crop is not None:
                score = clf.classify(crop)
            else:
                score = 0.5
            faces.append({"bbox": [x, y, bw, bh], "score": score})

        if faces:
            avg = float(np.mean([f["score"] for f in faces]))
        else:
            avg = None

        return {"score": avg, "faces": faces, "face_found": bool(faces)}

    # ── uploaded video analysis ───────────────────────────────────────────────

    def detect(self, video_path: str, mode: str = "lite"):
        """
        Analyse an uploaded video: sample 3-5 evenly-spaced frames,
        classify each face crop, and return stabilized aggregate metrics.
        """
        logger.info(f"Step 1/4: Opening video")
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

        scores = []
        combined_faces = []

        num_samples = 0
        if frames:
            num_samples = min(5, max(3, len(frames)))
            indices = np.linspace(0, len(frames) - 1, num_samples, dtype=int)
            for i in indices:
                result = self.classify_frame(frames[i], mode=mode)
                combined_faces.extend(result["faces"])
                if result["score"] is not None:
                    scores.append(result["score"])

        logger.info(f"Step 3/4: Scored {len(scores)} frames with detected faces")

        if scores:
            final_score = float(np.mean(scores))
        else:
            final_score = 0.5

        final_score = float(np.clip(final_score, 0.0, 1.0))

        logger.info(
            f"Step 4/4: Aggregated – final={final_score:.4f}"
        )

        return {
            "score": final_score,
            "faces": combined_faces,
            "framesAnalyzed": len(scores)
        }

    # ── live-webcam helper ────────────────────────────────────────────────────

    def classify_frame_live(self, frame, mode: str = "lite"):
        """
        Classify a single live-webcam frame and maintain temporal smoothing.
        Returns the smoothed fake_probability plus raw face data.
        """
        result = self.classify_frame(frame, mode=mode)

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

