# pyre-ignore-all-errors
import numpy as np
import mediapipe as mp
import cv2
import logging
import collections
import torch
from backend.frame_classifier import FrameClassifier

logger = logging.getLogger(__name__)

# ── CPU optimization ──────────────────────────────────────────────────────────
torch.set_num_threads(4)

# Use MediaPipe Solutions API (consistent with webcam_detector.py)
mp_face_detection = mp.solutions.face_detection


class VideoDeepfakeDetector:
    def __init__(self, classifier=None):
        # Initialize face detection (Solutions API)
        self.face_detector = mp_face_detection.FaceDetection(
            model_selection=1,  # 1 for full-range model (better for videos)
            min_detection_confidence=0.5
        )
        self.smoothing_window = collections.deque(maxlen=15)

        # Lite mode: lightweight heuristic pipeline (MediaPipe + OpenCV)
        self.lite_classifier = None  # Will use heuristic scoring

        # Pro mode: Xception-FFPP neural classifier (loaded once at startup)
        self.pro_classifier = classifier or FrameClassifier(model_type="xception_ffpp")

    def get_model_name(self, mode: str = "lite"):
        """Returns the active model name based on mode."""
        if mode == "pro":
            return "Xception-FFPP"
        return "Lite-Heuristic"

    # ── helpers ──────────────────────────────────────────────────────────────

    def detect_faces(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detector.process(rgb)
        return results.detections if (results and results.detections) else []

    def _crop_face(self, frame, detection, target_size=(299, 299)):
        """
        Crop face with 20% padding, clamped to image bounds.
        Default target_size is 299×299 (Xception native resolution).
        Returns a BGR numpy array or None.
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

    # ── heuristic scoring (lite mode) ─────────────────────────────────────

    def _heuristic_score(self, frame, crop):
        """
        Lightweight heuristic scoring using OpenCV metrics.
        Returns a fake probability between 0 and 1.
        """
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blur_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        brightness = float(np.mean(gray))
        color_std = float(np.std(crop))

        blur_score = 0.0
        if blur_var < 50:
            blur_score = 0.3
        elif blur_var < 100:
            blur_score = 0.15

        brightness_score = 0.0
        if brightness < 60 or brightness > 200:
            brightness_score = 0.2

        color_score = 0.0
        if color_std < 20:
            color_score = 0.15

        raw = blur_score + brightness_score + color_score
        return float(np.clip(raw, 0.0, 1.0))

    # ── batch classification for Pro mode ────────────────────────────────────

    def _batch_classify_pro(self, crops):
        """
        Classify multiple face crops in a single batched forward pass.
        Much faster than classifying one-by-one.
        Returns list of float scores.
        """
        if not crops:
            return []

        from PIL import Image

        clf = self.pro_classifier
        tensors = []
        for crop in crops:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            tensors.append(clf.transform(pil))

        # Stack into batch [N, C, H, W]
        batch = torch.stack(tensors).to(clf.device)

        with torch.no_grad():
            outputs = clf.model(batch)

        probs = torch.sigmoid(outputs).squeeze(-1)
        scores = probs.cpu().tolist()

        # Ensure list format even for single item
        if isinstance(scores, float):
            scores = [scores]

        for i, s in enumerate(scores):
            logger.info(f"[XCEPTION_BATCH] crop_{i}: prob={s:.4f}")

        return scores

    # ── single-frame classification ──────────────────────────────────────────

    def classify_frame(self, frame, mode: str = "lite"):
        """
        Detect all faces in frame, classify each, and return aggregate info.

        mode == "lite": uses heuristic scoring (fast, no neural model)
        mode == "pro":  uses Xception-FFPP neural classifier (batched)
        """
        detections = self.detect_faces(frame)
        faces = []

        if mode == "pro" and detections:
            # Collect all crops first, then batch-classify
            crops = []
            coords = []
            for det in detections:
                x, y, bw, bh = self._box_coords(frame, det)
                crop = self._crop_face(frame, det, target_size=(299, 299))
                coords.append((x, y, bw, bh))
                crops.append(crop)

            # Batch all valid crops through the neural model
            valid_indices = [i for i, c in enumerate(crops) if c is not None]
            valid_crops = [crops[i] for i in valid_indices]

            if valid_crops:
                scores = self._batch_classify_pro(valid_crops)
            else:
                scores = []

            # Build faces list
            score_idx = 0
            for i, (x, y, bw, bh) in enumerate(coords):
                if i in valid_indices and score_idx < len(scores):
                    s = scores[score_idx]
                    score_idx += 1
                else:
                    s = 0.5
                faces.append({"bbox": [x, y, bw, bh], "score": s})

        else:
            # Lite mode: heuristic per-face
            for det in detections:
                x, y, bw, bh = self._box_coords(frame, det)
                crop = self._crop_face(frame, det, target_size=(224, 224))
                if crop is not None:
                    score = self._heuristic_score(frame, crop)
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
        Analyse an uploaded video by sampling up to 5 evenly-spaced frames,
        detecting faces, classifying each crop, and averaging valid scores.

        Frames where no face is detected are skipped entirely so that the
        final score only reflects frames with actual face data.
        """
        logger.info(f"Step 1/4: Opening video (mode={mode})")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_video = cap.get(cv2.CAP_PROP_FPS) or 30.0

        # ── Always sample up to 5 frames evenly across the video ──────────
        max_frames = 5
        num_samples = min(max_frames, total_frames) if total_frames > 0 else 0

        # Calculate evenly-spaced frame indices
        if num_samples > 0:
            sample_indices = set(
                int(i) for i in np.linspace(0, total_frames - 1, num_samples)
            )
        else:
            sample_indices = set()

        logger.info(
            f"Step 2/4: Sampling {len(sample_indices)} from {total_frames} "
            f"total frames (video fps={fps_video:.0f})"
        )

        # Read ONLY the frames we need (skip the rest for speed)
        sampled_frames = []
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx in sample_indices:
                sampled_frames.append(frame)
                if len(sampled_frames) >= num_samples:
                    break  # Got all we need — stop reading early
            frame_idx += 1
        cap.release()

        # ── Classify each sampled frame ───────────────────────────────────
        scores = []          # Only scores from frames where a face was found
        combined_faces = []  # All face detections across frames

        for i, frame in enumerate(sampled_frames):
            result = self.classify_frame(frame, mode=mode)

            # Skip frames where no face was detected
            if not result.get("face_found"):
                logger.info(f"  Frame {i+1}/{len(sampled_frames)}: no face detected — skipping")
                continue

            # Collect face bounding boxes
            faces_list = result.get("faces")
            if isinstance(faces_list, list):
                combined_faces.extend(faces_list)

            # Only include valid scores (face was found and score is not None)
            frame_score = result.get("score")
            if frame_score is not None:
                scores.append(frame_score)
                logger.info(
                    f"  Frame {i+1}/{len(sampled_frames)}: score={frame_score:.4f} "
                    f"(faces={len(faces_list) if faces_list else 0})"
                )

        logger.info(f"Step 3/4: Scored {len(scores)} valid face-frames out of {len(sampled_frames)} sampled")

        # ── Final aggregation ─────────────────────────────────────────────
        if scores:
            final_score = sum(scores) / len(scores)
        else:
            # No faces detected in any frame — return neutral
            final_score = 0.5

        final_score = float(np.clip(final_score, 0.0, 1.0))

        frames_analyzed = len(scores)

        logger.info(
            f"Step 4/4: Aggregated – final={final_score:.4f} "
            f"(model={self.get_model_name(mode)}, frames_analyzed={frames_analyzed})"
        )

        return {
            "score": final_score,
            "faces": combined_faces,
            "framesAnalyzed": frames_analyzed,
        }

    # ── live-webcam helper ────────────────────────────────────────────────────

    def classify_frame_live(self, frame, mode: str = "lite"):
        """
        Classify a single live-webcam frame and maintain temporal smoothing.
        Returns the smoothed fake_probability plus raw face data.
        """
        result = self.classify_frame(frame, mode=mode)

        if result["face_found"] and result["score"] is not None:
            self.smoothing_window.append(result["score"])

        smoothed = (
            float(np.mean(self.smoothing_window))
            if self.smoothing_window
            else 0.0
        )
        return {
            "score":      smoothed,
            "faces":      result["faces"],
            "face_found": result["face_found"],
        }
