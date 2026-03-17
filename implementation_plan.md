# VERI-AI EDGE — 8-Hour Atomic Implementation Plan

> **Deadline**: ~8 hours from now  
> **Constraint**: No model training. Use only pre-trained weights (MesoNet / EfficientNet-B0 via `timm` / torchvision).

---

## Codebase Audit Summary

| Layer | File(s) | Current State |
|---|---|---|
| **FastAPI Backend** | [main.py](file:///d:/Code%20Projects/Vigyan-deepfake-detector/backend/main.py) | POST `/analyze/video`, `/analyze/text`, GET `/health`. File-upload only. |
| **Video Detector** | [video_detector.py](file:///d:/Code%20Projects/Vigyan-deepfake-detector/backend/video_detector.py) | MediaPipe face detection → hardcoded heuristic score (0.8 / 0.2). **No neural model loaded.** |
| **Fusion Engine** | [fusion_engine.py](file:///d:/Code%20Projects/Vigyan-deepfake-detector/backend/fusion_engine.py) | Simple average [(v + t) / 2](file:///d:/Code%20Projects/Vigyan-deepfake-detector/components/dashboard/SystemStatus.tsx#29-38). Single-line placeholder. |
| **Decision Engine** | [decision_engine.py](file:///d:/Code%20Projects/Vigyan-deepfake-detector/backend/decision_engine.py) | Threshold-based classifier (safe / suspicious / unknown). Functional. |
| **Explainability** | [explainability.py](file:///d:/Code%20Projects/Vigyan-deepfake-detector/backend/explainability.py) | Returns canned processing-step strings. Functional but static. |
| **Frontend** | [index.tsx](file:///d:/Code%20Projects/Vigyan-deepfake-detector/pages/index.tsx), [UploadPanel.tsx](file:///d:/Code%20Projects/Vigyan-deepfake-detector/components/dashboard/UploadPanel.tsx), [ResultsDashboard.tsx](file:///d:/Code%20Projects/Vigyan-deepfake-detector/components/dashboard/ResultsDashboard.tsx) | Upload → show result. No webcam UI, no bounding-box display, no live-stream polling. |
| **API Layer** | [api.ts](file:///d:/Code%20Projects/Vigyan-deepfake-detector/lib/api.ts) | Axios calls for video upload, text analysis, and health. No WebSocket / SSE / polling for live frames. |

### Critical Gaps
1. No real neural network inference — the "score" is a hardcoded 0.8/0.2.
2. No live-stream / webcam endpoint on the backend.
3. No webcam capture component on the frontend.
4. No bounding-box data returned from the backend, and no overlay renderer on the frontend.
5. The fusion engine is a single-line placeholder.

---

## User Review Required

> [!IMPORTANT]
> **Pre-trained model choice**: The plan uses **EfficientNet-B0** (`timm` library, ImageNet-pretrained) as a feature extractor with a custom binary classification head. This is a pragmatic choice because MesoNet weights are not available in a standard registry and the existing codebase has no `.pth` file. EfficientNet-B0 is lightweight enough for CPU edge inference (~5M params). If you have a specific `.pth` checkpoint you'd like to use instead, let me know and I'll swap the loading logic.

> [!WARNING]  
> **Live webcam stream**: The plan uses a polling approach (frontend captures webcam frames via `getUserMedia`, sends them as base64 JPEG to a new `/analyze/frame` endpoint at ~2-3 FPS). A WebSocket approach would be lower-latency but significantly more complex. The polling approach is more achievable within 8 hours.

---

## Proposed Changes — Atomic Task List

Each task below is **isolated and atomic** — it can be assigned to a single developer/agent and completed in one prompt without breaking the rest of the application.

---

### Phase 1: Backend — Neural Model Pipeline (Hours 0–3)

---

#### Task 1 · Add EfficientNet-B0 model loader module
**File**: `[NEW]` [frame_classifier.py](file:///d:/Code%20Projects/Vigyan-deepfake-detector/backend/frame_classifier.py)

1. Create `backend/frame_classifier.py`.
2. Import `timm` and `torch`.
3. Define a class `FrameClassifier` that in [__init__](file:///d:/Code%20Projects/Vigyan-deepfake-detector/backend/explainability.py#8-10):
   - Loads `efficientnet_b0` from `timm` with `pretrained=True`.
   - Replaces the classifier head with `nn.Linear(1280, 1)` (binary: real vs fake).
   - Sets the model to `eval()` mode.
   - Defines an image transform pipeline: resize to 224×224, normalize with ImageNet stats.
4. Expose a method `classify(frame: np.ndarray) -> float` that:
   - Receives a BGR OpenCV frame.
   - Converts to RGB PIL Image, applies transforms, runs `torch.no_grad()` inference.
   - Returns `torch.sigmoid(output).item()` — a 0–1 deepfake probability score.
5. No other file is modified.

---

#### Task 2 · Integrate FrameClassifier into VideoDeepfakeDetector
**File**: `[MODIFY]` [video_detector.py](file:///d:/Code%20Projects/Vigyan-deepfake-detector/backend/video_detector.py)

1. Import `FrameClassifier` from `backend.frame_classifier`.
2. In [__init__](file:///d:/Code%20Projects/Vigyan-deepfake-detector/backend/explainability.py#8-10), instantiate `self.classifier = FrameClassifier()`.
3. Add a new method `classify_frame(self, frame) -> dict` that:
   - Calls `self.detect_faces(frame)` to get face bounding boxes.
   - For each detected face, crops the face region from the frame.
   - Calls `self.classifier.classify(face_crop)` on each crop.
   - Returns `{"score": float, "faces": [{"bbox": [x,y,w,h], "score": float}]}`.
4. Modify the existing [detect()](file:///d:/Code%20Projects/Vigyan-deepfake-detector/backend/video_detector.py#32-85) method:
   - Instead of hardcoded 0.8/0.2, sample up to 10 evenly-spaced frames.
   - Call `classify_frame()` on each sampled frame.
   - Aggregate per-frame scores via `np.mean()` to produce `video_score`.
   - Include face bounding-box data in the return value.
5. No other file is modified.

---

#### Task 3 · Create the `/analyze/frame` endpoint for live single-frame analysis
**File**: `[MODIFY]` [main.py](file:///d:/Code%20Projects/Vigyan-deepfake-detector/backend/main.py)

1. Add a new POST route `@app.post("/analyze/frame")`.
2. Accept a JSON body with schema `{"image": "<base64-encoded JPEG>"}`.
3. Decode the base64 string → bytes → `cv2.imdecode` → OpenCV frame.
4. Call `video_detector.classify_frame(frame)`.
5. Run the result through [fusion_engine()](file:///d:/Code%20Projects/Vigyan-deepfake-detector/backend/fusion_engine.py#1-5) and [decision_engine()](file:///d:/Code%20Projects/Vigyan-deepfake-detector/backend/decision_engine.py#30-33).
6. Return JSON: `{ confidence, verdict, faces: [{bbox, score}], metrics }`.
7. Keep latency low by skipping [explainability()](file:///d:/Code%20Projects/Vigyan-deepfake-detector/backend/explainability.py#47-60) for live frames.
8. No other file is modified.

---

#### Task 4 · Upgrade the fusion engine to weighted scoring
**File**: `[MODIFY]` [fusion_engine.py](file:///d:/Code%20Projects/Vigyan-deepfake-detector/backend/fusion_engine.py)

1. Replace the single-line average with a function that accepts keyword arguments: `video_score`, `text_score`, optional `face_count`.
2. Implement weighted fusion:
   - If only video: return `video_score` directly.
   - If only text: return `text_score` directly.
   - If both: `0.7 * video_score + 0.3 * text_score`.
   - Boost score by 0.05 if `face_count == 0` (no faces = more suspicious for video).
3. Clamp output to `[0.0, 1.0]`.
4. Ensure backward compatibility — existing calls [fusion_engine(video_score, text_score)](file:///d:/Code%20Projects/Vigyan-deepfake-detector/backend/fusion_engine.py#1-5) must still work.
5. No other file is modified.

---

#### Task 5 · Update `/analyze/video` response to include bounding-box data
**File**: `[MODIFY]` [main.py](file:///d:/Code%20Projects/Vigyan-deepfake-detector/backend/main.py)

1. Modify the [analyze_video](file:///d:/Code%20Projects/Vigyan-deepfake-detector/backend/main.py#38-106) handler.
2. Extract [faces](file:///d:/Code%20Projects/Vigyan-deepfake-detector/backend/video_detector.py#19-31) data from `video_detector.detect()` result.
3. Add a `"faces"` key to the JSON response containing `[{bbox: [x,y,w,h], score: float}]`.
4. Update the `"metrics"` → `"modelUsed"` to `"EfficientNet-B0 + MediaPipe"`.
5. No other file is modified.

---

### Phase 2: Frontend — Live Webcam & Bounding Boxes (Hours 3–6)

---

#### Task 6 · Add `LiveStreamPanel` webcam component
**File**: `[NEW]` [LiveStreamPanel.tsx](file:///d:/Code%20Projects/Vigyan-deepfake-detector/components/dashboard/LiveStreamPanel.tsx)

1. Create a new React component `LiveStreamPanel`.
2. Use `navigator.mediaDevices.getUserMedia({ video: true })` to capture webcam.
3. Render a `<video>` element displaying the live feed.
4. On a `setInterval` (every ~333ms / 3 FPS):
   - Draw the current video frame to an off-screen `<canvas>`.
   - Call `canvas.toDataURL("image/jpeg", 0.7)` to get a base64 string.
   - POST it to `/analyze/frame` via the API layer.
   - Store the latest result in component state.
5. Display an overlay `<canvas>` on top of the video showing:
   - Bounding boxes around detected faces (green = authentic, red = deepfake).
   - Per-face score labels.
6. Show a "Start / Stop" toggle button.
7. Show the latest fusion score and verdict badge.
8. No other file is modified.

---

#### Task 7 · Add `analyzeFrame()` function to the API layer
**File**: `[MODIFY]` [api.ts](file:///d:/Code%20Projects/Vigyan-deepfake-detector/lib/api.ts)

1. Add a new interface `FrameAnalysisResult` extending the base fields with `faces: Array<{bbox: number[], score: number}>`.
2. Add an `async function analyzeFrame(imageBase64: string): Promise<FrameAnalysisResult>`.
3. POST to `/analyze/frame` with JSON body `{ image: imageBase64 }`.
4. Handle errors identically to the existing [analyzeText](file:///d:/Code%20Projects/Vigyan-deepfake-detector/lib/api.ts#106-130) function.
5. Update the [AnalysisResult](file:///d:/Code%20Projects/Vigyan-deepfake-detector/lib/api.ts#35-58) type to include an optional [faces](file:///d:/Code%20Projects/Vigyan-deepfake-detector/backend/video_detector.py#19-31) field.
6. No other file is modified.

---

#### Task 8 · Add a "Live" tab to the UploadPanel
**File**: `[MODIFY]` [UploadPanel.tsx](file:///d:/Code%20Projects/Vigyan-deepfake-detector/components/dashboard/UploadPanel.tsx)

1. Add a third tab type: `'live'` alongside `'video'` and `'text'`.
2. Add a camera icon SVG and a "Live" tab button.
3. When the `'live'` tab is active, render the `<LiveStreamPanel />` component.
4. Pass `onAnalysisComplete` through so that live results update the parent dashboard.
5. No other file is modified.

---

#### Task 9 · Add bounding-box overlay rendering to ResultsDashboard
**File**: `[MODIFY]` [ResultsDashboard.tsx](file:///d:/Code%20Projects/Vigyan-deepfake-detector/components/dashboard/ResultsDashboard.tsx)

1. Check if `result.faces` exists and has entries.
2. If present, render a new "Detected Faces" section below the metrics.
3. For each face, display:
   - A small row with the face index, bounding-box coordinates, and per-face deepfake score.
   - A colored indicator (green / yellow / red) based on the per-face score.
4. No other file is modified.

---

#### Task 10 · Add CSS styles for the live-stream panel and bounding-box overlays
**File**: `[MODIFY]` [globals.css](file:///d:/Code%20Projects/Vigyan-deepfake-detector/styles/globals.css)

1. Add a `.live-stream-container` class with `position: relative` and appropriate sizing.
2. Add a `.bbox-overlay` canvas style (absolute positioned, pointer-events: none).
3. Add `.face-label` badge styles for the per-face score labels.
4. Add a pulsing "live" indicator animation (`@keyframes pulse-live`).
5. No other file is modified.

---

### Phase 3: Integration, Polish & Verification (Hours 6–8)

---

#### Task 11 · Wire LiveStreamPanel results into the main dashboard state
**File**: `[MODIFY]` [index.tsx](file:///d:/Code%20Projects/Vigyan-deepfake-detector/pages/index.tsx)

1. Import `LiveStreamPanel` if needed for any state wiring.
2. Ensure that when a live-stream result arrives, the `ProcessingTimeline` updates with live-specific step labels (e.g., "Capturing frame", "Face detection", "EfficientNet inference", "Scoring").
3. Ensure the [ResultsDashboard](file:///d:/Code%20Projects/Vigyan-deepfake-detector/components/dashboard/ResultsDashboard.tsx#64-140) receives the latest live result including [faces](file:///d:/Code%20Projects/Vigyan-deepfake-detector/backend/video_detector.py#19-31) data.
4. No other file is modified.

---

#### Task 12 · Update the `/health` endpoint to report the loaded model
**File**: `[MODIFY]` [main.py](file:///d:/Code%20Projects/Vigyan-deepfake-detector/backend/main.py)

1. In the `/health` handler, add fields:
   - `"model"`: `"EfficientNet-B0"`.
   - `"liveStreamEnabled"`: `true`.
2. Update the [HealthResponse](file:///d:/Code%20Projects/Vigyan-deepfake-detector/lib/api.ts#59-65) type to include these new fields.
3. No other file is modified.

---

#### Task 13 · Update SystemStatus to show model info and live-stream capability
**File**: `[MODIFY]` [SystemStatus.tsx](file:///d:/Code%20Projects/Vigyan-deepfake-detector/components/dashboard/SystemStatus.tsx)

1. Add a new row "Model" showing `health.model`.
2. Add a new row "Live stream" showing a green badge if `health.liveStreamEnabled` is `true`.
3. Update the [HealthResponse](file:///d:/Code%20Projects/Vigyan-deepfake-detector/lib/api.ts#59-65) interface in [api.ts](file:///d:/Code%20Projects/Vigyan-deepfake-detector/lib/api.ts) to include `model?: string` and `liveStreamEnabled?: boolean`.
4. No other file is modified.

---

#### Task 14 · Update [requirements.txt](file:///d:/Code%20Projects/Vigyan-deepfake-detector/requirements.txt) with new Python dependencies
**File**: `[MODIFY]` requirements.txt

1. Add `timm>=0.9.0` (for EfficientNet-B0).
2. Ensure `torch` and `torchvision` are listed.
3. Ensure `Pillow` is listed (needed for image transforms).
4. No other file is modified.

---

#### Task 15 · End-to-end smoke test
**No file changes** — manual verification only.

1. Start the backend: `cd backend && uvicorn main:app --reload`.
2. Start the frontend: `npm run dev`.
3. **Test A — Video upload**: Upload a short .mp4 file. Verify the dashboard shows a confidence score, a verdict badge, frames analyzed count, and EfficientNet-B0 as the model used.
4. **Test B — Live webcam**: Switch to the "Live" tab. Verify webcam activates, bounding boxes appear on detected faces, and the confidence score + verdict update in real time (~2-3 FPS).
5. **Test C — Health check**: Verify the System Status panel shows "Online", lists "EfficientNet-B0" as the model, and shows "Live stream: Enabled".
6. **Test D — Text analysis**: Paste text into the text tab and submit. Verify it still works exactly as before (regression check).

---

## Verification Plan

### Automated Tests
- Run `python -c "from backend.frame_classifier import FrameClassifier; fc = FrameClassifier(); print('Model loaded OK')"` from the project root to verify model loads.
- Run `curl -X GET http://127.0.0.1:8000/health` to verify health endpoint.
- Run `curl -X POST http://127.0.0.1:8000/analyze/frame -H "Content-Type: application/json" -d '{"image": "<base64-test-string>"}'` to verify frame endpoint.

### Manual Verification (User)
- Open `http://localhost:3000` in a browser.
- Upload a video → confirm results with bounding-box data in the results panel.
- Click the "Live" tab → confirm webcam activates, bounding boxes render, scores update live.
- Check System Status → confirm model name and live-stream capability displayed.

---

## Time Budget

| Phase | Tasks | Hours |
|---|---|---|
| Phase 1: Backend ML Pipeline | Tasks 1–5 | ~3h |
| Phase 2: Frontend Live Stream | Tasks 6–10 | ~3h |
| Phase 3: Integration & Polish | Tasks 11–15 | ~2h |
| **Total** | **15 tasks** | **~8h** |
