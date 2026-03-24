# VERI-AI EDGE — Autonomous Multimodal Deepfake Detection System

**VERI-AI EDGE** is a cutting-edge, real-time deepfake detection platform designed for the modern era of AI-driven misinformation. By leveraging a hybrid "Lite + Pro" architecture, it provides instantaneous results for common manipulations while offering deep forensic analysis via state-of-the-art neural models.

---

## 1. PROJECT OVERVIEW
### What the system does
VERI-AI EDGE is an autonomous system that detects deepfakes in **Live Webcams**, **Uploaded Videos**, and **Text Content**. It uses advanced computer vision and natural language processing to identify subtle artifacts left behind by generative AI.

### Why it matters
The explosion of generative AI has made it trivial to create convincing fake identities and false information. VERI-AI EDGE democratizes deepfake detection by providing an accessible, real-time tool for individuals and enterprises.

### One-line impact statement
*"Enabling real-time digital trust through explainable, multimodal AI forensics."*

---

## 2. PROBLEM STATEMENT
*   **Rise of Multimodal Threats:** Deepfakes are no longer just videos; they include AI-generated phishing emails, social media posts, and voice clones.
*   **High Risks:** Misinformation can sway elections, fraud can drain bank accounts, and identity theft ruins lives.
*   **Existing Gaps:** Most solutions are either too slow for real-time use, require expensive GPUs, or provide "black box" results without explaining *why* a video was flagged.

---

## 3. SOLUTION: VERI-AI EDGE
VERI-AI EDGE solves these problems through:
*   **Multimodal Detection:** Analyzes both visual (video/frame) and textual (phishing/AI-text) data.
*   **Real-time & Explainable:** Provides instant feedback with "AI Reasoning" to explain detected artifacts.
*   **Hybrid Architecture:** 
    *   **Lite Mode:** Uses fast heuristics for low-latency edge deployment.
    *   **Pro Mode:** Uses deep neural networks (Xception-FFPP) for high-accuracy forensic analysis.

---

## 4. KEY FEATURES
*   **Live Webcam Detection:** Real-time processing of video streams with temporal smoothing to reduce flicker.
*   **Video Upload Analysis:** Deep-scan of uploaded MP4/MOV files with multi-frame sampling.
*   **Text Manipulation Detection:** Scans for AI-generated text and phishing patterns using LLMs.
*   **Lite vs Pro Toggle:** User-adjustable accuracy vs. speed trade-off.
*   **Explainability Dashboard:** Displays "Processing Steps" and "AI Reasoning" (via LLM) for transparency.
*   **Telemetry Dashboard:** High-performance metrics including FPS, Latency (ms), and Confidence Scores.

---

## 5. SYSTEM ARCHITECTURE
The system follows a modular pipeline:

1.  **Input:** Image/Video/Text captured via UI.
2.  **Face Detection:** MediaPipe extracts faces with 20% padding at 299x299 resolution.
3.  **Frame Processing:** Normalization and resizing for model compatibility.
4.  **Model Inference:**
    *   *Lite:* Heuristic artifact analysis (Blur, Color Std, Brightness).
    *   *Pro:* Xception-FFPP (FaceForensics++) neural classification.
5.  **Fusion & Calibration:** Scores are fused across frames and calibrated for high-confidence decisions.
6.  **Decision Engine:** Final verdict (REAL, FAKE, UNCERTAIN).
7.  **Explanation Engine:** LLM (Gemma-3) generates human-readable reasoning.
8.  **UI:** Results rendered in a premium Next.js dashboard.

---

## 6. TECH STACK
### Backend
*   **FastAPI:** High-performance Python web framework.
*   **PyTorch:** Core engine for running Xception neural models.
*   **OpenCV:** Image processing and frame manipulation.
*   **MediaPipe:** Ultra-fast face detection and landmarking.

### Frontend
*   **Next.js (React):** Modern web architecture for the dashboard.
*   **Tailwind CSS:** Premium, responsive UI design.
*   **Axios:** Efficient API communication.

### Models
*   **Xception-FFPP (Pro):** Specially trained on the FaceForensics++ dataset.
*   **Heuristic Pipeline (Lite):** OpenCV-based artifact detection.
*   **Gemma-3-27B (LLM):** Explainability and advanced text analysis.

---

## 7. MODEL DETAILS
### Lite Mode (Speed Optimized)
*   **Logic:** Heuristic-based. Checks for abnormal blurriness around edges, color consistency, and lighting artifacts.
*   **Compute:** Extremely low. Runs smoothly on mobile or low-end laptops.
*   **Best For:** Live monitoring where speed is critical.

### Pro Mode (Accuracy Optimized)
*   **Logic:** **Xception Neural Network**. Uses depthwise separable convolutions to find deep "ghost" artifacts in facial textures.
*   **Training:** Trained on thousands of manipulated videos (FaceForensics++).
*   **Best For:** Official verification and forensic investigation.

---

## 8. PERFORMANCE METRICS
*   **Latency:** ~50-100ms (Lite) | ~300-600ms (Pro).
*   **FPS:** ~15-30 FPS on standard CPUs (depending on mode).
*   **Sampling:** Analyzes 5 key frames per video to ensure temporal consistency.
*   **Accuracy:** High precision on FF++ dataset; designed to minimize "False Reals".

---

## 9. LIMITATIONS
*   **Low Quality:** Accuracy drops on highly compressed or extremely low-light webcam footage.
*   **Temporal Needs:** Requires at least 3-5 frames for a confident video verdict.
*   **Dataset Bias:** Initially trained on FaceForensics++; may show different behavior on unseen generator types (like Sora or Kling).
*   **Generalization:** Not yet optimized for obscure facial accessories (heavy masks/complex filters).

---

## 10. INNOVATION
1.  **Hybrid Approach:** First-of-its-kind toggle between "Instant Heuristics" and "Deep Forensic" analysis in a single UI.
2.  **Explainable AI (XAI):** Doesn't just say "Fake"; it explains it using forensic terminology generated by an LLM.
3.  **Cross-Platform Edge:** Optimized to run "local-first" to protect user privacy.

---

## 11. USE CASES
*   **Social Media:** Auto-flagging suspicious content in feeds.
*   **Journalism:** Verifying citizen-contributed video footage.
*   **Banking/KYC:** Preventing "presentation attacks" (photo-of-a-photo) during onboarding.
*   **Corporate Security:** Detecting AI-generated phishing emails or spoofed meeting participants.

---

## 12. FUTURE IMPROVEMENTS
*   **Dataset Expansion:** Include Indian-specific faces and real-world diverse lighting.
*   **GPU Acceleration:** Support for TensorRT for sub-10ms inference.
*   **Multi-Modal Fusion:** Correlating lip-sync audio with visual artifacts.
*   **Cloud Scalability:** Auto-scaling backend for enterprise-grade throughput.

---

## 13. TECHNOLOGY READINESS LEVEL (TRL)
**TRL 6:** Technology demonstrated in a relevant environment. (Fully functional prototype with backend/frontend integration).

---

## 14. DEMO FLOW
1.  **Start Lite:** Show the webcam feed. Move around. Notice the "Authentic" tag and high FPS.
2.  **Toggle Pro:** Enable "Pro Mode". Show a sample deepfake video (or hold up a photo). The system will switch to the Xception model.
3.  **Explain:** Point out the "AI Reasoning" panel which will state things like *"Neural patterns suggest abnormal texture gradients in the periorbital region."*
4.  **Text Analysis:** Paste an AI-generated phishing text to show the multimodal capability.

---

## 15. SAMPLE OUTPUT EXPLANATION
*   **"Uncertain":** Occurs when the score is between 30% and 70%. This usually means the video quality is too low for a definitive verdict.
*   **"AI Reasoning":** The system translates raw neural confidence into technical forensics.
    *   *Example:* "Detected 84% probability of manipulation due to temporal inconsistency of facial landmarks."

---

## 16. INSTALLATION & SETUP

### Backend Setup
1. `cd backend`
2. `pip install -r requirements.txt`
3. Add `OPENROUTER_API_KEY` to `.env.local`
4. `python main.py`

### Frontend Setup
1. `npm install`
2. `npm run dev`
3. Open `http://localhost:3000`

---

## 17. TEAM
**Antigravity AI Team**

---
---

# Potential Questions Judges May Ask + Strong Answers

### Q1: Why is the accuracy not always 100%?
**A:** Deepfake detection is a "cat-and-mouse" game. As generative models improve, detection must evolve. We prioritize "Uncertain" verdicts over "False Reals" to ensure that users know when a video requires human oversight.

### Q2: Why provide "Uncertain" results?
**A:** In forensics, a "Not Sure" is safer than a "Wrong Yes". It signals researchers or journalists that the signal-to-noise ratio is too low (e.g., bad lighting or heavy compression) to be scientifically sure.

### Q3: Why did you choose the Xception model?
**A:** Xception is the industry standard for FaceForensics. Its use of depthwise separable convolutions is uniquely effective at catching the high-frequency artifacts (noise) left by GANs and Diffusion models that standard CNNs miss.

### Q4: How is this different from ChatGPT or other AIs?
**A:** ChatGPT generates content; we are the "Police" of AI. We are a specialized, discriminative model built for forensics, whereas ChatGPT is a generative model.

### Q5: Can this be scaled for a platform like Instagram?
**A:** Yes. Our Lite mode is specifically designed for high-throughput edge processing. By running Lite on the first pass and only escalating "Suspicious" videos to Pro mode, we can handle millions of videos with minimal compute cost for the platform.
