import React, { useState, useRef, useCallback, useEffect } from 'react';
import { analyzeFrame, FrameAnalysisResult } from '@/lib/api';

interface LiveStreamPanelProps {
  onAnalysisComplete?: (result: FrameAnalysisResult) => void;
}

const CameraIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path strokeLinecap="round" strokeLinejoin="round" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
  </svg>
);

const StopIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
    <rect x="6" y="6" width="12" height="12" rx="1" />
  </svg>
);

const LiveStreamPanel: React.FC<LiveStreamPanelProps> = ({ onAnalysisComplete }) => {
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [latestResult, setLatestResult] = useState<FrameAnalysisResult | null>(null);
  const [fps, setFps] = useState(0);

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const frameCountRef = useRef(0);
  const lastFpsUpdateRef = useRef(Date.now());

  const drawBoundingBoxes = useCallback((result: FrameAnalysisResult) => {
    const overlay = overlayCanvasRef.current;
    const video = videoRef.current;
    if (!overlay || !video) return;

    const ctx = overlay.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, overlay.width, overlay.height);

    if (!result.faces || result.faces.length === 0) return;

    const scaleX = overlay.width / video.videoWidth;
    const scaleY = overlay.height / video.videoHeight;

    result.faces.forEach((face) => {
      const [x, y, w, h] = face.bbox;
      const score = face.score;

      const color = score > 0.5 ? '#ef4444' : '#22c55e';
      const label = score > 0.5 ? `Deepfake: ${(score * 100).toFixed(0)}%` : `Authentic: ${((1 - score) * 100).toFixed(0)}%`;

      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.strokeRect(x * scaleX, y * scaleY, w * scaleX, h * scaleY);

      ctx.fillStyle = color;
      ctx.font = '12px sans-serif';
      ctx.fillText(label, x * scaleX, (y * scaleY) - 4);
    });
  }, []);

  const captureAndAnalyze = useCallback(async () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);

    const imageBase64 = canvas.toDataURL('image/jpeg', 0.7).split(',')[1];
    if (!imageBase64) return;

    try {
      const result = await analyzeFrame(imageBase64);
      setLatestResult(result);
      drawBoundingBoxes(result);
      onAnalysisComplete?.(result);
    } catch (err) {
      console.error('Frame analysis error:', err);
    }

    frameCountRef.current++;
    const now = Date.now();
    if (now - lastFpsUpdateRef.current >= 1000) {
      setFps(frameCountRef.current);
      frameCountRef.current = 0;
      lastFpsUpdateRef.current = now;
    }
  }, [drawBoundingBoxes, onAnalysisComplete]);

  const startStream = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      if (overlayCanvasRef.current && videoRef.current) {
        overlayCanvasRef.current.width = videoRef.current.videoWidth || 640;
        overlayCanvasRef.current.height = videoRef.current.videoHeight || 480;
      }

      intervalRef.current = setInterval(captureAndAnalyze, 333);
      setIsStreaming(true);
      setError(null);
    } catch (err) {
      setError('Failed to access camera. Please grant permission.');
      console.error('Camera error:', err);
    }
  }, [captureAndAnalyze]);

  const stopStream = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    setIsStreaming(false);
    setLatestResult(null);
  }, []);

  useEffect(() => {
    return () => {
      stopStream();
    };
  }, [stopStream]);

  const getVerdictColor = (verdict: string) => {
    switch (verdict) {
      case 'deepfake': return 'var(--red)';
      case 'authentic': return 'var(--green)';
      default: return 'var(--yellow)';
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      <div className="live-stream-container" style={{ position: 'relative', borderRadius: 'var(--radius)', overflow: 'hidden', background: 'var(--bg)', aspectRatio: '4/3' }}>
        <video
          ref={videoRef}
          playsInline
          muted
          style={{ width: '100%', height: '100%', objectFit: 'cover', display: isStreaming ? 'block' : 'none' }}
        />
        <canvas ref={canvasRef} style={{ display: 'none' }} />
        <canvas
          ref={overlayCanvasRef}
          className="bbox-overlay"
          style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', pointerEvents: 'none' }}
        />

        {!isStreaming && (
          <div style={{ position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 12 }}>
            <div style={{ color: 'var(--text-dim)' }}><CameraIcon /></div>
            <p style={{ fontSize: 13, color: 'var(--text-muted)', textAlign: 'center' }}>Click Start to begin live detection</p>
          </div>
        )}

        {isStreaming && (
          <div style={{ position: 'absolute', top: 12, left: 12, display: 'flex', alignItems: 'center', gap: 6, padding: '4px 8px', background: 'rgba(0,0,0,0.6)', borderRadius: 4 }}>
            <span className="pulse-dot" style={{ width: 8, height: 8, borderRadius: '50%', background: 'var(--red)' }} />
            <span style={{ fontSize: 11, color: 'white', fontWeight: 500 }}>LIVE</span>
            <span style={{ fontSize: 10, color: 'rgba(255,255,255,0.7)' }}>{fps} FPS</span>
          </div>
        )}
      </div>

      {error && (
        <div style={{ padding: '10px 14px', background: 'var(--red-dim)', border: '1px solid rgba(248,113,113,0.25)', borderRadius: 'var(--radius-sm)', fontSize: 12, color: 'var(--red)' }}>
          {error}
        </div>
      )}

      {latestResult && isStreaming && (
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: 12, background: 'var(--surface)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <div style={{ fontSize: 24, fontWeight: 600, fontFamily: 'var(--font-mono)' }}>
              {latestResult.confidence}%
            </div>
            <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
              Confidence
            </div>
          </div>
          <div style={{
            padding: '4px 10px',
            borderRadius: 4,
            fontSize: 12,
            fontWeight: 500,
            background: getVerdictColor(latestResult.verdict),
            color: 'white',
          }}>
            {latestResult.verdict.toUpperCase()}
          </div>
        </div>
      )}

      <button
        className={`btn ${isStreaming ? 'btn-danger' : 'btn-primary'}`}
        style={{ width: '100%', justifyContent: 'center' }}
        onClick={isStreaming ? stopStream : startStream}
      >
        {isStreaming ? <><StopIcon /> Stop Detection</> : <><CameraIcon /> Start Live Detection</>}
      </button>
    </div>
  );
};

export default LiveStreamPanel;
