import React, { useState, useCallback, useRef, useEffect } from 'react';
import axios, { CancelTokenSource } from 'axios';
import { AnalysisResult, analyzeText, analyzeTextFile, FrameAnalysisResult } from '@/lib/api';
import LiveStreamPanel from './LiveStreamPanel';

interface UploadPanelProps {
  onAnalysisComplete?: (result: AnalysisResult | FrameAnalysisResult, steps: string[]) => void;
}

type TabType = 'video' | 'text' | 'live';

const VideoIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path strokeLinecap="round" strokeLinejoin="round" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
  </svg>
);
const TextIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
  </svg>
);
const CameraIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path strokeLinecap="round" strokeLinejoin="round" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
  </svg>
);
const UploadIcon = () => (
  <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
    <path strokeLinecap="round" strokeLinejoin="round" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
  </svg>
);
const XIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
  </svg>
);

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';

const formatFileSize = (bytes: number): string => {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
};

const UploadPanel: React.FC<UploadPanelProps> = ({ onAnalysisComplete }) => {
  const [activeTab, setActiveTab] = useState<TabType>('video');
  const [loading, setLoading] = useState(false);
  const [textInput, setTextInput] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState(false);

  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [filePreview, setFilePreview] = useState<string | null>(null);
  const [uploadPct, setUploadPct] = useState(0);
  const [isUploading, setIsUploading] = useState(false);

  const [textFile, setTextFile] = useState<File | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const textFileInputRef = useRef<HTMLInputElement>(null);
  const cancelTokenRef = useRef<CancelTokenSource | null>(null);

  useEffect(() => {
    return () => {
      if (filePreview) URL.revokeObjectURL(filePreview);
    };
  }, [filePreview]);

  const handleCancel = useCallback(() => {
    cancelTokenRef.current?.cancel('Cancelled by user');
    setIsUploading(false);
    setLoading(false);
    setUploadPct(0);
    setSelectedFile(null);
    if (filePreview) { URL.revokeObjectURL(filePreview); setFilePreview(null); }
  }, [filePreview]);

  const handleVideoSelect = useCallback((file: File) => {
    if (!file.type.startsWith('video/')) { setError('Please select a valid video file'); return; }
    if (file.size > 100 * 1024 * 1024) { setError('File must be under 100 MB'); return; }
    setError(null);
    setSelectedFile(file);
    setFilePreview(URL.createObjectURL(file));
  }, []);

  const handleVideoUpload = useCallback(async () => {
    if (!selectedFile) return;
    setLoading(true);
    setIsUploading(true);
    setUploadPct(0);
    setError(null);

    cancelTokenRef.current = axios.CancelToken.source();
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await axios.post<AnalysisResult>(
        `${API_BASE_URL}/analyze/video`,
        formData,
        {
          headers: { 'Content-Type': 'multipart/form-data' },
          cancelToken: cancelTokenRef.current.token,
          timeout: 120000,
          onUploadProgress: (e) => {
            const pct = e.total ? Math.round((e.loaded * 100) / e.total) : 0;
            setUploadPct(pct);
          },
        }
      );
      const data = response.data;
      onAnalysisComplete?.(data, data.processing_steps ?? []);
      setSelectedFile(null);
      if (filePreview) { URL.revokeObjectURL(filePreview); setFilePreview(null); }
    } catch (err) {
      if (axios.isCancel(err)) { setError('Upload cancelled'); }
      else { setError('Video analysis failed. Please check the backend is running.'); }
    } finally {
      setLoading(false);
      setIsUploading(false);
      setUploadPct(0);
      cancelTokenRef.current = null;
    }
  }, [selectedFile, filePreview, onAnalysisComplete]);

  const handleTextFileSelect = useCallback(async (file: File) => {
    const ok = file.type.startsWith('text/') || file.name.endsWith('.txt') || file.name.endsWith('.md');
    if (!ok) { setError('Please select a .txt or .md file'); return; }
    try {
      const text = await file.text();
      setTextInput(text);
      setTextFile(file);
      setError(null);
    } catch { setError('Failed to read text file'); }
  }, []);

  const handleTextAnalyze = useCallback(async () => {
    if (!textInput.trim()) return;
    setLoading(true);
    setError(null);
    cancelTokenRef.current = axios.CancelToken.source();

    try {
      const data = await analyzeText(textInput, cancelTokenRef.current);
      onAnalysisComplete?.(data, data.processing_steps ?? []);
      setTextInput('');
      setTextFile(null);
    } catch (err: any) {
      if (axios.isCancel(err)) { setError('Analysis cancelled'); }
      else { setError(err.message || 'Text analysis failed'); }
    } finally {
      setLoading(false);
      cancelTokenRef.current = null;
    }
  }, [textInput, onAnalysisComplete]);

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault(); e.stopPropagation();
    setDragActive(e.type === 'dragenter' || e.type === 'dragover');
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault(); e.stopPropagation();
    setDragActive(false);
    const file = e.dataTransfer.files?.[0];
    if (!file) return;
    activeTab === 'video' ? handleVideoSelect(file) : handleTextFileSelect(file);
  }, [activeTab, handleVideoSelect, handleTextFileSelect]);

  /* ── styles ─────────────────────────────────────────────────────────────── */
  const tabStyle = (active: boolean): React.CSSProperties => ({
    flex: 1,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 6,
    padding: '8px 0',
    border: 'none',
    borderBottom: `2px solid ${active ? 'var(--accent)' : 'transparent'}`,
    background: 'transparent',
    color: active ? 'var(--accent)' : 'var(--text-muted)',
    fontSize: 13,
    fontWeight: 500,
    cursor: 'pointer',
    transition: 'all 0.2s',
    fontFamily: 'var(--font-sans)',
  });

  return (
    <div className="card" style={{ padding: 0 }}>
      {/* Header */}
      <div style={{ padding: '16px 20px 0' }}>
        <div className="section-title">
          <UploadIcon />
          Upload &amp; Analyze
        </div>

        {/* Tabs */}
        <div style={{ display: 'flex', marginTop: 16, borderBottom: '1px solid var(--border)' }}>
          <button style={tabStyle(activeTab === 'video')} onClick={() => { setActiveTab('video'); setError(null); }}>
            <VideoIcon /> Video
          </button>
          <button style={tabStyle(activeTab === 'text')} onClick={() => { setActiveTab('text'); setError(null); }}>
            <TextIcon /> Text
          </button>
          <button style={tabStyle(activeTab === 'live')} onClick={() => { setActiveTab('live'); setError(null); }}>
            <CameraIcon /> Live
          </button>
        </div>
      </div>

      {/* Content */}
      <div style={{ padding: 20 }}>

        {/* ── VIDEO TAB ─────────────────────────────────────────────────── */}
        {activeTab === 'video' ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            {!selectedFile ? (
              <div
                onDragEnter={handleDrag} onDragLeave={handleDrag}
                onDragOver={handleDrag} onDrop={handleDrop}
                style={{
                  border: `2px dashed ${dragActive ? 'var(--accent)' : 'var(--border)'}`,
                  borderRadius: 'var(--radius)',
                  padding: '40px 24px',
                  textAlign: 'center',
                  background: dragActive ? 'var(--accent-dim)' : 'transparent',
                  transition: 'all 0.2s',
                  cursor: 'pointer',
                }}
                onClick={() => !loading && fileInputRef.current?.click()}
              >
                {loading ? (
                  <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 12 }}>
                    <div style={{
                      width: 36, height: 36,
                      border: '2px solid var(--border)',
                      borderTop: '2px solid var(--accent)',
                      borderRadius: '50%',
                    }} className="animate-spin" />
                    <span style={{ fontSize: 12, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                      ANALYZING VIDEO…
                    </span>
                  </div>
                ) : (
                  <>
                    <div style={{ color: 'var(--text-dim)', marginBottom: 12 }}><UploadIcon /></div>
                    <p style={{ fontSize: 13, color: 'var(--text-muted)', marginBottom: 4 }}>
                      Drop a video file or click to browse
                    </p>
                    <p style={{ fontSize: 11, color: 'var(--text-dim)' }}>MP4, AVI, MOV — max 100 MB</p>
                  </>
                )}
                <input ref={fileInputRef} type="file" accept="video/*" style={{ display: 'none' }}
                  onChange={(e) => e.target.files?.[0] && handleVideoSelect(e.target.files[0])}
                  disabled={loading} />
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                {/* File info */}
                <div style={{
                  display: 'flex', alignItems: 'center', gap: 12,
                  padding: 12, background: 'var(--surface)',
                  borderRadius: 'var(--radius-sm)', border: '1px solid var(--border)',
                }}>
                  <div style={{
                    width: 52, height: 52, borderRadius: 6,
                    background: 'var(--bg)', overflow: 'hidden', flexShrink: 0,
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                  }}>
                    {filePreview ? (
                      <video src={filePreview} style={{ width: '100%', height: '100%', objectFit: 'cover' }} muted />
                    ) : (
                      <div style={{ color: 'var(--text-dim)' }}><VideoIcon /></div>
                    )}
                  </div>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <p style={{ fontSize: 13, fontWeight: 500, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {selectedFile.name}
                    </p>
                    <p style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 2 }}>
                      {formatFileSize(selectedFile.size)}
                    </p>
                  </div>
                  {!isUploading && (
                    <button className="btn btn-ghost" style={{ padding: 6 }}
                      onClick={() => { setSelectedFile(null); if (filePreview) { URL.revokeObjectURL(filePreview); setFilePreview(null); } }}>
                      <XIcon />
                    </button>
                  )}
                </div>

                {/* Upload progress */}
                {isUploading && (
                  <div style={{ padding: '10px 0' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: 'var(--text-muted)', marginBottom: 6 }}>
                      <span>Uploading…</span>
                      <span className="mono">{uploadPct}%</span>
                    </div>
                    <div className="progress-track">
                      <div className="progress-fill" style={{ width: `${uploadPct}%` }} />
                    </div>
                  </div>
                )}

                {/* Action buttons */}
                {isUploading ? (
                  <button className="btn btn-danger" style={{ width: '100%', justifyContent: 'center' }} onClick={handleCancel}>
                    Cancel Upload
                  </button>
                ) : (
                  <button className="btn btn-primary" style={{ width: '100%', justifyContent: 'center' }}
                    onClick={handleVideoUpload} disabled={loading}>
                    {loading ? 'Analyzing…' : 'Start Analysis'}
                  </button>
                )}
              </div>
            )}
          </div>
        ) : (
          /* ── TEXT TAB ─────────────────────────────────────────────────── */
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            <textarea
              value={textInput}
              onChange={(e) => setTextInput(e.target.value)}
              placeholder="Paste text content for deepfake or phishing detection…"
              rows={7}
              disabled={loading}
            />

            {/* File upload option */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <input ref={textFileInputRef} type="file" accept=".txt,.md,text/*" style={{ display: 'none' }}
                onChange={(e) => e.target.files?.[0] && handleTextFileSelect(e.target.files[0])}
                disabled={loading} />
              <button className="btn btn-ghost" style={{ fontSize: 12 }}
                onClick={() => textFileInputRef.current?.click()} disabled={loading}>
                <TextIcon />
                {textFile ? textFile.name : 'Upload .txt / .md'}
              </button>
              {textFile && (
                <button className="btn btn-ghost" style={{ padding: '6px 8px', color: 'var(--text-dim)' }}
                  onClick={() => setTextFile(null)}>
                  <XIcon />
                </button>
              )}
            </div>

            <button className="btn btn-primary" style={{ width: '100%', justifyContent: 'center' }}
              onClick={handleTextAnalyze} disabled={loading || !textInput.trim()}>
              {loading ? 'Analyzing…' : 'Analyze Text'}
            </button>
          </div>
        )}

        {/* ── LIVE TAB ─────────────────────────────────────────────────────── */}
        {activeTab === 'live' && (
          <LiveStreamPanel onAnalysisComplete={(result) => onAnalysisComplete?.(result, [])} />
        )}

        {/* Error */}
        {error && (
          <div style={{
            marginTop: 12, padding: '10px 14px',
            background: 'var(--red-dim)', border: '1px solid rgba(248,113,113,0.25)',
            borderRadius: 'var(--radius-sm)', fontSize: 12, color: 'var(--red)',
          }}>
            {error}
          </div>
        )}
      </div>
    </div>
  );
};

export default UploadPanel;
