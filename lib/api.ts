/**
 * API service layer for the VERI-AI EDGE dashboard.
 *
 * Endpoints:
 *   • uploadVideo()   – POST /analyze/video
 *   • analyzeText()  – POST /analyze/text
 *   • checkHealth()  – GET  /health
 */

import axios, {
  AxiosInstance,
  AxiosError,
  CancelTokenSource,
} from 'axios';

/* -------------------------------------------------------------------------- */
/* Axios instance                                                              */
/* -------------------------------------------------------------------------- */
const api: AxiosInstance = axios.create({
  baseURL: process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000',
  timeout: 60000,
  headers: {
    Accept: 'application/json',
    'ngrok-skip-browser-warning': 'true',
  },
});

/* -------------------------------------------------------------------------- */
/* Shared types                                                                */
/* -------------------------------------------------------------------------- */
export interface ApiError extends Error {
  status?: number;
  originalError: AxiosError;
}

export interface AnalysisResult {
  /** Confidence score as a percentage (0-100). */
  confidence: number;
  /** Verdict:  "authentic" | "suspicious" | "deepfake" */
  verdict: 'authentic' | 'suspicious' | 'deepfake';
  /** Decision from the backend engine */
  decision: { label: string; confidence: number };
  /** Human-readable explanation */
  reason: string;
  /** Processing step labels */
  processing_steps: string[];
  /** Structured metrics for the results panel */
  metrics: {
    framesAnalyzed?: number;
    processingTime?: number;
    facesDetected?: number;
    modelUsed: string;
    inferenceDevice: string;
  };
  /** Analysis time in seconds */
  analysis_time_seconds?: number;
  /** Phishing score (text only, 0-100) */
  phishing_score?: number;
  /** Detected faces with bounding boxes */
  faces?: Array<{ bbox: number[]; score: number }>;
}

export interface FrameAnalysisResult {
  confidence: number;
  verdict: 'authentic' | 'suspicious' | 'deepfake';
  decision: { label: string; confidence: number };
  faces: Array<{ bbox: number[]; score: number }>;
  reason?: string;
  metrics: {
    framesAnalyzed?: number;
    processingTime?: number;
    facesDetected?: number;
    modelUsed: string;
    inferenceDevice: string;
  };
}

export interface HealthResponse {
  status: string;        // "ok" | "error"
  edgeMode: string;      // "enabled" | "disabled"
  inferenceDevice: string;
  latency: number;       // ms
  model?: string;
  liveStreamEnabled?: boolean;
}

/* -------------------------------------------------------------------------- */
/* Helper – typed ApiError                                                    */
/* -------------------------------------------------------------------------- */
function createApiError(message: string, err: AxiosError): ApiError {
  const apiError = new Error(message) as ApiError;
  apiError.status = err.response?.status;
  apiError.originalError = err;
  return apiError;
}

/* -------------------------------------------------------------------------- */
/* uploadVideo – POST /analyze/video                                          */
/* -------------------------------------------------------------------------- */
export async function uploadVideo(
  file: File,
  cancelSource?: CancelTokenSource,
  onProgress?: (pct: number) => void
): Promise<AnalysisResult> {
  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await api.post<AnalysisResult>('/analyze/video', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      cancelToken: cancelSource?.token,
      onUploadProgress: (e) => {
        if (e.total && onProgress) {
          onProgress(Math.round((e.loaded * 100) / e.total));
        }
      },
    });
    return response.data;
  } catch (err) {
    const axiosErr = err as AxiosError;
    if (axios.isCancel(axiosErr)) throw createApiError('Video upload cancelled', axiosErr);
    if (axiosErr.code === 'ECONNABORTED') throw createApiError('Video upload timed out', axiosErr);
    throw createApiError('Failed to upload video', axiosErr);
  }
}

/* -------------------------------------------------------------------------- */
/* analyzeText – POST /analyze/text (JSON body)                              */
/* -------------------------------------------------------------------------- */
export async function analyzeText(
  text: string,
  cancelSource?: CancelTokenSource
): Promise<AnalysisResult> {
  try {
    const response = await api.post<AnalysisResult>(
      '/analyze/text',
      { text },
      {
        headers: { 'Content-Type': 'application/json' },
        cancelToken: cancelSource?.token,
      }
    );
    return response.data;
  } catch (err) {
    const axiosErr = err as AxiosError;
    if (axios.isCancel(axiosErr)) throw createApiError('Text analysis cancelled', axiosErr);
    if (axiosErr.code === 'ECONNABORTED') throw createApiError('Text analysis timed out', axiosErr);
    throw createApiError('Failed to analyze text', axiosErr);
  }
}

/* -------------------------------------------------------------------------- */
/* analyzeTextFile – POST /analyze/text (multipart file upload)              */
/* -------------------------------------------------------------------------- */
export async function analyzeTextFile(
  file: File,
  cancelSource?: CancelTokenSource
): Promise<AnalysisResult> {
  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await api.post<AnalysisResult>('/analyze/text', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      cancelToken: cancelSource?.token,
    });
    return response.data;
  } catch (err) {
    const axiosErr = err as AxiosError;
    if (axios.isCancel(axiosErr)) throw createApiError('Text analysis cancelled', axiosErr);
    throw createApiError('Failed to analyze text file', axiosErr);
  }
}

/* -------------------------------------------------------------------------- */
/* checkHealth – GET /health                                                  */
/* -------------------------------------------------------------------------- */
export async function checkHealth(
  cancelSource?: CancelTokenSource
): Promise<HealthResponse> {
  try {
    const response = await api.get<HealthResponse>('/health', {
      cancelToken: cancelSource?.token,
    });
    return response.data;
  } catch (err) {
    const axiosErr = err as AxiosError;
    if (axios.isCancel(axiosErr)) throw createApiError('Health check cancelled', axiosErr);
    if (axiosErr.code === 'ECONNABORTED') throw createApiError('Health check timed out', axiosErr);
    throw createApiError('Failed to fetch health status', axiosErr);
  }
}

/* -------------------------------------------------------------------------- */
/* analyzeFrame – POST /analyze/frame (live webcam)                           */
/* -------------------------------------------------------------------------- */
export async function analyzeFrame(
  imageBase64: string
): Promise<FrameAnalysisResult> {
  try {
    const response = await api.post<FrameAnalysisResult>(
      '/analyze/frame',
      { image: imageBase64 },
      {
        headers: { 'Content-Type': 'application/json' },
      }
    );
    return response.data;
  } catch (err) {
    const axiosErr = err as AxiosError;
    if (axios.isCancel(axiosErr)) throw createApiError('Frame analysis cancelled', axiosErr);
    throw createApiError('Failed to analyze frame', axiosErr);
  }
}

export default api;
