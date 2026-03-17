import type { NextPage } from 'next';
import Head from 'next/head';
import { useState, useCallback } from 'react';
import dynamic from 'next/dynamic';
import { AnalysisResult, FrameAnalysisResult } from '@/lib/api';
import UploadToast from '@/components/dashboard/UploadToast';

const Header = dynamic(() => import('@/components/dashboard/Header'), { ssr: false });
const UploadPanel = dynamic(() => import('@/components/dashboard/UploadPanel'), { ssr: false });
const ProcessingTimeline = dynamic(() => import('@/components/dashboard/ProcessingTimeline'), { ssr: false });
const ResultsDashboard = dynamic(() => import('@/components/dashboard/ResultsDashboard'), { ssr: false });
const SystemStatus = dynamic(() => import('@/components/dashboard/SystemStatus'), { ssr: false });

type StepStatus = 'pending' | 'active' | 'complete' | 'error';

interface Step {
  id: string;
  label: string;
  status: StepStatus;
  timestamp?: string;
}

const defaultSteps: Step[] = [
  { id: '1', label: 'Extracting frames',    status: 'pending' },
  { id: '2', label: 'Detecting faces',      status: 'pending' },
  { id: '3', label: 'Artifact analysis',    status: 'pending' },
  { id: '4', label: 'Aggregating results',  status: 'pending' },
];

const liveSteps: Step[] = [
  { id: '1', label: 'Capturing frame',      status: 'pending' },
  { id: '2', label: 'Face detection',       status: 'pending' },
  { id: '3', label: 'EfficientNet inference', status: 'pending' },
  { id: '4', label: 'Scoring',              status: 'pending' },
];

const Home: NextPage = () => {
  const [result, setResult] = useState<AnalysisResult | FrameAnalysisResult | null>(null);
  const [steps, setSteps] = useState<Step[]>(defaultSteps);
  const [showToast, setShowToast] = useState(false);
  const [isLiveMode, setIsLiveMode] = useState(false);

  const handleAnalysisComplete = useCallback(
    (analysisResult: AnalysisResult | FrameAnalysisResult, processingSteps: string[]) => {
      const isLive = 'facesDetected' in (analysisResult as FrameAnalysisResult).metrics;
      setIsLiveMode(isLive);
      
      const now = new Date().toLocaleTimeString();
      const stepLabels = processingSteps.length 
        ? processingSteps 
        : (isLive ? liveSteps.map(s => s.label) : defaultSteps.map(s => s.label));
      
      setSteps(
        stepLabels.map((label, i) => ({
          id: String(i + 1),
          label,
          status: 'complete' as StepStatus,
          timestamp: now,
        }))
      );
      setResult(analysisResult);
      if (!isLive) {
        setShowToast(true);
      }
    },
    []
  );

  const handleReset = useCallback(() => {
    setResult(null);
    setSteps(defaultSteps);
    setIsLiveMode(false);
  }, []);

  return (
    <>
      <Head>
        <title>VERI-AI EDGE — Deepfake Detection</title>
      </Head>

      <div style={{ minHeight: '100vh', position: 'relative', zIndex: 1 }}>
        <Header />

        <main style={{ maxWidth: 1200, margin: '0 auto', padding: '32px 24px 64px' }}>

          {/* Hero label */}
          <div style={{ marginBottom: 32 }}>
            <h2 style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-muted)', letterSpacing: '0.06em', textTransform: 'uppercase' }}>
              Dashboard
            </h2>
            <p style={{ fontSize: 22, fontWeight: 700, color: 'var(--text)', marginTop: 4, letterSpacing: '-0.02em' }}>
              Autonomous Deepfake Detection
            </p>
          </div>

          {/* Grid */}
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))',
            gap: 20,
          }}>
            {/* Row 1 */}
            <UploadPanel onAnalysisComplete={handleAnalysisComplete} />
            <SystemStatus />

            {/* Row 2 */}
            <ProcessingTimeline events={steps} />
            <ResultsDashboard result={result} />
          </div>

          {/* Reset button – only shown after a result */}
          {result && (
            <div style={{ marginTop: 24, display: 'flex', justifyContent: 'center' }}>
              <button
                className="btn btn-ghost"
                onClick={handleReset}
                style={{ fontSize: 12 }}
              >
                ↺ &nbsp;New Analysis
              </button>
            </div>
          )}
        </main>

        {/* Footer */}
        <footer style={{
          borderTop: '1px solid var(--border)',
          padding: '16px 24px',
          textAlign: 'center',
        }}>
          <p style={{ fontSize: 11, color: 'var(--text-dim)', fontFamily: 'var(--font-mono)' }}>
            VERI-AI EDGE v1.0.0 &nbsp;·&nbsp; Edge-Optimised Multimodal Deepfake Detection
          </p>
        </footer>
      </div>

      <UploadToast
        show={showToast}
        message="Analysis complete — results updated."
        onClose={() => setShowToast(false)}
      />
    </>
  );
};

export default Home;
