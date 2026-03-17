import React, { memo } from 'react';
import { motion, useAnimation } from 'framer-motion';
import { useEffect } from 'react';
import { AnalysisResult, FrameAnalysisResult } from '@/lib/api';

interface Props {
  result: AnalysisResult | FrameAnalysisResult | null;
}

const verdictConfig: Record<string, any> = {
  authentic: { color: 'var(--green)', dim: 'var(--green-dim)', label: 'Authentic', emoji: '✓' },
  suspicious: { color: 'var(--yellow)', dim: 'var(--yellow-dim)', label: 'Suspicious', emoji: '⚠' },
  deepfake: { color: 'var(--red)', dim: 'var(--red-dim)', label: 'Likely Deepfake', emoji: '✕' },
  REAL: { color: 'var(--green)', dim: 'var(--green-dim)', label: 'REAL', emoji: '✓' },
  UNCERTAIN: { color: 'var(--yellow)', dim: 'var(--yellow-dim)', label: 'UNCERTAIN', emoji: '⚠' },
  FAKE: { color: 'var(--red)', dim: 'var(--red-dim)', label: 'FAKE', emoji: '✕' },
};

/* ── Confidence gauge ─────────────────────────────────────────────────────── */
const Gauge: React.FC<{ value: number; color: string }> = ({ value, color }) => {
  const size = 120;
  const sw = 10;
  const r = (size - sw) / 2;
  const circ = 2 * Math.PI * r;
  const offset = circ * (1 - (Math.min(100, Math.max(0, value)) / 100));
  const ctrl = useAnimation();

  useEffect(() => {
    ctrl.start({ strokeDashoffset: offset, transition: { duration: 0.8, ease: 'easeOut' } });
  }, [offset, ctrl]);

  return (
    <div style={{ position: 'relative', display: 'inline-flex' }}>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} style={{ transform: 'rotate(-90deg)' }}>
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="var(--border)" strokeWidth={sw} />
        <motion.circle
          cx={size / 2} cy={size / 2} r={r}
          fill="none" stroke={color} strokeWidth={sw}
          strokeLinecap="round" strokeDasharray={circ} strokeDashoffset={circ}
          animate={ctrl}
        />
      </svg>
      <div style={{
        position: 'absolute', inset: 0,
        display: 'flex', flexDirection: 'column',
        alignItems: 'center', justifyContent: 'center',
      }}>
        <span style={{ fontSize: 22, fontWeight: 700, color, fontFamily: 'var(--font-mono)' }}>
          {Math.round(Math.min(100, Math.max(0, value)))}%
        </span>
      </div>
    </div>
  );
};

/* ── Metric row ───────────────────────────────────────────────────────────── */
const Metric: React.FC<{ label: string; value: React.ReactNode }> = ({ label, value }) => (
  <div style={{
    display: 'flex', justifyContent: 'space-between', alignItems: 'center',
    padding: '8px 0', borderBottom: '1px solid var(--border)',
  }}>
    <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>{label}</span>
    <span style={{ fontSize: 12, fontFamily: 'var(--font-mono)', color: 'var(--text)' }}>{value}</span>
  </div>
);

/* ── Main component ───────────────────────────────────────────────────────── */
const ResultsDashboard: React.FC<Props> = ({ result }) => {
  if (!result) {
    return (
      <div className="card" style={{ padding: 24, textAlign: 'center' }}>
        <div style={{ fontSize: 32, marginBottom: 12, opacity: 0.3 }}>◯</div>
        <p style={{ fontSize: 13, color: 'var(--text-muted)' }}>
          Upload a file to see analysis results
        </p>
      </div>
    );
  }

  const apiLabel = 'label' in result.decision ? result.decision.label : result.verdict;
  const vc = verdictConfig[apiLabel] ?? verdictConfig.suspicious;
  const m = result.metrics || { modelUsed: 'N/A', inferenceDevice: 'N/A' };

  return (
    <motion.div
      className="card"
      style={{ padding: 20 }}
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35 }}
    >
      {/* Section title */}
      <div className="section-title" style={{ marginBottom: 20 }}>
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
        </svg>
        Analysis Results
      </div>

      {/* Gauge + Verdict */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 24, marginBottom: 20 }}>
        <div>
          <Gauge value={result.confidence} color={vc.color} />
          <p style={{ fontSize: 11, color: 'var(--text-muted)', textAlign: 'center', marginTop: 6 }}>
            Confidence
          </p>
        </div>

        <div style={{ flex: 1 }}>
          <div style={{
            display: 'inline-flex', alignItems: 'center', gap: 8,
            padding: '10px 18px', borderRadius: 'var(--radius-sm)',
            background: vc.dim, border: `1px solid ${vc.color}30`,
            marginBottom: 12,
          }}>
            <span style={{ fontSize: 18, color: vc.color }}>{vc.emoji}</span>
            <span style={{ fontSize: 16, fontWeight: 700, color: vc.color }}>{vc.label}</span>
          </div>

          {'reason' in result && (
            <p style={{ fontSize: 12, color: 'var(--text-muted)', lineHeight: 1.6 }}>
              {result.reason}
            </p>
          )}
        </div>
      </div>

      {/* Metrics */}
      <div style={{ borderTop: '1px solid var(--border)', paddingTop: 16 }}>
        <p style={{ fontSize: 11, color: 'var(--text-dim)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
          Metrics
        </p>
        {m.framesAnalyzed !== undefined && (
          <Metric label="Frames analyzed" value={m.framesAnalyzed} />
        )}
        {m.facesDetected !== undefined && (
          <Metric label="Faces detected" value={m.facesDetected} />
        )}
        {m.processingTime !== undefined && (
          <Metric label="Processing time" value={`${m.processingTime}s`} />
        )}
        <Metric label="Model" value={m.modelUsed} />
        <Metric label="Device" value={m.inferenceDevice} />
        {'phishing_score' in result && result.phishing_score !== undefined && (
          <Metric label="Phishing score" value={`${result.phishing_score}%`} />
        )}
      </div>

      {/* Detected Faces */}
      {'faces' in result && result.faces && result.faces.length > 0 && (
        <div style={{ borderTop: '1px solid var(--border)', paddingTop: 16, marginTop: 16 }}>
          <p style={{ fontSize: 11, color: 'var(--text-dim)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
            Detected Faces
          </p>
          {result.faces.map((face, idx) => {
            const score = face.score;
            const indicatorColor = score > 0.5 ? 'var(--red)' : score > 0.3 ? 'var(--yellow)' : 'var(--green)';
            return (
              <div key={idx} style={{
                display: 'flex', alignItems: 'center', gap: 8,
                padding: '8px 10px', marginBottom: 6,
                background: 'var(--surface)', borderRadius: 'var(--radius-sm)',
                border: '1px solid var(--border)',
              }}>
                <span style={{
                  width: 8, height: 8, borderRadius: '50%',
                  background: indicatorColor, flexShrink: 0,
                }} />
                <span style={{ fontSize: 11, color: 'var(--text-muted)', width: 50 }}>
                  Face {idx + 1}
                </span>
                <span style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--text)', flex: 1 }}>
                  bbox: [{face.bbox.map((b: number) => Math.round(b)).join(', ')}]
                </span>
                <span style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: indicatorColor }}>
                  {score > 0.5 ? 'Fake' : 'Real'}: {((score > 0.5 ? score : 1 - score) * 100).toFixed(0)}%
                </span>
              </div>
            );
          })}
        </div>
      )}
    </motion.div>
  );
};

export default memo(ResultsDashboard);
