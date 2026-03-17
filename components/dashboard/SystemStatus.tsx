import React, { useEffect, useState, useCallback, memo } from 'react';
import { checkHealth, HealthResponse } from '@/lib/api';

const SystemStatus: React.FC = () => {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [lastChecked, setLastChecked] = useState<string>('');

  const fetchHealth = useCallback(async () => {
    try {
      const data = await checkHealth();
      setHealth(data);
      setError(null);
      setLastChecked(new Date().toLocaleTimeString());
    } catch {
      setError('Backend unreachable');
      setHealth(null);
    }
  }, []);

  useEffect(() => {
    fetchHealth();
    const id = setInterval(fetchHealth, 10000);
    return () => clearInterval(id);
  }, [fetchHealth]);

  const isOk = health?.status === 'ok';

  const Row: React.FC<{ label: string; value: React.ReactNode }> = ({ label, value }) => (
    <div style={{
      display: 'flex', justifyContent: 'space-between', alignItems: 'center',
      padding: '7px 0', borderBottom: '1px solid var(--border)',
    }}>
      <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>{label}</span>
      <span style={{ fontSize: 12, color: 'var(--text)', fontFamily: 'var(--font-mono)' }}>{value}</span>
    </div>
  );

  return (
    <div className="card" style={{ padding: 20 }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
        <div className="section-title">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path strokeLinecap="round" strokeLinejoin="round" d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />
          </svg>
          System Status
        </div>
        <span className={isOk ? 'badge badge-green' : error ? 'badge badge-red' : 'badge badge-muted'}
          style={{ fontSize: 10 }}>
          {isOk ? '● Online' : error ? '● Offline' : '● Checking…'}
        </span>
      </div>

      {/* Content */}
      {error ? (
        <div style={{
          padding: '12px 14px', background: 'var(--red-dim)',
          border: '1px solid rgba(248,113,113,0.2)', borderRadius: 'var(--radius-sm)',
        }}>
          <p style={{ fontSize: 12, color: 'var(--red)' }}>{error}</p>
          <p style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>
            Make sure the Python backend is running on{' '}
            <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text)' }}>
              {process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}
            </span>
          </p>
        </div>
      ) : health ? (
        <div>
          <Row label="API status"        value={<span style={{ color: 'var(--green)' }}>Healthy</span>} />
          <Row label="Edge mode"         value={health.edgeMode} />
          <Row label="Inference device"  value={health.inferenceDevice} />
          <Row label="Model"            value={health.model || 'N/A'} />
          <Row 
            label="Live stream" 
            value={
              health.liveStreamEnabled ? (
                <span style={{ color: 'var(--green)' }}>Enabled</span>
              ) : (
                <span style={{ color: 'var(--text-muted)' }}>Disabled</span>
              )
            } 
          />
          <Row label="Latency"           value={`${health.latency} ms`} />
          {lastChecked && (
            <p style={{ fontSize: 10, color: 'var(--text-dim)', marginTop: 8 }}>
              Last checked: {lastChecked}
            </p>
          )}
        </div>
      ) : (
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <div style={{
            width: 14, height: 14,
            border: '1.5px solid var(--border)',
            borderTop: '1.5px solid var(--accent)',
            borderRadius: '50%',
          }} className="animate-spin" />
          <p style={{ fontSize: 12, color: 'var(--text-muted)' }}>Connecting to backend…</p>
        </div>
      )}
    </div>
  );
};

export default memo(SystemStatus);
