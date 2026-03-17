import React from 'react';

const ShieldIcon = () => (
  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
    <path d="M9 12l2 2 4-4" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
);

const Header: React.FC = () => (
  <header style={{
    background: 'var(--surface)',
    borderBottom: '1px solid var(--border)',
    position: 'sticky',
    top: 0,
    zIndex: 50,
  }}>
    <div style={{
      maxWidth: 1200,
      margin: '0 auto',
      padding: '0 24px',
      height: 60,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
    }}>
      {/* Logo */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
        <div style={{ color: 'var(--accent)' }}>
          <ShieldIcon />
        </div>
        <div>
          <div style={{ fontSize: 15, fontWeight: 700, color: 'var(--text)', letterSpacing: '-0.02em' }}>
            VERI-AI <span style={{ color: 'var(--accent)' }}>EDGE</span>
          </div>
          <div style={{ fontSize: 10, color: 'var(--text-muted)', letterSpacing: '0.05em' }}>
            MULTIMODAL DEEPFAKE DETECTION
          </div>
        </div>
      </div>

      {/* Status badges */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <span className="badge badge-accent">
          <span className="pulse-dot" style={{ width: 5, height: 5, background: 'var(--accent)', boxShadow: '0 0 4px var(--accent)' }} />
          Edge AI Active
        </span>
        <span className="badge badge-muted" style={{ fontSize: 10 }}>
          CPU Inference
        </span>
      </div>
    </div>
  </header>
);

export default Header;
