import React, { useEffect, useState } from 'react';

interface Props {
  show: boolean;
  message?: string;
  onClose: () => void;
}

const UploadToast: React.FC<Props> = ({ show, message = 'Analysis complete!', onClose }) => {
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    if (show) {
      setVisible(true);
      const t = setTimeout(() => { setVisible(false); onClose(); }, 4000);
      return () => clearTimeout(t);
    }
  }, [show, onClose]);

  if (!visible) return null;

  return (
    <div className="toast">
      <span style={{ color: 'var(--green)', fontSize: 18, lineHeight: 1 }}>✓</span>
      <span style={{ color: 'var(--text)', fontSize: 13 }}>{message}</span>
      <button
        onClick={() => { setVisible(false); onClose(); }}
        style={{
          marginLeft: 8, background: 'none', border: 'none', cursor: 'pointer',
          color: 'var(--text-muted)', fontSize: 16, lineHeight: 1,
          padding: '0 2px',
        }}
      >
        ×
      </button>
    </div>
  );
};

export default UploadToast;
