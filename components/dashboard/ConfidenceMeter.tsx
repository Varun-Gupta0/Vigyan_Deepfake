import React, { useEffect, memo } from 'react';
import { motion, useAnimation } from 'framer-motion';

interface ConfidenceMeterProps {
  confidence: number; // 0-100
  size?: number;
  strokeWidth?: number;
}

const ConfidenceMeter: React.FC<ConfidenceMeterProps> = ({
  confidence,
  size = 120,
  strokeWidth = 10,
}) => {
  const clamped = Math.min(100, Math.max(0, confidence));
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference * (1 - clamped / 100);

  const color =
    clamped >= 70 ? 'var(--green)' :
    clamped >= 40 ? 'var(--yellow)' :
    'var(--red)';

  const strokeCtrl = useAnimation();
  useEffect(() => {
    strokeCtrl.start({ strokeDashoffset: offset, transition: { duration: 0.7, ease: 'easeOut' } });
  }, [offset, strokeCtrl]);

  return (
    <div style={{ position: 'relative', display: 'inline-flex' }}>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}
        style={{ transform: 'rotate(-90deg)' }}>
        <circle
          cx={size / 2} cy={size / 2} r={radius}
          fill="none" stroke="var(--border)" strokeWidth={strokeWidth}
        />
        <motion.circle
          cx={size / 2} cy={size / 2} r={radius}
          fill="none" stroke={color}
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={circumference}
          animate={strokeCtrl}
        />
      </svg>
      <div style={{
        position: 'absolute', inset: 0,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
      }}>
        <span style={{ fontSize: 20, fontWeight: 700, color, fontFamily: 'var(--font-mono)' }}>
          {Math.round(clamped)}%
        </span>
      </div>
    </div>
  );
};

export default memo(ConfidenceMeter);
