import React, { memo } from 'react';
import { motion, Variants } from 'framer-motion';

type Status = 'pending' | 'active' | 'complete' | 'error';

interface TimelineEvent {
  id: string;
  label: string;
  status: Status;
  timestamp?: string;
}

interface Props {
  events?: TimelineEvent[];
}

const defaultEvents: TimelineEvent[] = [
  { id: '1', label: 'Extracting frames',    status: 'pending' },
  { id: '2', label: 'Detecting faces',      status: 'pending' },
  { id: '3', label: 'Artifact analysis',    status: 'pending' },
  { id: '4', label: 'Aggregating results',  status: 'pending' },
];

const statusColor: Record<Status, string> = {
  complete: 'var(--green)',
  active:   'var(--accent)',
  error:    'var(--red)',
  pending:  'var(--border-light)',
};

const StatusDot: React.FC<{ status: Status }> = ({ status }) => {
  const color = statusColor[status];

  if (status === 'active') {
    return (
      <span style={{
        width: 8, height: 8, borderRadius: '50%',
        background: color, display: 'block', flexShrink: 0,
        boxShadow: `0 0 6px ${color}`,
        animation: 'pulse 1.2s ease infinite',
      }} />
    );
  }
  if (status === 'complete') {
    return (
      <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="3">
        <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
      </svg>
    );
  }
  if (status === 'error') {
    return (
      <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="3">
        <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
      </svg>
    );
  }
  return <span style={{ width: 8, height: 8, borderRadius: '50%', background: color, display: 'block', flexShrink: 0 }} />;
};

const listV: Variants = { hidden: {}, visible: { transition: { staggerChildren: 0.15 } } };
const itemV: Variants = { hidden: { opacity: 0, x: -6 }, visible: { opacity: 1, x: 0, transition: { duration: 0.2 } } };

const ProcessingTimeline: React.FC<Props> = ({ events }) => {
  const items = events ?? defaultEvents;
  const done = items.filter(e => e.status === 'complete').length;
  const pct = Math.round((done / items.length) * 100);

  return (
    <div className="card" style={{ padding: 20 }}>
      {/* Header */}
      <div className="section-title" style={{ marginBottom: 16 }}>
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path strokeLinecap="round" strokeLinejoin="round" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
        Processing Pipeline
      </div>

      {/* Steps */}
      <motion.div variants={listV} initial="hidden" animate="visible" style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
        {items.map((event, i) => {
          const color = statusColor[event.status];
          return (
            <motion.div key={event.id} variants={itemV} style={{ display: 'flex', alignItems: 'flex-start', gap: 12, position: 'relative' }}>
              {/* Dot + connector */}
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', paddingTop: 10 }}>
                <div style={{
                  width: 20, height: 20, borderRadius: '50%',
                  border: `1.5px solid ${color}`,
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  background: event.status === 'pending' ? 'transparent' : `${color}18`,
                  flexShrink: 0,
                }}>
                  <StatusDot status={event.status} />
                </div>
                {i < items.length - 1 && (
                  <div style={{
                    width: 1, flexGrow: 1, minHeight: 20,
                    background: event.status === 'complete' ? 'var(--green)' : 'var(--border)',
                    margin: '3px 0',
                    opacity: 0.5,
                  }} />
                )}
              </div>

              {/* Label */}
              <div style={{ paddingTop: 6, paddingBottom: i < items.length - 1 ? 14 : 0, flex: 1 }}>
                <span style={{
                  fontSize: 13,
                  color: event.status === 'pending' ? 'var(--text-muted)' : 'var(--text)',
                  fontWeight: event.status === 'complete' ? 500 : 400,
                }}>
                  {event.label}
                </span>
                {event.timestamp && (
                  <span style={{ fontSize: 11, color: 'var(--text-dim)', marginLeft: 8, fontFamily: 'var(--font-mono)' }}>
                    {event.timestamp}
                  </span>
                )}
              </div>

              {/* Badge */}
              <div style={{ paddingTop: 6 }}>
                <span className={
                  event.status === 'complete' ? 'badge badge-green' :
                  event.status === 'active'   ? 'badge badge-accent' :
                  event.status === 'error'    ? 'badge badge-red' :
                  'badge badge-muted'
                } style={{ fontSize: 10 }}>
                  {event.status}
                </span>
              </div>
            </motion.div>
          );
        })}
      </motion.div>

      {/* Progress */}
      <div style={{ marginTop: 16 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: 'var(--text-muted)', marginBottom: 6 }}>
          <span>Progress</span>
          <span className="mono">{pct}%</span>
        </div>
        <div className="progress-track">
          <motion.div
            className="progress-fill"
            initial={{ width: 0 }}
            animate={{ width: `${pct}%` }}
            transition={{ duration: 0.5, ease: 'easeOut' }}
          />
        </div>
      </div>
    </div>
  );
};

export default memo(ProcessingTimeline);
