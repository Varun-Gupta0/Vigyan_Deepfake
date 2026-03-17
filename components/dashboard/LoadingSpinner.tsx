import React, { memo } from 'react';
import { motion } from 'framer-motion';

/**
 * A tiny, CSS‑only spinner using Framer Motion for a smooth infinite rotation.
 * Used as a fallback loading indicator for lazy‑loaded components.
 */
const LoadingSpinner: React.FC = () => (
  <motion.div
    className="flex items-center justify-center h-12"
    animate={{ rotate: 360 }}
    transition={{ repeat: Infinity, duration: 1.2, ease: 'linear' }}
  >
    <svg className="w-6 h-6 text-cyber-accent" viewBox="0 0 24 24" fill="none">
      <circle
        className="opacity-25"
        cx="12"
        cy="12"
        r="10"
        stroke="currentColor"
        strokeWidth="4"
      />
      <path
        className="opacity-75"
        fill="currentColor"
        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
      />
    </svg>
  </motion.div>
);

export default memo(LoadingSpinner);
