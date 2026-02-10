import React, { useState, useEffect } from 'react';
import styles from './SkillToast.module.css';

const STORAGE_KEY = 'hindsight-skill-toast-dismissed';

export default function SkillToast(): JSX.Element | null {
  const [isVisible, setIsVisible] = useState(false);
  const [isAnimating, setIsAnimating] = useState(false);

  useEffect(() => {
    // Check if user has already dismissed the toast
    const dismissed = localStorage.getItem(STORAGE_KEY);

    if (!dismissed) {
      // Show toast after a short delay
      const timer = setTimeout(() => {
        setIsVisible(true);
        setIsAnimating(true);
      }, 1500);

      return () => clearTimeout(timer);
    }
  }, []);

  const handleDismiss = () => {
    setIsAnimating(false);
    setTimeout(() => {
      setIsVisible(false);
      localStorage.setItem(STORAGE_KEY, 'true');
    }, 300); // Match animation duration
  };

  if (!isVisible) return null;

  return (
    <div className={`${styles.toastContainer} ${isAnimating ? styles.show : ''}`}>
      <div className={styles.toast}>
        <div className={styles.icon}>🤖</div>
        <div className={styles.content}>
          <div className={styles.title}>Building with a coding agent?</div>
          <div className={styles.message}>
            Install the Hindsight documentation skill for faster development:
          </div>
          <code className={styles.command}>
            curl -fsSL https://hindsight.vectorize.io/get-skill | bash
          </code>
        </div>
        <button
          className={styles.closeButton}
          onClick={handleDismiss}
          aria-label="Dismiss notification"
        >
          ×
        </button>
      </div>
    </div>
  );
}
