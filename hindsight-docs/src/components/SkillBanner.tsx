import React, { useState } from 'react';
import styles from './SkillBanner.module.css';

export default function SkillBanner(): JSX.Element {
  const [copied, setCopied] = useState(false);
  const command = 'npx skills add https://github.com/vectorize-io/hindsight --skill hindsight-docs';

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(command);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  return (
    <div className={styles.container}>
      <div className={styles.banner}>
        <div className={styles.icon}>🤖</div>
        <div className={styles.content}>
          <div className={styles.title}>
            Using a coding agent? Install the docs skill for instant access
          </div>
          <div className={styles.commandWrapper}>
            <code className={styles.command}>
              {command}
            </code>
            <button
              className={styles.copyButton}
              onClick={handleCopy}
              aria-label="Copy command"
              title={copied ? 'Copied!' : 'Copy to clipboard'}
            >
              {copied ? (
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <polyline points="20 6 9 17 4 12"></polyline>
                </svg>
              ) : (
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                  <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                </svg>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
