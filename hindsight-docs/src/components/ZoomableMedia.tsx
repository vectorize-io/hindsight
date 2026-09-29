import React, { Children, isValidElement, useRef, type ReactElement, type ReactNode } from "react";
import styles from "./ZoomableMedia.module.css";

interface ZoomableMediaProps {
  /** A single <img> or <video>; its `src` is also the fallback when fullscreen isn't available. */
  children: ReactElement<{ src?: string }>;
}

/**
 * Wraps one image or video with a button that shows it full screen.
 *
 * Uses the browser's own Fullscreen API, so Esc and the browser's controls close
 * it with no lightbox to maintain. iOS Safari only allows fullscreen on video, so
 * where the element has no requestFullscreen the full-size file opens in a new
 * tab instead — the next best way to read a screenshot at full resolution.
 */
export default function ZoomableMedia({ children }: ZoomableMediaProps): ReactNode {
  const frame = useRef<HTMLDivElement>(null);
  const child = Children.only(children);
  const src = isValidElement(child) ? child.props.src : undefined;

  const openFile = () => {
    if (src) window.open(src, "_blank", "noopener");
  };

  const showFullScreen = () => {
    const media = frame.current?.firstElementChild as HTMLElement | null;
    if (media?.requestFullscreen) {
      // Rejects when the page isn't allowed fullscreen (e.g. embedded in an iframe
      // without allowfullscreen); fall back to the file rather than doing nothing.
      media.requestFullscreen().catch(openFile);
    } else {
      openFile();
    }
  };

  return (
    <div ref={frame} className={styles.frame} onClick={showFullScreen}>
      {child}
      <button
        type="button"
        className={styles.zoom}
        aria-label="View full screen"
        title="View full screen"
        onClick={(event) => {
          // The frame's own click handler would otherwise fire it a second time.
          event.stopPropagation();
          showFullScreen();
        }}
      >
        <svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true">
          <path
            d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      </button>
    </div>
  );
}
