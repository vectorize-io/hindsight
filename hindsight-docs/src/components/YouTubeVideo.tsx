import React, { useState } from "react";
import styles from "./YouTubeVideo.module.css";

/**
 * A YouTube video that shows our own poster and play button until it's clicked, so the page
 * carries no YouTube title bar, channel badge or "More videos" strip before playback, and loads
 * nothing from YouTube until the reader asks for the video.
 */
export default function YouTubeVideo({
  id,
  title,
  poster,
  duration,
}: {
  id: string;
  title: string;
  poster: string;
  duration?: string;
}) {
  const [playing, setPlaying] = useState(false);

  return (
    <div className={styles.frame}>
      {playing ? (
        <iframe
          className={styles.fill}
          src={`https://www.youtube-nocookie.com/embed/${id}?autoplay=1&rel=0&modestbranding=1&playsinline=1`}
          title={title}
          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
          allowFullScreen
        />
      ) : (
        <button
          type="button"
          className={styles.poster}
          onClick={() => setPlaying(true)}
          aria-label={`Play video: ${title}`}
        >
          <img className={styles.fill} src={poster} alt="" loading="lazy" />
          <span className={styles.play} aria-hidden="true">
            <svg viewBox="0 0 24 24" width="28" height="28">
              <path d="M8 5.5v13l11-6.5z" fill="currentColor" />
            </svg>
          </span>
          <span className={styles.caption}>
            <span className={styles.title}>{title}</span>
            {duration && <span className={styles.duration}>{duration}</span>}
          </span>
        </button>
      )}
    </div>
  );
}
