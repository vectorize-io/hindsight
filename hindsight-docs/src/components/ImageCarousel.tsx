import React, { useState, type ReactNode } from 'react';

interface ImageCarouselProps {
  images: { src: string; alt: string; caption?: string }[];
}

export default function ImageCarousel({ images }: ImageCarouselProps): ReactNode {
  const [current, setCurrent] = useState(0);

  const prev = () => setCurrent((c) => (c - 1 + images.length) % images.length);
  const next = () => setCurrent((c) => (c + 1) % images.length);

  const caption = images[current].caption;

  return (
    <div style={{
      width: '100%',
      borderRadius: '0.5rem',
      overflow: 'hidden',
      border: '1px solid var(--ifm-color-emphasis-200)',
      marginBottom: '1.5rem',
    }}>
      {/* Slide strip */}
      <div style={{ position: 'relative', background: 'var(--ifm-background-surface-color, #f8f9fa)' }}>
        <div style={{
          display: 'flex',
          transform: `translateX(-${current * 100}%)`,
          transition: 'transform 0.35s ease',
          willChange: 'transform',
        }}>
          {images.map((img, i) => (
            <div key={i} style={{ minWidth: '100%' }}>
              <img
                src={img.src}
                alt={img.alt}
                style={{ width: '100%', display: 'block', maxHeight: '520px', objectFit: 'contain' }}
              />
            </div>
          ))}
        </div>

        {images.length > 1 && (
          <>
            <button
              onClick={prev}
              aria-label="Previous image"
              style={{
                position: 'absolute', top: '50%', left: '0.75rem',
                transform: 'translateY(-50%)',
                background: 'rgba(0,0,0,0.45)', color: '#fff', border: 'none',
                borderRadius: '50%', width: '2rem', height: '2rem',
                cursor: 'pointer', fontSize: '1rem', lineHeight: 1,
              }}
            >‹</button>
            <button
              onClick={next}
              aria-label="Next image"
              style={{
                position: 'absolute', top: '50%', right: '0.75rem',
                transform: 'translateY(-50%)',
                background: 'rgba(0,0,0,0.45)', color: '#fff', border: 'none',
                borderRadius: '50%', width: '2rem', height: '2rem',
                cursor: 'pointer', fontSize: '1rem', lineHeight: 1,
              }}
            >›</button>
          </>
        )}
      </div>

      {/* Caption + indicators bar */}
      {(caption || images.length > 1) && (
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '0.6rem 1rem',
          borderTop: '1px solid var(--ifm-color-emphasis-200)',
          background: 'var(--ifm-background-surface-color, #f8f9fa)',
          gap: '1rem',
        }}>
          <span style={{
            fontSize: '0.85rem',
            color: 'var(--ifm-color-emphasis-700)',
            flex: 1,
          }}>
            {caption ?? ''}
          </span>

          {images.length > 1 && (
            <div style={{ display: 'flex', gap: '0.3rem', alignItems: 'center', flexShrink: 0 }}>
              {images.map((_, i) => (
                <button
                  key={i}
                  onClick={() => setCurrent(i)}
                  aria-label={`Go to image ${i + 1}`}
                  style={{
                    width: i === current ? '1.5rem' : '0.4rem',
                    height: '0.25rem',
                    borderRadius: '9999px',
                    border: 'none',
                    background: i === current
                      ? 'var(--ifm-color-primary)'
                      : 'var(--ifm-color-emphasis-300)',
                    cursor: 'pointer',
                    padding: 0,
                    transition: 'width 0.25s ease, background 0.25s ease',
                  }}
                />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
