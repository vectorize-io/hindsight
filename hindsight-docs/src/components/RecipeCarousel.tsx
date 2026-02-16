import React from 'react';
import Link from '@docusaurus/Link';
import styles from './RecipeCarousel.module.css';

export interface RecipeCard {
  title: string;
  href: string;
  tags?: {
    sdk?: string;       // Package name: "hindsight-python", "hindsight-nodejs", "litellm-python", "ai-sdk", etc.
    topic?: string;     // "Learning", "Quick Start", "Recommendation", "Chat"
  };
  description?: string;
}

interface RecipeCarouselProps {
  title: string;
  items: RecipeCard[];
}

// Language icons using inline SVG data URIs or image paths
const LANGUAGE_ICONS: Record<string, string> = {
  Python: "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath fill='%233776ab' d='M14.25.18l.9.2.73.26.59.3.45.32.34.34.25.34.16.33.1.3.04.26.02.2-.01.13V8.5l-.05.63-.13.55-.21.46-.26.38-.3.31-.33.25-.35.19-.35.14-.33.1-.3.07-.26.04-.21.02H8.77l-.69.05-.59.14-.5.22-.41.27-.33.32-.27.35-.2.36-.15.37-.1.35-.07.32-.04.27-.02.21v3.06H3.17l-.21-.03-.28-.07-.32-.12-.35-.18-.36-.26-.36-.36-.35-.46-.32-.59-.28-.73-.21-.88-.14-1.05-.05-1.23.06-1.22.16-1.04.24-.87.32-.71.36-.57.4-.44.42-.33.42-.24.4-.16.36-.1.32-.05.24-.01h.16l.06.01h8.16v-.83H6.18l-.01-2.75-.02-.37.05-.34.11-.31.17-.28.25-.26.31-.23.38-.2.44-.18.51-.15.58-.12.64-.1.71-.06.77-.04.84-.02 1.27.05zm-6.3 1.98l-.23.33-.08.41.08.41.23.34.33.22.41.09.41-.09.33-.22.23-.34.08-.41-.08-.41-.23-.33-.33-.22-.41-.09-.41.09zm13.09 3.95l.28.06.32.12.35.18.36.27.36.35.35.47.32.59.28.73.21.88.14 1.04.05 1.23-.06 1.23-.16 1.04-.24.86-.32.71-.36.57-.4.45-.42.33-.42.24-.4.16-.36.09-.32.05-.24.02-.16-.01h-8.22v.82h5.84l.01 2.76.02.36-.05.34-.11.31-.17.29-.25.25-.31.24-.38.2-.44.17-.51.15-.58.13-.64.09-.71.07-.77.04-.84.01-1.27-.04-1.07-.14-.9-.2-.73-.25-.59-.3-.45-.33-.34-.34-.25-.34-.16-.33-.1-.3-.04-.25-.02-.2.01-.13v-5.34l.05-.64.13-.54.21-.46.26-.38.3-.32.33-.24.35-.2.35-.14.33-.1.3-.06.26-.04.21-.02.13-.01h5.84l.69-.05.59-.14.5-.21.41-.28.33-.32.27-.35.2-.36.15-.36.1-.35.07-.32.04-.28.02-.21V6.07h2.09l.14.01zm-6.47 14.25l-.23.33-.08.41.08.41.23.33.33.23.41.08.41-.08.33-.23.23-.33.08-.41-.08-.41-.23-.33-.33-.23-.41-.08-.41.08z'/%3E%3C/svg%3E",
  'Node.js': "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath fill='%23339933' d='M11.998 0c-.27 0-.54.07-.772.202L2.428 5.05C1.983 5.321 1.7 5.802 1.7 6.32v11.36c0 .518.283 1 .728 1.27l2.375 1.371c.64.321 1.094.32 1.468.32 1.203 0 1.89-.73 1.89-1.996V7.362c0-.146-.117-.264-.262-.264H7.11c-.146 0-.263.118-.263.264v11.283c0 .876-.906 1.753-2.38 1.01L2.103 18.28c-.046-.026-.073-.08-.073-.132V6.754c0-.051.027-.106.073-.132l8.798-5.08c.044-.026.102-.026.145 0l8.798 5.08c.046.026.074.081.074.132v11.394c0 .051-.028.106-.074.132l-8.798 5.08c-.043.026-.101.026-.144 0l-2.248-1.336c-.064-.037-.144-.04-.21-.011-.55.307-.658.373-1.177.45-.12.019-.301.06.073.276l2.93 1.738c.23.133.49.202.772.202s.542-.069.772-.202l8.798-5.08c.476-.27.772-.772.772-1.27V6.32c0-.518-.296-.999-.772-1.27L12.77.202C12.538.07 12.268 0 11.998 0zm2.657 6.343c-2.432 0-2.945.953-2.945 2.146 0 .145.117.263.263.263h.788c.131 0 .24-.095.261-.221.177-.718.708-1.08 1.633-1.08.738 0 1.177.168 1.177.803 0 .325-.128.567-.678.73l-1.69.419c-.899.223-1.47.756-1.47 1.636 0 1.076.905 1.715 2.423 1.715 1.704 0 2.55-.593 2.656-1.866.006-.073-.018-.144-.066-.197-.047-.053-.114-.083-.186-.083h-.791c-.123 0-.23.089-.258.207-.286.644-.98.849-1.817.849-.65 0-1.16-.207-1.16-.725 0-.325.144-.424.903-.609l1.476-.367c.898-.223 1.462-.72 1.462-1.613 0-1.12-.937-1.787-2.574-1.787z'/%3E%3C/svg%3E",
  TypeScript: "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath fill='%233178c6' d='M1.125 0C.502 0 0 .502 0 1.125v21.75C0 23.498.502 24 1.125 24h21.75c.623 0 1.125-.502 1.125-1.125V1.125C24 .502 23.498 0 22.875 0zm17.363 9.75c.612 0 1.154.037 1.627.111.472.074.914.187 1.323.34v2.458c-.444-.223-.935-.39-1.473-.501-.539-.111-1.09-.167-1.655-.167-.562 0-1.011.062-1.349.187-.338.124-.507.335-.507.632 0 .234.095.42.285.558.19.138.503.275.94.411l1.503.434c.915.262 1.577.609 1.984 1.04.408.432.612.998.612 1.699 0 .915-.35 1.638-1.05 2.168-.7.53-1.667.795-2.9.795-.591 0-1.178-.051-1.76-.153-.582-.102-1.13-.258-1.645-.468v-2.503c.544.287 1.09.507 1.637.66.546.153 1.084.23 1.613.23.609 0 1.071-.073 1.386-.219.315-.146.472-.369.472-.669 0-.262-.106-.471-.318-.628-.212-.157-.551-.306-1.017-.447l-1.42-.395c-.877-.234-1.515-.563-1.916-.985-.4-.422-.6-.98-.6-1.673 0-.857.348-1.545 1.044-2.063.696-.518 1.633-.777 2.811-.777zm-13.6 1.77H8.45l-.031 4.18c0 .754-.13 1.314-.39 1.68-.26.367-.65.55-1.168.55-.286 0-.56-.037-.822-.11-.262-.074-.506-.173-.733-.297v1.818c.319.111.665.187 1.038.228.373.04.736.06 1.089.06.924 0 1.623-.247 2.097-.74.474-.494.711-1.254.711-2.28V11.52z'/%3E%3C/svg%3E",
  Go: "/img/icons/golang.png",
};

// Get language icon based on package name
function getPackageIcon(packageName: string): string | undefined {
  // If it starts with @vectorize-io, it's Node.js
  if (packageName.startsWith('@vectorize-io')) {
    return LANGUAGE_ICONS['Node.js'];
  }
  // If it ends with -go or contains go-, it's Go
  if (packageName.endsWith('-go') || packageName.includes('go-')) {
    return LANGUAGE_ICONS.Go;
  }
  // Otherwise assume Python
  return LANGUAGE_ICONS.Python;
}

// Generate color scheme from tag text using hash
function getTagColor(tag: string): any {
  // Hash function to get consistent color from string
  let hash = 0;
  for (let i = 0; i < tag.length; i++) {
    hash = tag.charCodeAt(i) + ((hash << 5) - hash);
  }

  // 12 vibrant color palettes with better contrast
  const palettes = [
    { h: 340, s: 75, l: 50 }, // Pink
    { h: 291, s: 65, l: 45 }, // Purple
    { h: 262, s: 55, l: 48 }, // Deep Purple
    { h: 231, s: 50, l: 50 }, // Indigo
    { h: 207, s: 80, l: 50 }, // Blue
    { h: 199, s: 85, l: 45 }, // Light Blue
    { h: 187, s: 70, l: 45 }, // Cyan
    { h: 174, s: 70, l: 50 }, // Teal
    { h: 142, s: 65, l: 45 }, // Green
    { h: 88, s: 55, l: 48 },  // Light Green
    { h: 38, s: 85, l: 50 },  // Orange
    { h: 14, s: 85, l: 50 },  // Deep Orange
  ];

  const palette = palettes[Math.abs(hash) % palettes.length];
  const { h, s, l } = palette;

  return {
    // Light mode: subtle background, darker text for contrast
    bg: `hsla(${h}, ${s}%, ${l}%, 0.15)`,
    text: `hsl(${h}, ${Math.min(s + 10, 90)}%, ${Math.max(l - 25, 25)}%)`,
    border: `hsla(${h}, ${s}%, ${l}%, 0.35)`,
    // Dark mode: more vibrant background, lighter text
    bgDark: `hsla(${h}, ${Math.max(s - 10, 50)}%, ${l}%, 0.25)`,
    textDark: `hsl(${h}, ${Math.max(s - 15, 40)}%, ${Math.min(l + 35, 85)}%)`,
    borderDark: `hsla(${h}, ${s}%, ${l}%, 0.4)`,
  };
}

export default function RecipeCarousel({ title, items }: RecipeCarouselProps): React.ReactElement {
  // Generate ID from title for anchor links
  const sectionId = title.toLowerCase().replace(/\s+/g, '-');

  return (
    <div className={styles.carouselSection} id={sectionId}>
      <h2 className={styles.sectionTitle}>{title}</h2>
      <div className={styles.carousel}>
        <div className={styles.carouselTrack}>
          {items.map((item, index) => {
            // Get topic color for card border
            const topicColors = item.tags?.topic ? getTagColor(item.tags.topic) : null;

            return (
              <Link
                key={index}
                to={item.href}
                className={styles.card}
                style={{
                  '--card-border': topicColors?.border,
                  '--card-border-dark': topicColors?.borderDark,
                } as React.CSSProperties}
              >
                <div className={styles.cardContent}>
                  <span className={styles.cardTitle}>{item.title}</span>
                  {item.description && (
                    <p className={styles.cardDescription}>{item.description}</p>
                  )}
                </div>
                <div className={styles.cardFooter}>
                  {item.tags && (
                    <div className={styles.cardTags}>
                      {item.tags.sdk && (() => {
                        const colors = getTagColor(item.tags.sdk);
                        const icon = getPackageIcon(item.tags.sdk);
                        return (
                          <span
                            className={styles.tag}
                            style={{
                              display: 'flex',
                              alignItems: 'center',
                              gap: '0.4rem',
                              '--tag-bg': colors.bg,
                              '--tag-text': colors.text,
                              '--tag-border': colors.border,
                              '--tag-bg-dark': colors.bgDark,
                              '--tag-text-dark': colors.textDark,
                              '--tag-border-dark': colors.borderDark,
                            } as React.CSSProperties}
                          >
                            {icon && (
                              <img
                                src={icon}
                                alt=""
                                style={{ width: '13px', height: '13px', flexShrink: 0 }}
                              />
                            )}
                            {item.tags.sdk}
                          </span>
                        );
                      })()}
                      {item.tags.topic && (() => {
                        const colors = getTagColor(item.tags.topic);
                        return (
                          <span
                            className={styles.tag}
                            style={{
                              '--tag-bg': colors.bg,
                              '--tag-text': colors.text,
                              '--tag-border': colors.border,
                              '--tag-bg-dark': colors.bgDark,
                              '--tag-text-dark': colors.textDark,
                              '--tag-border-dark': colors.borderDark,
                            } as React.CSSProperties}
                          >
                            {item.tags.topic}
                          </span>
                        );
                      })()}
                    </div>
                  )}
                  <span className={styles.cardLink}>→</span>
                </div>
              </Link>
            );
          })}
        </div>
      </div>
    </div>
  );
}
