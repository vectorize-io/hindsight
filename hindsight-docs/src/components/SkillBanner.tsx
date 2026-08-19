import React, { useState, useCallback, useEffect, useRef } from 'react';
import useBaseUrl from '@docusaurus/useBaseUrl';
import styles from './SkillBanner.module.css';

const INSTALL_COMMAND =
  'npx skills add https://github.com/vectorize-io/hindsight --skill hindsight-docs';

// Brand marks only: one command installs the skill for any of them, so these
// say "this works with your agent" rather than offering a choice.
// `mono` marks a one-colour black mark that has to be flipped to stay visible
// on the dark theme; the others carry their own brand colours.
const AGENTS = [
  { name: 'Claude Code', icon: 'img/icons/claude-code.png', mono: false },
  { name: 'Codex', icon: 'img/icons/codex.svg', mono: true },
  { name: 'Cursor', icon: 'img/icons/cursor.svg', mono: false },
];

function PromptIcon(): React.ReactElement {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <polyline points="4 17 10 11 4 5" />
      <line x1="12" y1="19" x2="20" y2="19" />
    </svg>
  );
}

function CheckIcon(): React.ReactElement {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <polyline points="20 6 9 17 4 12" />
    </svg>
  );
}

function extractMarkdown(element: Element): string {
  let text = '';

  const processNode = (node: Node): string => {
    if (node.nodeType === Node.TEXT_NODE) {
      return node.textContent || '';
    }
    if (node.nodeType === Node.ELEMENT_NODE) {
      const el = node as Element;
      const tagName = el.tagName.toLowerCase();
      const children = Array.from(el.childNodes).map(processNode).join('');
      switch (tagName) {
        case 'h1': return `# ${children}\n\n`;
        case 'h2': return `## ${children}\n\n`;
        case 'h3': return `### ${children}\n\n`;
        case 'h4': return `#### ${children}\n\n`;
        case 'h5': return `##### ${children}\n\n`;
        case 'h6': return `###### ${children}\n\n`;
        case 'p': return `${children}\n\n`;
        case 'ul': return `${children}\n`;
        case 'ol': return `${children}\n`;
        case 'li': {
          const parent = el.parentElement;
          const isOrdered = parent?.tagName.toLowerCase() === 'ol';
          if (isOrdered) {
            const index = Array.from(parent?.children || []).indexOf(el) + 1;
            return `${index}. ${children}\n`;
          }
          return `- ${children}\n`;
        }
        case 'code': {
          const isBlock = el.parentElement?.tagName.toLowerCase() === 'pre';
          if (isBlock) {
            const lang = el.className.replace('language-', '');
            return `\`\`\`${lang}\n${children}\n\`\`\`\n\n`;
          }
          return `\`${children}\``;
        }
        case 'pre': return children;
        case 'blockquote': return children.split('\n').map(line => `> ${line}`).join('\n') + '\n\n';
        case 'a': return `[${children}](${el.getAttribute('href') || ''})`;
        case 'strong': case 'b': return `**${children}**`;
        case 'em': case 'i': return `*${children}*`;
        case 'br': return '\n';
        case 'hr': return '---\n\n';
        case 'table': return `${children}\n`;
        case 'thead': case 'tbody': return children;
        case 'tr': return `${children}|\n`;
        case 'th': case 'td': return `| ${children} `;
        case 'img': return `![${el.getAttribute('alt') || ''}](${el.getAttribute('src') || ''})`;
        default: return children;
      }
    }
    return '';
  };

  Array.from(element.childNodes).forEach(node => { text += processNode(node); });
  return text;
}

/** Serialises the rendered page back to markdown, title first. */
function collectPageMarkdown(): { title: string; markdown: string } | null {
  const contentElement = document.querySelector('.markdown');
  if (!contentElement) return null;
  const title = document.querySelector('h1')?.textContent ?? '';
  const body = Array.from(contentElement.children)
    .filter(child => !(child.tagName === 'H1' && child.textContent === title))
    .map(child => extractMarkdown(child))
    .join('');
  const markdown = `${title ? `# ${title}\n\n` : ''}${body}`
    .replace(/\n{3,}/g, '\n\n')
    .trim();
  return { title, markdown };
}

export default function SkillBanner(): React.ReactElement {
  const [commandCopied, setCommandCopied] = useState(false);
  const [pageCopied, setPageCopied] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const actionsRef = useRef<HTMLDivElement>(null);

  // Dismiss the menu the way any menu is expected to go away.
  useEffect(() => {
    if (!menuOpen) return undefined;
    const onPointerDown = (event: MouseEvent) => {
      if (!actionsRef.current?.contains(event.target as Node)) {
        setMenuOpen(false);
      }
    };
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setMenuOpen(false);
    };
    document.addEventListener('mousedown', onPointerDown);
    document.addEventListener('keydown', onKeyDown);
    return () => {
      document.removeEventListener('mousedown', onPointerDown);
      document.removeEventListener('keydown', onKeyDown);
    };
  }, [menuOpen]);

  const handleCopyCommand = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(INSTALL_COMMAND);
      setCommandCopied(true);
      setTimeout(() => setCommandCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  }, []);

  const handleCopyPage = useCallback(async () => {
    try {
      const page = collectPageMarkdown();
      if (!page) return;
      await navigator.clipboard.writeText(page.markdown);
      setPageCopied(true);
      setMenuOpen(false);
      setTimeout(() => setPageCopied(false), 2000);
    } catch (error) {
      console.error('Failed to copy page content:', error);
    }
  }, []);

  const handleDownloadPage = useCallback(() => {
    const page = collectPageMarkdown();
    if (!page) return;
    const url = URL.createObjectURL(
      new Blob([page.markdown], { type: 'text/markdown;charset=utf-8' }),
    );
    const link = document.createElement('a');
    link.href = url;
    link.download = `${
      page.title.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '') || 'page'
    }.md`;
    link.click();
    // Revoked on the next tick, not inline: the download is only queued by the
    // click, and pulling the URL out from under it in the same frame can cancel
    // it before it starts.
    setTimeout(() => URL.revokeObjectURL(url), 0);
    setMenuOpen(false);
  }, []);

  const agentIconBase = useBaseUrl('/');

  return (
    <div className={styles.container}>
      <div className={styles.banner}>
        <span className={styles.prompt} aria-hidden="true">
          <PromptIcon />
        </span>
        <span className={styles.title}>Building with a coding agent?</span>

        <span className={styles.agents}>
          {AGENTS.map(agent => (
            <img
              key={agent.name}
              className={`${styles.agentLogo}${agent.mono ? ` ${styles.agentLogoMono}` : ''}`}
              src={agentIconBase + agent.icon}
              alt={agent.name}
              title={`Works with ${agent.name}`}
              loading="lazy"
              width={18}
              height={18}
            />
          ))}
        </span>

        <div className={styles.actions} ref={actionsRef}>
          <button
            type="button"
            className={`${styles.installButton} ${commandCopied ? styles.copied : ''}`}
            onClick={handleCopyCommand}
            title={commandCopied ? 'Copied!' : `Copy: ${INSTALL_COMMAND}`}
          >
            {commandCopied ? <CheckIcon /> : <PromptIcon />}
            <span>{commandCopied ? 'Copied!' : 'Install skill'}</span>
          </button>
          <button
            type="button"
            className={styles.menuButton}
            onClick={() => setMenuOpen(open => !open)}
            aria-label="More page actions"
            aria-haspopup="menu"
            aria-expanded={menuOpen}
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <polyline points="6 9 12 15 18 9" />
            </svg>
          </button>

          {menuOpen && (
            <div className={styles.menu} role="menu">
              <button type="button" className={styles.menuItem} role="menuitem" onClick={handleCopyPage}>
                <PromptIcon />
                <span>{pageCopied ? 'Copied!' : 'Copy page'}</span>
              </button>
              <button type="button" className={styles.menuItem} role="menuitem" onClick={handleDownloadPage}>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                  <polyline points="7 10 12 15 17 10" />
                  <line x1="12" y1="15" x2="12" y2="3" />
                </svg>
                <span>Download page (.md)</span>
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
