import React, {type ReactNode} from 'react';
import Link from '@docusaurus/Link';
import {SiPostgresql} from 'react-icons/si';
import {LuArrowRight} from 'react-icons/lu';
import styles from './HomeFlow.module.css';

/**
 * The one-glance version of how Hindsight is used.
 *
 * It exists because the two audiences arriving on this page want different
 * answers to the same question. Someone running a coding agent wants to know it
 * plugs into the tool they already use; someone building an app wants to know
 * there is an SDK. The interactive figure further down explains the pipeline,
 * which is the second question, not the first.
 *
 * The memory bank is drawn INSIDE the Hindsight boundary, not as a third stage
 * beside it: an earlier version put it in its own box on the right, which read
 * as a separate product you had to bring yourself. What actually sits outside
 * is the database, so Postgres is the thing on the right-hand end.
 *
 * Logos come from static/img/icons — the same set the integrations hub uses.
 */
const AGENTS = [
  {label: 'Claude Code', src: '/img/icons/claude-code.png'},
  {label: 'Cursor', src: '/img/icons/cursor.svg'},
  {label: 'Codex', src: '/img/icons/codex.svg'},
  {label: 'Copilot', src: '/img/icons/github-copilot.svg'},
  {label: 'Gemini CLI', src: '/img/icons/gemini.svg'},
  {label: 'opencode', src: '/img/icons/opencode.png'},
];

const CLIENTS = [
  {label: 'Python', src: '/img/icons/python.svg'},
  {label: 'TypeScript', src: '/img/icons/typescript.png'},
  {label: 'Go', src: '/img/icons/golang.png'},
  {label: 'MCP', src: '/img/icons/mcp.png'},
];

function Icons({items}: {items: {label: string; src: string}[]}): ReactNode {
  return (
    <div className={styles.icons}>
      {items.map((i) => (
        <img key={i.label} src={i.src} alt={i.label} title={i.label} loading="lazy" />
      ))}
    </div>
  );
}

export default function HomeFlow(): ReactNode {
  return (
    <div className={styles.flow}>
      <div className={styles.sources}>
        <Link className={styles.source} to="/sdks/integrations/coding-agents">
          <span className={styles.sourceLabel}>Coding agents</span>
          <Icons items={AGENTS} />
          <span className={styles.sourceNote}>18 harnesses, one install</span>
        </Link>

        <Link className={styles.source} to="/developer/api/quickstart">
          <span className={styles.sourceLabel}>Your apps</span>
          <Icons items={CLIENTS} />
          <span className={styles.sourceNote}>SDKs, MCP or plain HTTP</span>
        </Link>
      </div>

      <div className={styles.connector} aria-hidden="true" />

      <div className={styles.core}>
        <span className={styles.coreName}>Hindsight</span>

        <div className={styles.ops}>
          <span><code>retain</code> store what happened</span>
          <span><code>recall</code> search it back</span>
          <span><code>reflect</code> reason over it</span>
        </div>

        {/* Inside the boundary: the bank is what Hindsight builds, not a
            component you supply. */}
        <div className={styles.bank}>
          <span className={styles.bankName}>Memory bank</span>
          <div className={styles.bankItems}>
            <span>Facts &amp; entities</span>
            <span>Observations</span>
            <span>Knowledge pages</span>
          </div>
          <span className={styles.bankNote}>One per user or agent, fully isolated</span>
        </div>
      </div>

      <div className={styles.connector} aria-hidden="true" />

      <div className={styles.store}>
        <SiPostgresql className={styles.storeIcon} size={30} />
        <span className={styles.storeName}>Postgres</span>
        <span className={styles.storeNote}>with pgvector</span>

        {/* Managed first: it is the one option that takes no decision. */}
        <ul className={styles.options}>
          <li className={styles.optionPrimary}>
            <Link to="https://ui.hindsight.vectorize.io/signup">
              Hindsight Cloud
              <LuArrowRight size={12} />
            </Link>
            <span>fully managed</span>
          </li>
          <li>
            Embedded
            <span>zero setup, runs locally</span>
          </li>
          <li>
            Your own cluster
            <span>bring a connection string</span>
          </li>
        </ul>
      </div>
    </div>
  );
}
