import React, {type ReactNode} from 'react';
import Link from '@docusaurus/Link';
import {SiPostgresql} from 'react-icons/si';
import {LuArrowRight} from 'react-icons/lu';
import CodingAgentsChart from './CodingAgentsChart';
import styles from './HomeFlow.module.css';

/**
 * The one-glance version of how Hindsight is used, and the homepage's router.
 *
 * It exists because three kinds of visitor arrive on this page wanting
 * different answers to the same question: someone running a coding agent wants
 * to know it plugs into the tool they already use, someone running a personal
 * assistant wants the plugin for it, someone building an app wants the SDK. The
 * interactive figure further down explains the pipeline, which is the second
 * question, not the first.
 *
 * Each card opens on hover and shows what that path actually looks like: the
 * benchmark for coding agents, a remembered exchange for assistants, three
 * calls for an app. An earlier version did this on scroll — pinning the section
 * and advancing a step at a time — which looked good and read badly, because
 * taking the scroll away from someone on a docs site is a toll rather than a
 * feature. Hover costs the reader nothing and skips nothing: it is their
 * pointer, their timing, and the page still scrolls at the speed they expect.
 *
 * It is CSS only (`:hover`, plus `:focus-within` so a keyboard gets the same
 * thing). Where there is no pointer — touch, and any narrow screen — every card
 * is simply open, because a reveal nobody can trigger is content nobody can
 * read.
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

/* The second arrival: someone who runs a personal assistant and wants it to
   remember. These are separate products with separate install paths, so each
   logo is its own link — one card-wide destination would send half of them to
   the wrong page. Two names and two arrows, nothing else: this card is a
   signpost, and the page it points at does the explaining. */
const ASSISTANTS = [
  {label: 'Hermes', src: '/img/icons/hermes.png', to: '/sdks/integrations/hermes'},
  {label: 'OpenClaw', src: '/img/icons/openclaw.svg', to: '/sdks/integrations/openclaw'},
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
          <span className={styles.sourceNote}>
            20+ harnesses, one install
            <LuArrowRight size={13} />
          </span>

          <div className={styles.reveal}>
            <div>
              <CodingAgentsChart variant="card" />
              <span className={styles.revealNote}>
                AMB’s sdebench, with and without memory: the same 60–61 of the 61 tasks
                solved either way, for fewer corrections and less money.
              </span>
            </div>
          </div>
        </Link>

        <div className={styles.source}>
          <span className={styles.sourceLabel}>Personal agents</span>
          <div className={styles.assistants}>
            {ASSISTANTS.map((a) => (
              <Link key={a.label} className={styles.assistant} to={a.to}>
                <img src={a.src} alt="" loading="lazy" />
                {a.label}
                <LuArrowRight size={12} />
              </Link>
            ))}
          </div>

          {/* What having memory sounds like, rather than a description of it.
              Invented, and obviously so — it is an illustration, not a claim. */}
          <div className={styles.reveal}>
            <div>
              <div className={styles.chat}>
                <span className={styles.ask}>Where did we land on the Lisbon trip?</span>
                <span className={styles.reply}>
                  You settled on 12–19 May, said you wanted to skip the tram queues, and
                  asked me to hold the two seafood places near Cais do Sodré.
                </span>
              </div>
            </div>
          </div>
        </div>

        <Link className={styles.source} to="/developer/api/quickstart">
          <span className={styles.sourceLabel}>Your apps</span>
          <Icons items={CLIENTS} />
          <span className={styles.sourceNote}>
            SDKs, MCP or plain HTTP
            <LuArrowRight size={13} />
          </span>

          <div className={styles.reveal}>
            <div>
              <pre className={styles.code}>
                <code>
                  {`await hs.retain(bank, transcript)
await hs.recall(bank, "what does she prefer?")
await hs.reflect(bank, "should I suggest Vue?")`}
                </code>
              </pre>
              <span className={styles.revealNote}>
                Three methods. Extraction, entity linking, four-way retrieval and reranking
                happen behind them.
              </span>
            </div>
          </div>
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
