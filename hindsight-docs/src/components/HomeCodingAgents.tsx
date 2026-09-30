import React, {type ReactNode} from 'react';
import Link from '@docusaurus/Link';
import {LuArrowUpRight} from 'react-icons/lu';
import styles from './HomeCodingAgents.module.css';

/**
 * What memory does to a coding agent, from AMB's sdebench dataset.
 *
 * It is a section on the page and not part of the hero: sharing the band with
 * the accuracy chart pushed that band past 900px and squeezed this one to half
 * width, where its three labelled marks landed on top of each other. It needs
 * the full column.
 *
 * The accuracy chart in the hero answers "does it retrieve well". This answers the
 * question a developer evaluating a coding-agent plugin actually has: does it
 * make the agent cheaper and less annoying. Both move at once, so the form has
 * to be two-dimensional — one arrow per agent, from no memory to Hindsight, and
 * every arrow points up and to the left.
 *
 * The agent's own logo marks the Hindsight end and a hollow grey dot marks the
 * other, so "where Hindsight is" needs no legend lookup: it is wherever the logo
 * is. Every point also carries its corrections value, the way AMB labels its
 * own version of this chart.
 *
 * Numbers are derived from the per-run table at
 * agentmemorybenchmark.ai/dataset/sdebench, 61 tasks × 3 runs per cell:
 * corrections/task = mean(interventions)/61, cost/task = mean(cost)/61. The
 * corrections figures reproduce the labels AMB prints on its own chart to two
 * decimals, which is what makes the derived cost column trustworthy too.
 *
 * sdebench is the one dataset where accuracy is NOT the story: every run solves
 * 60–61 of 61 either way. Memory changes what it costs to get there.
 */
const AGENTS: {
  name: string;
  icon: string;
  from: {cost: number; fixes: number};
  to: {cost: number; fixes: number};
}[] = [
  {
    name: 'Claude Code',
    icon: '/img/icons/claude-code.png',
    from: {cost: 0.445, fixes: 0.85},
    to: {cost: 0.338, fixes: 0.36},
  },
  {
    name: 'Codex CLI',
    icon: '/img/icons/codex.svg',
    from: {cost: 0.577, fixes: 1.34},
    to: {cost: 0.276, fixes: 0.47},
  },
  {
    name: 'opencode',
    icon: '/img/icons/opencode.png',
    from: {cost: 0.634, fixes: 1.20},
    to: {cost: 0.553, fixes: 0.80},
  },
];

// Wide and short. Squeezed into half the proof row this chart's three labelled
// marks landed on top of each other; given the full width it has room for the
// names, the values and both axis captions without a single hand-placed offset.
const W = 940;
const H = 270;
const PLOT = {left: 60, right: 860, top: 34, bottom: 196};

// Cost on a log scale, as AMB plots it: the spread is multiplicative.
const COST = {min: 0.24, max: 0.72};
const FIXES_MAX = 1.5;

const lx = (c: number) =>
  PLOT.left +
  ((Math.log(c) - Math.log(COST.min)) / (Math.log(COST.max) - Math.log(COST.min))) *
    (PLOT.right - PLOT.left);

// Inverted: fewer corrections is better, so fewer is higher up.
const ly = (f: number) => PLOT.top + (f / FIXES_MAX) * (PLOT.bottom - PLOT.top);

const CHIP = 26;

export default function HomeCodingAgents(): ReactNode {
  return (
    <section className={styles.section}>
      <div className={styles.head}>
        <h3 className={styles.title}>Coding agents</h3>
        <Link className={styles.more} to="https://agentmemorybenchmark.ai/dataset/sdebench">
          Full results
          <LuArrowUpRight size={14} />
        </Link>
      </div>

      <figure className={styles.figure}>
    <svg
      className={styles.chart}
      viewBox={`0 0 ${W} ${H}`}
      role="img"
      aria-label="Corrections per task against cost per task on AMB's sdebench. With Hindsight, Claude Code goes from 0.85 corrections at $0.45 to 0.36 at $0.34; Codex CLI from 1.34 at $0.58 to 0.47 at $0.28; opencode from 1.20 at $0.63 to 0.80 at $0.55. Every agent moves to fewer corrections and lower cost.">
      <defs>
        <marker
          id="hca-arrow"
          viewBox="0 0 10 10"
          refX="9"
          refY="5"
          markerWidth="4.5"
          markerHeight="4.5"
          orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" className={styles.arrowHead} />
        </marker>
      </defs>

      {[0, 0.5, 1.0, 1.5].map((f) => (
        <g key={f}>
          <line className={styles.grid} x1={PLOT.left} x2={PLOT.right} y1={ly(f)} y2={ly(f)} />
          <text className={styles.axis} x={PLOT.left - 10} y={ly(f) + 4} textAnchor="end">
            {f.toFixed(1)}
          </text>
        </g>
      ))}
      {[0.25, 0.35, 0.5, 0.7].map((c) => (
        <text key={c} className={styles.axis} x={lx(c)} y={PLOT.bottom + 22} textAnchor="middle">
          ${c.toFixed(2)}
        </text>
      ))}

      <text className={styles.axisTitle} x={PLOT.left - 48} y={PLOT.top - 16}>
        Corrections / task — fewer is better ↑
      </text>
      <text
        className={styles.axisTitle}
        x={(PLOT.left + PLOT.right) / 2}
        y={PLOT.bottom + 48}
        textAnchor="middle">
        Cost / task (USD, log) — cheaper is better ←
      </text>

      {AGENTS.map((a) => (
        <g key={a.name}>
          {/* Where it starts. Hollow and grey: "no memory" is a state every
              agent shares, not a series of its own. */}
          <circle className={styles.before} cx={lx(a.from.cost)} cy={ly(a.from.fixes)} r={4.5} />
          <text
            className={styles.valueBefore}
            x={lx(a.from.cost) + 10}
            y={ly(a.from.fixes) + 4}>
            {a.from.fixes.toFixed(2)}
          </text>

          <line
            className={styles.arrow}
            x1={lx(a.from.cost)}
            y1={ly(a.from.fixes)}
            x2={lx(a.to.cost)}
            y2={ly(a.to.fixes)}
            markerEnd="url(#hca-arrow)"
          />

          {/* Where Hindsight is: the agent's own mark, at its true position.
              The white chip is not decoration — several of these logos are
              near-black line art and would disappear against the hero band.
              Name above and value below, both centred on the mark: beside it,
              two of the three agents land close enough to collide. */}
          <rect
            className={styles.chip}
            x={lx(a.to.cost) - CHIP / 2}
            y={ly(a.to.fixes) - CHIP / 2}
            width={CHIP}
            height={CHIP}
            rx={7}
          />
          <image
            href={a.icon}
            x={lx(a.to.cost) - 9}
            y={ly(a.to.fixes) - 9}
            width={18}
            height={18}
          />
          <text
            className={styles.label}
            x={lx(a.to.cost)}
            y={ly(a.to.fixes) - CHIP / 2 - 7}
            textAnchor="middle">
            {a.name}
          </text>
          <text
            className={styles.valueAfter}
            x={lx(a.to.cost)}
            y={ly(a.to.fixes) + CHIP / 2 + 13}
            textAnchor="middle">
            {a.to.fixes.toFixed(2)}
          </text>
        </g>
      ))}
    </svg>

        <figcaption className={styles.caption}>
          <span className={styles.key}>
            <i className={styles.keyBefore} /> no memory
            <i className={styles.keyAfter} /> with Hindsight
          </span>
          Every agent solves 60–61 of the 61 tasks either way — memory changes
          what it costs to get there. Mean of 3 runs.
        </figcaption>
      </figure>

    </section>
  );
}
