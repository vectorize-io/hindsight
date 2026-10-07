import React, {type ReactNode} from 'react';
import Link from '@docusaurus/Link';
import {LuArrowUpRight} from 'react-icons/lu';
import styles from './HomeBenchmarks.module.css';

/**
 * Every benchmark we have a published comparison for, Hindsight against the
 * next-best system on that same dataset.
 *
 * It sits with the sdebench chart (CodingAgentsChart) in one Benchmarks section
 * rather than in the hero: the two belong together, and a band carrying both
 * ran past 900px and stopped reading as a hero at all.
 *
 * The runner-up is drawn in grey and left UNNAMED on purpose. It is a different
 * system on almost every row — cognee on LoComo, hybrid-search on three others
 * — so naming them would turn one honest "here is the bar to clear" into five
 * competitor callouts that need their own footnotes. Grey also keeps colour
 * carrying one meaning only: us or not us.
 *
 * Sources, and what to re-check when these move:
 *  - Everything except BEAM comes from the AMB comparison table at
 *    agentmemorybenchmark.ai, which is the live leaderboard.
 *  - AMB lists no competitor on BEAM, so that row's 40.6% is the best published
 *    result on the 10M tier (Honcho), as tabulated in
 *    blog/2026-04-02-beam-sota.md.
 * Every value is direct-labelled, so the chart needs no axis, no hover and no
 * table beside it.
 */
const ROWS: {name: string; ours: number; best?: number}[] = [
  {name: 'LongMemEval-S', ours: 94.6, best: 74.0},
  {name: 'LoComo', ours: 92.0, best: 80.3},
  {name: 'PersonaMem', ours: 86.6, best: 84.4},
  // No `best`: nobody else has published on PrecisionMemBench. Showing the row
  // without a runner-up is the honest way to include it — inventing a bar, or
  // leaving the benchmark out because it flatters us, would both be worse.
  {name: 'PrecisionMemBench', ours: 85.7},
  {name: 'LifeBench', ours: 71.5, best: 61.0},
  {name: 'BEAM · 10M tokens', ours: 64.1, best: 40.6},
];

export default function HomeBenchmarks(): ReactNode {
  return (
    <section className={styles.section}>
      <div className={styles.head}>
        <h3 className={styles.title}>Retrieval accuracy</h3>
        <Link className={styles.more} to="https://benchmarks.hindsight.vectorize.io/">
          All benchmarks
          <LuArrowUpRight size={14} />
        </Link>
      </div>

      <div className={styles.legend}>
        <span className={styles.legendOurs}>Hindsight</span>
        <span className={styles.legendBest}>Next best system</span>
      </div>

      <div className={styles.rows}>
        {ROWS.map((r) => (
          <div className={styles.row} key={r.name}>
            <span className={styles.rowName}>{r.name}</span>
            <div className={styles.bars}>
              <div className={styles.track}>
                <div className={styles.barOurs} style={{width: `${r.ours}%`}} />
                <span className={styles.valueOurs}>{r.ours.toFixed(1)}%</span>
              </div>
              {r.best === undefined ? (
                <span className={styles.noCompare}>no published comparison</span>
              ) : (
                <div className={styles.track}>
                  <div className={styles.barBest} style={{width: `${r.best}%`}} />
                  <span className={styles.valueBest}>{r.best.toFixed(1)}%</span>
                </div>
              )}
            </div>
          </div>
        ))}
      </div>

    </section>
  );
}
