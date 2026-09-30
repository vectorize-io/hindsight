/**
 * Build-time GitHub star count.
 *
 * The navbar GitHub item and the homepage hero both show the star count — it is
 * the cheapest social proof on the page and turns an unlabeled icon into a
 * target worth clicking. Fetching it at build time rather than from the browser
 * keeps it off the critical path and out of GitHub's unauthenticated rate limit,
 * which a docs site at this traffic would hit within minutes.
 *
 * The count therefore refreshes on deploy, not live. That is the point: a number
 * that is a few hours stale reads the same to a visitor, and nothing here is
 * worth a client-side request.
 *
 * FALLBACK is what renders when the fetch fails — offline dev, a CI runner with
 * no egress, or a rate-limited IP. It is deliberately a round number below the
 * real count: a stale-low number is honest, a stale-high one is not.
 */
const FALLBACK = 10000;
const REPO = 'vectorize-io/hindsight';

/* A build must not be able to hang on this. Without a deadline a slow or
   black-holed api.github.com stalls `docusaurus build` indefinitely, which in CI
   means a job that burns its whole timeout for a decoration. Five seconds is far
   more than the call needs and the fallback covers the rest. */
const TIMEOUT_MS = 5000;

module.exports = function githubStarsPlugin() {
  return {
    name: 'github-stars',

    async loadContent() {
      try {
        const res = await fetch(`https://api.github.com/repos/${REPO}`, {
          signal: AbortSignal.timeout(TIMEOUT_MS),
          headers: {
            Accept: 'application/vnd.github+json',
            // GITHUB_TOKEN lifts the 60/hr unauthenticated limit on shared CI IPs.
            ...(process.env.GITHUB_TOKEN
              ? {Authorization: `Bearer ${process.env.GITHUB_TOKEN}`}
              : {}),
          },
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const {stargazers_count: stars} = await res.json();
        if (typeof stars !== 'number') throw new Error('no stargazers_count');
        return {stars};
      } catch (err) {
        console.warn(
          `[github-stars] could not fetch ${REPO} stars (${err.message}) — using ${FALLBACK}`,
        );
        return {stars: FALLBACK};
      }
    },

    async contentLoaded({content, actions}) {
      actions.setGlobalData(content);
    },
  };
};
