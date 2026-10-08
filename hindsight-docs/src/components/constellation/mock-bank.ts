// A synthetic memory bank for the docs hero, built the way `GET /graph` builds a
// real one (MemoryEngine.get_graph_data), so it can be hovered without a server:
//
// - Facts (world, experience) carry the links retain writes between facts:
//   `semantic` (similar meaning), `temporal` (close in time) and `caused_by`
//   (one fact explains another — retain writes only this causal kind).
// - Observations have no links of their own. They are consolidated from source
//   facts, inherit those facts' links and entities, are linked `semantic` to
//   observations sharing a source, and their size is their source count.
// - `entity` links are derived, not stored: units mentioning the same entity are
//   linked to their next 10 neighbours in that entity's list.
//
// Deterministic by design (a seeded LCG, no Math.random): the hero must render
// the same field on the server, after hydration, and in every screenshot, and a
// docs figure that reshuffles on reload is a figure nobody can point at.

import type { GraphData, GraphLink, GraphNode } from "./graph-data";

/** Seeded LCG (Numerical Recipes). Enough for placing dots. */
function rng(seed: number) {
  let s = seed >>> 0;
  return () => (s = (s * 1664525 + 1013904223) >>> 0) / 4294967296;
}

type Topic = {
  world: string[];
  experience: string[];
  /** `from` indexes the topic's facts: world first, then experience. */
  observation: Array<{ text: string; from: number[] }>;
  /** [effect, cause] pairs, same indexing as `from`. */
  causedBy?: Array<[number, number]>;
};

/**
 * What a coding agent working with one platform team would remember. Every
 * observation is consolidated from the facts it lists, and says nothing those
 * facts do not support.
 */
const TOPICS: Topic[] = [
  {
    world: [
      "Alice works at Google as a staff engineer", // 0
      "Alice joined the platform team in March 2023", // 1
      "Alice owns the ingestion pipeline", // 2
      "Alice is based in Zurich", // 3
      "Alice reviews every migration before it ships", // 4
      "Alice prefers to be paged over being emailed", // 5
      "Alice ran the Postgres 16 upgrade", // 6
      "Alice wrote the retry policy the ingestion pipeline uses", // 7
    ],
    experience: [
      "I asked Alice to review the retention migration", // 8
      "I paged Alice during the June ingestion incident", // 9
      "I sent Alice the recall benchmark numbers", // 10
      "I summarised the migration plan for Alice", // 11
    ],
    observation: [
      { text: "Ingestion questions should be routed to Alice", from: [2, 7, 9] },
      { text: "Alice answers fastest when paged, not emailed", from: [5, 9] },
      { text: "Migrations need Alice's review before they ship", from: [4, 8, 11] },
    ],
    causedBy: [
      [8, 4],
      [9, 5],
    ],
  },
  {
    world: [
      "The team migrated the dashboard from React to Vue", // 0
      "Vue 3 composition API is the house style", // 1
      "The old React components stayed in the admin app", // 2
      "Vue components are linted with eslint-plugin-vue", // 3
      "The design system was rewritten for Vue in Q2", // 4
      "Storybook runs against the Vue components", // 5
    ],
    experience: [
      "I recommended React to Bob before the switch to Vue", // 6
      "I rewrote the settings page in Vue", // 7
      "I removed the last React dependency from the dashboard", // 8
    ],
    observation: [
      { text: "The team has moved from React to Vue for the dashboard", from: [0, 4, 7, 8] },
      { text: "New UI work should be proposed in Vue, not React", from: [0, 1, 4] },
    ],
    causedBy: [
      [7, 0],
      [8, 0],
    ],
  },
  {
    world: [
      "Memories are stored in Postgres with pgvector", // 0
      "The pgvector index is HNSW with m=16", // 1
      "Keyword search runs on Postgres full-text search", // 2
      "Migrations are managed with Alembic", // 3
      "The production cluster runs Postgres 16", // 4
      "Postgres read replicas serve the recall path", // 5
      "Connection pooling goes through pgbouncer", // 6
    ],
    experience: [
      "I tuned the HNSW parameters for the 10M row bank", // 7
      "I traced a slow recall to a missing Postgres index", // 8
      "I ran the Alembic migration on the tenant schemas", // 9
    ],
    observation: [
      { text: "Slow recall is usually a missing index, not pgvector", from: [0, 8] },
      { text: "Schema changes ship as Alembic migrations", from: [3, 9] },
    ],
  },
  {
    world: [
      "Alice visited Paris in spring 2023", // 0
      "The team offsite was held in Paris", // 1
      "Google has an office in Paris", // 2
      "The Paris team owns the EU region", // 3
      "EU customer data stays in the EU region", // 4
    ],
    experience: [
      "I booked the travel for the Paris offsite", // 5
      "I wrote the trip report after the Paris offsite", // 6
    ],
    observation: [{ text: "EU data residency questions go to the Paris team", from: [3, 4] }],
    causedBy: [[5, 1]],
  },
  {
    world: [
      "Hindsight extracts typed facts instead of storing transcripts", // 0
      "Hindsight recall runs four searches in parallel", // 1
      "Recall results are fused by rank, not by score", // 2
      "A cross-encoder reranks the fused recall results", // 3
      "Recall fills a token budget instead of a top-k", // 4
      "Each observation keeps its evidence and its history", // 5
      "Knowledge pages are rewritten as consolidation runs", // 6
      "Knowledge pages never cite each other", // 7
    ],
    experience: [
      "I ran consolidation on the support bank overnight", // 8
      "I mounted the knowledge pages with hindsight fs mount", // 9
      "I answered an on-call question with reflect", // 10
    ],
    observation: [
      { text: "Recall merges several searches by rank, then reranks", from: [1, 2, 3] },
      { text: "Knowledge pages stay current only while consolidation runs", from: [6, 8] },
    ],
  },
  {
    world: [
      "The on-call rotation is weekly, with a handover on Monday", // 0
      "A Sev1 pages the primary and the secondary on-call at once", // 1
      "The on-call runbook lives in the knowledge pages, not in Notion", // 2
      "Every incident gets a written review within 5 days", // 3
      "Dashboards are in Grafana, alerts go through Alertmanager", // 4
      "More than 5 pages a night triggers an alert tuning pass", // 5
    ],
    experience: [
      "I took the on-call handover from Dana", // 6
      "I silenced the flapping disk alert for a week", // 7
      "I wrote the review for the June ingestion incident", // 8
    ],
    observation: [{ text: "Most on-call pages come from the ingestion pipeline", from: [7, 8] }],
    causedBy: [
      [6, 0],
      [7, 5],
      [8, 3],
    ],
  },
  {
    world: [
      "Bob leads the retrieval team", // 0
      "Bob wrote the reranker benchmark harness", // 1
      "Bob prefers written proposals over meetings", // 2
      "Bob works from the Paris office", // 3
      "Bob owns the recall evaluation suite", // 4
    ],
    experience: [
      "I sent Bob the rank fusion proposal", // 5
      "I reviewed Bob's reranker benchmark numbers", // 6
      "I argued for rank fusion in Bob's design review", // 7
    ],
    observation: [
      { text: "Retrieval changes need Bob's sign-off before they ship", from: [0, 4, 7] },
      { text: "Bob responds to a written proposal, not a meeting invite", from: [2, 5] },
    ],
    causedBy: [[5, 2]],
  },
  {
    world: [
      "Embeddings default to a local SentenceTransformers model", // 0
      "Remote embedding providers include OpenAI, Cohere and TEI", // 1
      "A bank's embedding dimension is fixed when it is created", // 2
      "Re-embedding a bank runs as a background migration", // 3
      "The ONNX embedding provider runs in-process with no server", // 4
    ],
    experience: [
      "I switched the support bank to Cohere embeddings", // 5
      "I benchmarked ONNX embeddings against the local model", // 6
    ],
    observation: [
      { text: "Changing embedding provider means re-embedding the bank", from: [2, 3, 5] },
    ],
  },
  {
    world: [
      "Memories are never deleted on a schedule", // 0
      "Forgetting a fact is explicit, through an invalidation", // 1
      "Invalidated facts stay as evidence for observations", // 2
      "A bank export carries its operations and its queues", // 3
      "A GDPR deletion removes the document and its derived facts", // 4
    ],
    experience: [
      "I invalidated the facts from the retracted ticket", // 5
      "I exported the staging bank before the migration", // 6
    ],
    observation: [
      { text: "Removing a memory must also remove the facts derived from it", from: [1, 4, 5] },
    ],
  },
  {
    world: [
      "The coding agent retains every session it finishes", // 0
      "Each retained document records which coding agent wrote it", // 1
      "The coding agent reads the knowledge pages before writing code", // 2
      "Sessions are captured automatically, not by hand", // 3
      "The Hindsight plugin ships as a skill the coding agent loads", // 4
    ],
    experience: [
      "I read the component map before changing the engine", // 5
      "I corrected a stale memory about the config defaults", // 6
      "I saved the retrieval initiative as a knowledge page", // 7
    ],
    observation: [
      { text: "Coding agent sessions are the bank's largest source of facts", from: [0, 3] },
      { text: "A correction supersedes the stale memory it contradicts", from: [6] },
    ],
  },
  {
    world: [
      "p50 recall latency is 180ms on a 1M memory bank", // 0
      "The cross-encoder is the largest single cost in recall", // 1
      "Graph search adds about 20ms per hop to recall", // 2
      "Retain is asynchronous and returns a receipt", // 3
      "Consolidation runs in a worker, off the request path", // 4
    ],
    experience: [
      "I profiled recall on the 10M row bank", // 5
      "I moved consolidation off the retain request path", // 6
    ],
    observation: [
      { text: "Recall latency is spent on reranking, so cap the candidates", from: [0, 1, 5] },
    ],
    causedBy: [[6, 5]],
  },
  {
    world: [
      "The platform team is based in Zurich", // 0
      "The Monday on-call handover call is run from Zurich", // 1
      "Zurich is the failover target for the EU region", // 2
    ],
    experience: ["I joined the Zurich handover call remotely"], // 3
    observation: [{ text: "The platform team runs on-call from Zurich", from: [0, 1, 3] }],
  },
];

/**
 * Entity extraction, standing in for the LLM's: a fact mentions an entity when
 * its text matches the pattern. Unmatched facts get no entities — as they would
 * in a real bank — and so no entity links.
 */
const ENTITIES: Array<[string, RegExp]> = [
  ["Alice", /\bAlice\b/],
  ["Bob", /\bBob\b/],
  ["Dana", /\bDana\b/],
  ["Google", /\bGoogle\b/],
  ["Zurich", /\bZurich\b/],
  ["Paris", /\bParis\b/],
  ["EU region", /\bEU\b/],
  ["Vue", /\bVue\b/],
  ["React", /\bReact\b/],
  ["Postgres", /\bPostgres|pgvector|pgbouncer\b/],
  ["Alembic", /\bAlembic\b/],
  ["HNSW", /\bHNSW\b/],
  ["Grafana", /\bGrafana\b/],
  ["Cohere", /\bCohere\b/],
  ["ONNX", /\bONNX\b/],
  ["Hindsight", /\bHindsight\b/],
  ["recall", /\brecall\b/i],
  ["reranker", /\brerank|cross-encoder/i],
  ["rank fusion", /\bfus(ed|ion)\b/i],
  ["consolidation", /\bconsolidat/i],
  ["knowledge pages", /\bknowledge pages?\b/i],
  ["embeddings", /\bembedding/i],
  ["ingestion pipeline", /\bingestion\b/i],
  ["on-call", /\bon-call\b|\bpaged\b/i],
  ["migration", /\bmigrations?\b/i],
  ["coding agent", /\bcoding agent\b|\bsessions?\b/i],
  ["invalidation", /\binvalidat|\bforget/i],
  ["deletion", /\bdelet|\bremov(e|ing) a memory/i],
  ["bank export", /\bexport/i],
  ["observations", /\bobservations?\b/i],
  ["incident", /\bincident\b/i],
  ["alerts", /\balerts?\b/i],
  ["retain", /\bretain/i],
  ["corrections", /\bcorrect/i],
];

/** The window the bank's facts were mentioned in, for the recency colour. */
const FIRST_MENTION = Date.parse("2026-01-05T00:00:00Z");
const LAST_MENTION = Date.parse("2026-09-15T00:00:00Z");
const DAY = 86_400_000;
/** The days the agent had sessions on: roughly weekly across the window. */
const SESSIONS = Array.from(
  { length: 36 },
  (_, i) => FIRST_MENTION + (i / 35) * (LAST_MENTION - FIRST_MENTION)
);

/** The API's own node colour: by how many entities the memory mentions. */
function entityColor(count: number): string {
  return count === 0 ? "#e0e0e0" : count === 1 ? "#90caf9" : "#42a5f5";
}

/** The API's own label: the text cut at 30 characters. */
function shortLabel(text: string): string {
  return text.length > 30 ? `${text.slice(0, 30)}...` : text;
}

type Unit = {
  id: string;
  text: string;
  type: "world" | "experience" | "observation";
  entities: string[];
  mentioned: number;
  sources: string[];
};

export function buildMockBank(): GraphData {
  const rand = rng(20260408);
  const facts: Unit[] = [];
  const observations: Unit[] = [];
  const links: GraphLink[] = [];
  const link = (a: string, b: string, type: string, weight: number, entity?: string) =>
    links.push({ source: a, target: b, type, weight, entity });

  TOPICS.forEach((topic, ti) => {
    const mine: Unit[] = [];
    [
      ...topic.world.map((t) => ["world", t] as const),
      ...topic.experience.map((t) => ["experience", t] as const),
    ].forEach(([type, text], i) => {
      const unit: Unit = {
        id: `t${ti}-${type}-${i}`,
        text,
        type,
        entities: ENTITIES.filter(([, re]) => re.test(text)).map(([name]) => name),
        // Facts arrive in sessions, so mention dates cluster by session.
        mentioned: SESSIONS[Math.floor(rand() * SESSIONS.length)] + rand() * 6 * 3_600_000,
        sources: [],
      };
      mine.push(unit);
      facts.push(unit);
    });

    // Semantic links: facts about one topic are mostly similar to each other.
    for (let a = 0; a < mine.length; a++) {
      for (let b = a + 1; b < mine.length; b++) {
        if (rand() < 0.8) link(mine[a].id, mine[b].id, "semantic", 0.7 + rand() * 0.25);
      }
    }
    for (const [effect, cause] of topic.causedBy ?? []) {
      link(mine[effect].id, mine[cause].id, "caused_by", 0.9);
    }

    for (const [k, obs] of topic.observation.entries()) {
      const sources = obs.from.map((i) => mine[i]);
      observations.push({
        id: `t${ti}-observation-${k}`,
        text: obs.text,
        type: "observation",
        // Inherited from the sources, as the graph endpoint does.
        entities: [...new Set(sources.flatMap((s) => s.entities))],
        // Consolidated a few days after its newest evidence was mentioned.
        mentioned: Math.max(...sources.map((s) => s.mentioned)) + (1 + rand() * 4) * DAY,
        sources: sources.map((s) => s.id),
      });
    }
  });

  // Cross-topic semantic links: fewer, and only between facts that share an
  // entity — "Alice ran the Postgres 16 upgrade" is near the Postgres facts.
  for (let a = 0; a < facts.length; a++) {
    for (let b = a + 1; b < facts.length; b++) {
      const sameTopic = facts[a].id.split("-")[0] === facts[b].id.split("-")[0];
      const shared = facts[a].entities.some((e) => facts[b].entities.includes(e));
      if (!sameTopic && shared && rand() < 0.7)
        link(facts[a].id, facts[b].id, "semantic", 0.7 + rand() * 0.2);
      // Temporal links: mentioned within eight days (about one session apart), stronger when closer.
      const gap = Math.abs(facts[a].mentioned - facts[b].mentioned);
      if (gap < 8 * DAY) link(facts[a].id, facts[b].id, "temporal", 1 - gap / (8 * DAY));
    }
  }

  // Observations inherit every link touching one of their sources.
  const factLinks = [...links];
  for (const obs of observations) {
    for (const l of factLinks) {
      if (obs.sources.includes(l.source) && !obs.sources.includes(l.target))
        link(obs.id, l.target, l.type!, l.weight!);
      else if (obs.sources.includes(l.target) && !obs.sources.includes(l.source))
        link(l.source, obs.id, l.type!, l.weight!);
    }
  }
  // ...and are semantically linked to observations sharing a source.
  for (let a = 0; a < observations.length; a++) {
    for (let b = a + 1; b < observations.length; b++) {
      if (observations[a].sources.some((s) => observations[b].sources.includes(s))) {
        link(observations[a].id, observations[b].id, "semantic", 1);
      }
    }
  }

  // Entity links, derived: each unit to its next 10 in each entity's list.
  const units = [...facts, ...observations];
  for (const [name] of ENTITIES) {
    const members = units.filter((u) => u.entities.includes(name));
    members.forEach((u, i) => {
      for (const v of members.slice(i + 1, i + 11)) link(u.id, v.id, "entity", 1, name);
    });
  }

  const nodes: GraphNode[] = units.map((u) => ({
    id: u.id,
    label: shortLabel(u.text),
    color: entityColor(u.entities.length),
    metadata: {
      text: u.text,
      fact_type: u.type,
      entities: u.entities.join(", "),
      mentioned_at: new Date(u.mentioned).toISOString(),
      ...(u.type === "observation" ? { proof_count: u.sources.length } : {}),
    },
  }));

  // The ungrouped layout places nodes around a ring by their index, so a bank
  // listed topic by topic draws as arcs. A real bank comes back interleaved;
  // shuffle (seeded) so the field reads like one.
  for (let i = nodes.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [nodes[i], nodes[j]] = [nodes[j], nodes[i]];
  }

  return { nodes, links };
}
