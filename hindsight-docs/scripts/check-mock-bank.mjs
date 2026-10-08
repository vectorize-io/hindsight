// Guards the synthetic memory bank behind the overview hero's constellation
// (src/components/constellation/mock-bank.ts). The bank is hand-written facts
// plus generated links, and a typo in an observation's `from` index or a broken
// link rule draws a graph that still renders — just wrong. This checks the
// rules the real `GET /graph` guarantees, so the figure stays one a real bank
// could have produced.
//
// CI runs Node 20 (no TypeScript stripping), so the module is transpiled here
// with the TypeScript compiler the docs package already depends on.
import {readFileSync} from 'node:fs';
import ts from 'typescript';

const src = readFileSync(
  new URL('../src/components/constellation/mock-bank.ts', import.meta.url),
  'utf8'
);
const js = ts.transpileModule(src, {
  compilerOptions: {module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022},
}).outputText;
const {buildMockBank} = await import(`data:text/javascript,${encodeURIComponent(js)}`);

const failures = [];
const check = (ok, msg) => ok || failures.push(msg);

let first;
try {
  first = buildMockBank();
} catch (err) {
  // Most likely an observation's `from` indexes a fact that does not exist.
  // Print the message only: the stack trace carries the whole module as a data URL.
  console.error(`check-mock-bank: buildMockBank() threw: ${err.message}`);
  process.exit(1);
}
const ids = new Set(first.nodes.map((n) => n.id));
check(ids.size === first.nodes.length, 'node ids are not unique');

for (const n of first.nodes) {
  check(typeof n.metadata?.text === 'string' && n.metadata.text.length > 0, `${n.id}: no text`);
  check(!Number.isNaN(Date.parse(n.metadata?.mentioned_at)), `${n.id}: bad mentioned_at`);
}

const LINK_TYPES = new Set(['semantic', 'temporal', 'entity', 'caused_by']);
const degree = new Map();
for (const l of first.links) {
  check(
    ids.has(l.source) && ids.has(l.target),
    `link ${l.source} -> ${l.target}: unknown endpoint`
  );
  check(l.source !== l.target, `link on ${l.source}: self-link`);
  check(LINK_TYPES.has(l.type), `link ${l.source} -> ${l.target}: unexpected type ${l.type}`);
  check(
    l.weight >= 0 && l.weight <= 1,
    `link ${l.source} -> ${l.target}: weight ${l.weight} outside 0..1`
  );
  degree.set(l.source, (degree.get(l.source) ?? 0) + 1);
  degree.set(l.target, (degree.get(l.target) ?? 0) + 1);
}
for (const n of first.nodes) check(degree.get(n.id) > 0, `${n.id}: no links`);

// Observations are sized by their source count, so each must have one.
for (const n of first.nodes.filter((n) => n.metadata.fact_type === 'observation')) {
  check(n.metadata.proof_count >= 1, `${n.id}: observation without sources`);
}

// Deterministic: the same field on the server, after hydration and on reload.
check(
  JSON.stringify(buildMockBank()) === JSON.stringify(first),
  'buildMockBank() is not deterministic'
);

if (failures.length) {
  console.error(
    `check-mock-bank: ${failures.length} problem(s)\n  ${failures.slice(0, 20).join('\n  ')}`
  );
  process.exit(1);
}
console.log(`check-mock-bank: ${first.nodes.length} memories, ${first.links.length} links OK`);
