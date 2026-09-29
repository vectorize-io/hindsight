#!/usr/bin/env node
/**
 * Render a figure as one animated SVG — for a README, a PR, an issue, a blog post.
 *
 *   npm run svg -- - out.svg < spec.json        # a spec on stdin: nothing is left on disk
 *   npm run svg -- what-hindsight-does          # a figure in figures/
 *   npm run svg -- spec.json out.svg            # a spec file (only for specs that live in the repo)
 *   npm run svg -- --spec out.svg               # print back the spec the SVG carries
 *
 * A spec is `{ "props": { layout, edges, steps } }` (or bare props) — the same shape the React
 * figures use, so anything here can move to figures/ and become interactive without a rewrite.
 *
 * Every rendered SVG carries its own spec in <metadata>, so a figure is editable later without
 * anyone having to keep the JSON: read it back with --spec, change it, render again.
 *
 * No install, no browser: plain node (22+ strips the types on the way in).
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { register } from 'node:module';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { toSvg } from '../src/svg.ts';

register('./figure-loader.mjs', import.meta.url); // lets a figure file load without its JSX entry
const root = dirname(dirname(fileURLToPath(import.meta.url)));
const SPEC_OPEN = '<metadata id="figure-spec"><![CDATA[';
const SPEC_CLOSE = ']]></metadata>';
const args = process.argv.slice(2);

if (args[0] === '--spec') {
  const svg = readFileSync(args[1] ?? '', 'utf8');
  const at = svg.indexOf(SPEC_OPEN);
  if (at === -1) throw new Error(`${args[1]}: no figure spec inside this SVG`);
  console.log(svg.slice(at + SPEC_OPEN.length, svg.indexOf(SPEC_CLOSE, at)));
  process.exit(0);
}

const [input, out] = args;
if (!input) {
  console.error('usage: npm run svg -- <-|spec.json|figure-slug> [out.svg]   (- reads the spec on stdin)');
  process.exit(2);
}

const stdin = input === '-';
const isJson = stdin || input.endsWith('.json');
const loaded = stdin
  ? JSON.parse(readFileSync(0, 'utf8'))
  : isJson
    ? (await import(resolve(input), { with: { type: 'json' } })).default
    : (await import(join(root, 'figures', `${input}.ts`))).default;
const props = loaded.props ?? loaded;
if (!props?.layout) throw new Error(`${input}: no figure props (expected { props: { layout, edges, steps } })`);

const dest = out ?? (stdin ? 'figure.svg' : `${isJson ? input.replace(/\.json$/, '') : input}.svg`);
const svg = toSvg(props);
// The spec rides along in <metadata>: an SVG is then its own source, and no JSON has to be kept.
// `]]>` would close the CDATA early; it cannot appear in JSON-encoded text, but be sure.
const spec = JSON.stringify({ props }).replaceAll(']]>', ']]\\u003e');
const withSpec = svg.replace(/(<svg[^>]*>\n?)/, `$1${SPEC_OPEN}${spec}${SPEC_CLOSE}\n`);
writeFileSync(dest, withSpec);
console.log(`${dest} — ${(withSpec.length / 1024).toFixed(1)} kB`);
