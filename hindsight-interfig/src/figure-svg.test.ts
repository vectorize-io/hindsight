import { test } from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

// The CLI is how a figure actually gets made, and its promise is that a spec never has to be kept:
// it goes in on stdin and comes back out of the SVG.
const cli = join(dirname(dirname(fileURLToPath(import.meta.url))), 'scripts', 'figure-svg.mjs');
const SPEC = {
  props: {
    layout: {
      children: [
        { id: 'a', label: 'Agent' },
        { id: 'b', label: 'Store', shape: 'store' },
      ],
    },
    edges: [{ id: 'w', from: 'a', to: 'b', label: 'write' }],
    steps: [{ label: 'write', flow: [{ edges: 'w', say: 'A fact goes in.' }] }],
  },
};

test('a spec piped in comes back out of the rendered SVG', () => {
  const dir = mkdtempSync(join(tmpdir(), 'figure-svg-'));
  const out = join(dir, 'out.svg');
  try {
    execFileSync('node', [cli, '-', out], { input: JSON.stringify(SPEC) });
    const svg = readFileSync(out, 'utf8');
    assert.match(svg, /<svg /);
    assert.match(svg, /Agent/);
    assert.deepEqual(JSON.parse(execFileSync('node', [cli, '--spec', out], { encoding: 'utf8' })), SPEC);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
