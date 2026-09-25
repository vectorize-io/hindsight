import { test } from 'node:test';
import assert from 'node:assert/strict';
import { toSvg } from './svg.ts';
import type { FlowProps } from './model.ts';

const fig: FlowProps = {
  speed: 1000,
  layout: {
    id: 'app',
    label: 'App',
    children: [
      { id: 'agent', label: 'Agent', sub: 'calls in' },
      { id: 'store', label: 'Memories', shape: 'store' },
    ],
  },
  edges: [{ id: 'call', from: 'agent', to: 'store', label: 'retain()' }],
  steps: [
    {
      label: 'retain()',
      flow: [
        { edges: { edge: 'call', data: 'the conversation' }, say: 'The agent sends what happened.' },
        { show: { store: [{ tag: 'world', tone: 'blue', text: 'Alice joined Google', meta: 'Mar', mark: 'new' }] } },
        { edges: { edge: 'call', back: true }, say: 'It comes back stored.' },
      ],
    },
  ],
};

test('draws every label, the edge and its packets', () => {
  const svg = toSvg(fig);
  for (const label of ['Agent', 'calls in', 'Memories', 'APP', 'retain()']) assert.ok(svg.includes(label), label);
  // one packet per hop, both riding the same routed path
  assert.equal(svg.match(/<animateMotion/g)?.length, 2);
  assert.equal(svg.match(/<mpath /g)?.length, 2);
  // the returning packet runs the path backwards
  assert.ok(svg.includes('keyPoints="1;1;0;0"'));
});

test('shows the card content and the narration', () => {
  const svg = toSvg(fig);
  assert.ok(svg.includes('Alice joined Google'), 'card text');
  assert.ok(svg.includes('WORLD'), 'tag pill');
  assert.ok(svg.includes('#3b82f6'), 'blue tone');
  assert.ok(svg.includes('new'), 'mark');
  assert.ok(svg.includes('The agent sends what happened.'), 'first caption');
  assert.ok(svg.includes('It comes back stored.'), 'later caption');
});

test('is a standalone, themeable, script-free image', () => {
  const svg = toSvg(fig);
  assert.match(svg, /^<svg xmlns="http:\/\/www\.w3\.org\/2000\/svg"/);
  assert.ok(svg.trimEnd().endsWith('</svg>'));
  assert.ok(!svg.includes('<script'), 'no scripts: GitHub would strip them');
  assert.ok(svg.includes('@media (prefers-color-scheme: dark)'), 'dark mode');
  assert.ok(!svg.includes('@font-face') && !svg.includes('http://fonts'), 'no font to fetch');
  // every class is a single attribute — two would make it invalid XML
  for (const tag of svg.match(/<[a-z]+[^>]*>/g) ?? [])
    assert.ok((tag.match(/ class="/g)?.length ?? 0) <= 1, `two class attributes: ${tag.slice(0, 80)}`);
});

test('a figure with no steps still draws its boxes', () => {
  const svg = toSvg({ layout: fig.layout, edges: fig.edges });
  assert.ok(svg.includes('Agent') && svg.includes('Memories'));
  assert.ok(!svg.includes('<animateMotion'), 'nothing to animate');
});
