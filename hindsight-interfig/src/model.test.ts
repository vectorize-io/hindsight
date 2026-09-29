import { test } from 'node:test';
import assert from 'node:assert/strict';
import { decisions, edgeId, isRows, toBeat } from './model.ts';

test('toBeat reads every way a beat can be written', () => {
  assert.deepEqual(toBeat('a->b'), { hops: [{ edge: 'a->b', back: false }] });
  assert.deepEqual(toBeat(['x', { edge: 'y', back: true }]), {
    hops: [
      { edge: 'x', back: false },
      { edge: 'y', back: true },
    ],
  });
  // A hop object is a hop, not a beat, even though both are objects.
  assert.deepEqual(toBeat({ edge: 'x', data: 'chip' }), { hops: [{ edge: 'x', back: false, data: 'chip' }] });
  // A beat keeps its other fields; no edges is a pause.
  assert.deepEqual(toBeat({ say: 'hold', show: { box: 'filled' }, ms: 900 }), {
    say: 'hold',
    show: { box: 'filled' },
    ms: 900,
    hops: [],
  });
});

test('isRows tells row cards from other content', () => {
  assert.equal(isRows([{ text: 'a' }, { tag: 'world', text: 'b' }]), true);
  assert.equal(isRows(['plain', 'strings']), false);
  assert.equal(isRows('text'), false);
  assert.equal(isRows(null), false);
});

test('edges get a default id; decisions are found in nested groups', () => {
  assert.equal(edgeId({ from: 'a', to: 'b' }), 'a->b');
  assert.equal(edgeId({ id: 'call', from: 'a', to: 'b' }), 'call');
  const layout = {
    children: [{ id: 'a', label: 'A' }, { children: [{ id: 'd', label: 'D', shape: 'decision' as const }] }],
  };
  assert.deepEqual(decisions(layout), ['d']);
});
