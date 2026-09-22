# interfig

Lives in this repo and is used from source by `hindsight-docs`. It is never published or built.

Interactive, animated flow figures for React. You describe the boxes, how they nest, and the edges.
interfig lays them out, draws the arrows, and plays each "step" as a packet moving along the edges.

```tsx
import { Flow } from '@vectorize-io/interfig';

<Flow
  layout={{
    children: [
      { id: 'app', label: 'Your App', children: [{ id: 'agent', label: 'AI Agent' }] },
      { id: 'api', label: 'API Server' },
    ],
  }}
  edges={[{ id: 'retain', from: 'agent', to: 'api', label: 'retain' }]}
  steps={[{ label: 'retain()', caption: 'Store a memory', flow: ['retain'] }]}
/>;
```

- `layout` — a tree. A group has `children` (and optional `label`, `direction: 'row' | 'column'`, `gap`, `align`). Only groups with a `label` get a frame. Anything else is a box: `{ id, label, sub?, shape?: 'decision' | 'store', lines?, width? }` (a content card grows to fit the most any step shows in it; `lines` is a minimum, `width` fixes its width) (decision = diamond, store = database cylinder: use it for data at rest, plain boxes for what processes it).
- `edges` — `{ from, to, label?, id?, around? }`. `from`/`to` can be a box or a group id. Default id is `from->to`. `around: 'above' | 'below'` arcs the edge over/under the boxes in between (loops, skip-ahead edges). `quiet: true` draws it only while a step uses it.
- `steps` — buttons under the figure. `flow` is a list of beats played in order. A beat is an edge id, an array of edge ids (run at the same time), or `{ edges?, say?, show?, ms? }`:
  - `edges` can be `{ edge, back?: true, data? }` — `back` runs it backwards, `data` rides along the packet as a small card.
  - `say` replaces the caption while the beat plays.
  - `show: { boxId: content }` fills the content card — either any React node, or rows: `{ tag?, tone?, text, meta?, mark?, mono? }` (a colored tag, the text, a muted detail, a ✓-style mark). Word tags head their row; number tags sit inline. It stays until the step ends. Boxes that some step fills get a fixed width and an empty card, so nothing jumps.
  - `light: [boxId]` lights boxes for just that beat.
  - No `edges` = a pause, for showing things without moving anything. `ms` sets how long the beat lasts (default `speed`).
  - `nodes` lights extra boxes. Steps play in a loop as tabs with a progress line; clicking a tab jumps to it, the pause button freezes the animation where it is, and the 1×/2× button doubles the speed.
- `theme` — `{ accent, fg, muted, bg, surface, border, font }`, or set `--fig-accent`, `--fig-bg`, ... in CSS.

Figures wider than their container shrink to fit (down to half size, then scroll). The ⤢ button opens the figure full screen (Esc closes). Hovering a box lights its edges. Honors `prefers-reduced-motion` (no packets, no auto-play).

`MiniGraph` draws a tiny entity graph you can pass as `show` content: `h(MiniGraph, { nodes, links, lit })`.

## Docusaurus (dark mode)

The docs site sets the `--fig-*` colors for its light and dark themes in `hindsight-docs/src/css/custom.css` (search for `interfig`).

## Gallery

`npm run dev` opens the gallery. Every file in `figures/` is one figure: add a file and it shows up.
Edits reload live. `npm test` runs the unit tests.

Use a figure in the docs:

```mdx
import { Flow } from '@vectorize-io/interfig';
import whatHindsightDoes from '@vectorize-io/interfig/figures/what-hindsight-does';

<Flow {...whatHindsightDoes.props} />
```
