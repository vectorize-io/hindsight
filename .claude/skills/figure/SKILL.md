---
name: figure
description: Draw an animated figure (boxes, arrows, moving data) as one self-contained SVG for a GitHub README, PR, issue or blog post. Use when a change or an explanation needs a diagram — how a request flows, what a background job does, what a feature changed — or when the user asks for a diagram, figure, animation or "show it visually". The docs site's own pages use the interactive player instead (see step 4).
user_invocable: true
---

# Figure

One animated SVG, no scripts, no upload: it renders and plays anywhere markdown does — GitHub
README, PR and issue comments, blog posts, Notion. Sharp at any size, and it follows the reader's
light or dark theme.

Figures are [Giotto](https://github.com/nicoloboschi/giotto) diagrams: one JSON document, the same
format the docs site's interactive figures use (`hindsight-docs/figures/*.json`).

**You write the diagram JSON — never SVG.** One command renders it. No build, no browser.

## 1. Write the diagram

**Do not save it as a file in the repo.** It is an input, not a deliverable — pipe it in (step 2),
and the rendered SVG keeps a copy of it for you.

```json
{
  "title": "Retain",
  "speed": 2000,
  "elements": [
    { "id": "root", "type": "group", "children": ["agent", "api", "bank"], "layout": { "direction": "row", "gap": 48 } },
    { "id": "agent", "type": "rectangle", "content": [{ "type": "title", "text": "Your AI Agent" }] },
    { "id": "api", "type": "group", "label": { "title": "Hindsight API" }, "children": ["retain"], "layout": { "direction": "column", "gap": 24 } },
    { "id": "retain", "type": "rectangle", "content": [{ "type": "title", "text": "Retain" }, { "type": "subtitle", "text": "LLM extraction" }] },
    { "id": "bank", "type": "group", "label": { "title": "Memory Bank" }, "children": ["facts", "obs"], "layout": { "direction": "column", "gap": 28 } },
    { "id": "facts", "type": "cylinder", "content": [{ "type": "title", "text": "Facts" }, { "type": "subtitle", "text": "world · experience" }] },
    { "id": "obs", "type": "cylinder", "content": [{ "type": "title", "text": "Observations" }] },
    { "id": "call", "type": "arrow", "route": "curved", "start": { "id": "agent" }, "end": { "id": "retain" }, "label": { "text": "retain()" } },
    { "id": "store", "type": "arrow", "route": "curved", "start": { "id": "retain" }, "end": { "id": "facts" }, "label": { "text": "extract" } },
    { "id": "consolidate", "type": "arrow", "route": "curved", "start": { "id": "facts" }, "end": { "id": "obs" }, "label": { "text": "consolidate" }, "quiet": true }
  ],
  "scenes": [
    {
      "label": "retain()",
      "beats": [
        { "edges": { "edge": "call", "data": "“Alice joined Google in March”" }, "say": "Your agent sends what happened." },
        {
          "edges": "store",
          "show": { "facts": [{ "tag": "world", "tone": "blue", "text": "Alice joined Google", "meta": "Mar 2026", "mark": "new" }] },
          "say": "An LLM pulls out the facts."
        },
        {
          "edges": "consolidate",
          "ms": 2600,
          "show": { "obs": [{ "text": "Alice works at Google", "meta": "2 sources" }] },
          "say": "The worker merges them into one belief."
        }
      ]
    }
  ]
}
```

**Elements** — a flat list. A `group` lists its `children` by id and lays them out
(`layout: { direction: "row" | "column", gap, align }`); a `label: { title }` draws a frame around
it. Children line up at the top/left unless the layout says otherwise: use `align: "center"` in rows
and `align: "stretch"` in columns (equal-width boxes), as the docs figures do. `rectangle` boxes are the things that _do_ something; `cylinder` is data at rest; `diamond` is a
decision. A box's `content` is blocks: `title`, `subtitle`, `text`, `list`, `chips`, `code`. Give every
element a stable `id`.

**Arrows** — `{ type: "arrow", start: { id }, end: { id }, label?: { text }, route: "curved", around?, quiet? }`;
the ends name a box _or a group_. `around: "above" | "below"` arcs over the boxes in between;
`quiet: true` draws the arrow only while a scene uses it (for long arrows that would cut across).

**Scenes and beats** — each scene is one story the figure tells; the SVG plays them in a loop. A beat
is one moment: `edges` (an arrow id, `{ edge, back, data }` for a reverse hop or a data chip, or an
array to run several at once), `say` (the caption), `show` (fills the card inside a box until the
scene ends), `light` (highlight boxes for that beat; on the scene, for the whole scene), `ms` (how
long the beat lasts).

**Card rows** — `{ tag?, tone?, text, meta?, mark?, mono? }`. `tone` is `blue | purple | green |
orange | red | gray`. Use `tag` for the kind of thing (`world`, `user`, `page`), `meta` for a detail,
and `mark` for what happened to it (`new`, `✓`, `cited`, `↻`). `**bold**` and `` `code` `` work in
any text.

The full format is the `FORMAT` text in `node_modules/giotto/lib/edit.js`. The 13 figures in
`hindsight-docs/figures/` are working examples — the fastest start is copying the closest one.

Keep it honest and specific: real example data beats placeholders, and every claim in a label,
card or caption must match what the code actually does — check the code, don't assume.

## 2. Render it — from stdin, so nothing is left on disk

```bash
npx giotto export - path/to/out.svg <<'DOC'
{ "title": …, "elements": […], "scenes": […] }
DOC
```

`giotto` comes from the docs site's dependencies (`npm install` at the repo root). The quoted
`<<'DOC'` heredoc passes the JSON through untouched, and the only file produced is the SVG. Layout
and scene warnings print on stderr: fix them. A reference to an element that doesn't exist (an arrow
end, a group child, a box a scene fills) is an error: the export fails and writes nothing.

Other forms:

```bash
npx giotto export hindsight-docs/figures/retain.json out.svg   # a docs figure, as an SVG
npx giotto export doc.json out.svg --theme dark --static       # one still frame, dark
npx giotto spec out.svg                                        # print the diagram an SVG carries
```

Every SVG embeds its diagram in `<metadata id="giotto-doc">`, so a figure stays editable without
anyone keeping the JSON: read it back with `giotto spec`, change what you need, render again.

## 3. Look at it before you ship it

Always. Text that overflows its box, an arrow crossing a box, a caption that does not match what is
moving — obvious on sight, invisible in the source.

```bash
cd $(dirname out.svg) && python3 -m http.server 8777 &   # the browser tool blocks file:// URLs
```

Then open `http://localhost:8777/out.svg`, screenshot it, wait a few seconds and screenshot again to
catch a later beat. `open out.svg` works too when a human is watching.

## 4. Put it where it belongs

- **PR or issue comment** — commit the SVG on the branch, then reference its raw URL:
  `![figure](https://raw.githubusercontent.com/<owner>/<repo>/<branch>/<path>.svg)`. GitHub renders
  and animates it immediately.
- **README or repo docs** — commit it and link it with a relative path.
- **Blog post** — `hindsight-docs/static/img/blog/`, referenced as `/img/blog/<name>.svg`.
- **The docs site's own pages** — don't use an SVG. Those pages use the interactive player, which has
  tabs, pause, speed and hover. Save the diagram as `hindsight-docs/figures/<name>.json` and put it
  on the page:

  ```mdx
  import Figure from '@site/src/components/Figure';
  import retain from '@site/figures/retain.json';

  <Figure doc={retain} />
  ```

## What the SVG cannot do

It loops through every scene with no controls: no tabs, no pause, no hover. If the figure needs
those, it belongs on the docs site. Keep an SVG to one or two scenes so the loop comes back round
quickly.
