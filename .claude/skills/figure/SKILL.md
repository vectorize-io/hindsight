---
name: figure
description: Draw an animated figure (boxes, arrows, moving data) as one self-contained SVG for a GitHub README, PR, issue or blog post. Use when a change or an explanation needs a diagram — how a request flows, what a background job does, what a feature changed — or when the user asks for a diagram, figure, animation or "show it visually". The docs site uses the interactive React figures instead.
user_invocable: true
---

# Figure

One animated SVG, no scripts, no upload: it renders and plays anywhere markdown does — GitHub
README, PR and issue comments, blog posts, Notion. ~15 kB for a figure that would be a 2 MB video,
sharp at any size, and it follows the reader's light or dark theme.

**You write a JSON spec — never SVG, never a `.ts` figure file** (those are only for the docs site's
interactive figures, see below). One command renders it. No install, no build, no browser.

## 1. Write the spec

A spec is `{ "props": { "layout": …, "edges": […], "steps": […] } }`. **Do not save it as a file in
the repo.** It is an input, not a deliverable, and a stray `spec.json` is the one mess this skill
must not leave behind — pipe it in (step 2), and the rendered SVG keeps a copy of it for you.

```json
{
  "props": {
    "speed": 2000,
    "layout": {
      "gap": 48,
      "children": [
        { "id": "agent", "label": "Your AI Agent" },
        {
          "id": "api",
          "label": "Hindsight API",
          "direction": "column",
          "gap": 24,
          "children": [{ "id": "retain", "label": "Retain", "sub": "LLM extraction" }]
        },
        {
          "id": "bank",
          "label": "Memory Bank",
          "direction": "column",
          "gap": 28,
          "children": [
            { "id": "facts", "label": "Facts", "sub": "world · experience", "shape": "store" },
            { "id": "obs", "label": "Observations", "shape": "store" }
          ]
        }
      ]
    },
    "edges": [
      { "id": "call", "from": "agent", "to": "retain", "label": "retain()" },
      { "id": "store", "from": "retain", "to": "facts", "label": "extract" },
      { "id": "consolidate", "from": "facts", "to": "obs", "label": "consolidate", "quiet": true }
    ],
    "steps": [
      {
        "label": "retain()",
        "flow": [
          {
            "edges": { "edge": "call", "data": "“Alice joined Google in March”" },
            "say": "Your agent sends what happened."
          },
          {
            "edges": "store",
            "show": {
              "facts": [
                {
                  "tag": "world",
                  "tone": "blue",
                  "text": "Alice joined Google",
                  "meta": "Mar 2026",
                  "mark": "new"
                }
              ]
            },
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
}
```

**Layout** — a tree. A group has `children`, and `label` (which draws a frame around it),
`direction: "row" | "column"`, `gap`, `align`. Anything else is a box: `{ id, label, sub?, shape? }`,
where `shape` is `"store"` for a database cylinder (data at rest) or `"decision"` for a diamond.
Plain boxes are the things that _do_ something. Give every box a stable `id`.

**Edges** — `{ from, to, label?, id?, around?, quiet? }`; `from`/`to` name a box _or a group_.
`around: "above" | "below"` arcs over the boxes in between; `quiet: true` draws the edge only while
a step uses it (for long edges that would cut across the picture).

**Steps and beats** — each step is one story the figure tells; the SVG plays them in a loop. A beat
is one moment: `edges` (a hop id, `{ edge, back, data }` for a reverse hop or a data chip, or an
array to run several at once), `say` (the caption; it stays until the next `say`), `show` (fills the
content card inside a box and persists to the end of the step), `light` (highlight boxes for that
beat), `ms` (how long the beat lasts).

**Card rows** — `{ tag?, tone?, text, meta?, mark?, mono? }`. `tone` is `blue | purple | green |
orange | gray`. Use `tag` for the kind of thing (`world`, `user`, `page`), `meta` for a detail, and
`mark` for what happened to it (`new`, `✓`, `cited`, `↻`).

Keep it honest and specific: real example data beats placeholders, and every claim in a label,
card or caption must match what the code actually does — check the code, don't assume.

## 2. Render it — from stdin, so nothing is left on disk

```bash
cd hindsight-interfig
npm run svg -- - ../path/to/out.svg <<'SPEC'
{ "props": { "layout": …, "edges": […], "steps": […] } }
SPEC
```

The quoted `<<'SPEC'` heredoc passes the JSON through untouched, and the only file produced is the
SVG. Zero dependencies, no browser, no build.

Other forms:

```bash
npm run svg -- what-hindsight-does out.svg   # a figure from figures/, by name
npm run svg -- --spec out.svg                # print the spec an SVG carries, to edit and re-render
```

Every SVG embeds its own spec in `<metadata>`, so a figure stays editable without anyone keeping the
JSON: read it back with `--spec`, change what you need, render again. That is why a spec file is
never worth committing.

Use `npm run svg`, not `node scripts/...` directly: the script name is the interface, the path is not.

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
  and animates it immediately. (Dragging a file into a comment also works, but only a human can do
  that, and GitHub keeps a fresh upload private for several minutes.)
- **README or repo docs** — commit it and link it with a relative path.
- **Blog post** — `hindsight-docs/static/img/blog/`, referenced as `/img/blog/<name>.svg`. An SVG under
  `hindsight-docs/static/` is also read by the docs-skill generator: it pulls the spec out of the
  image and writes the narration into the skill, so an agent reading the docs gets the content and
  not a dead image link.
- **The docs site's own pages** — don't use an SVG. Those pages embed the interactive React figure,
  which has tabs, pause, speed and hover. Add a `hindsight-interfig/figures/<name>.ts` instead and
  `<Flow {...figure.props} />` on the page (see `hindsight-interfig/README.md`).

## A worked example

The spec in step 1 is a complete, working figure — copy it and edit. Any SVG you find carries its
own spec too: `npm run svg -- --spec <file>.svg` prints it back, so an existing figure is the
fastest starting point for a new one.

## What the SVG cannot do

It loops through every step with no controls: no tabs, no pause, no hover. If the figure needs those,
it belongs on the docs site as a React figure. Keep an SVG to one or two steps so the loop comes back
round quickly.
