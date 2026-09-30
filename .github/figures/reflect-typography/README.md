# Reflect Markdown typography

`before.png` and `after.png` show the same invented astronomy-club response in
Reflect at a 1440px viewport, cropped to the answer card. All example names and
content are fictional. The browser intercepted API requests locally; no bank data
or model calls were used.

The baseline is main at `0bf42b207`. The before capture uses that version's compiled
CSS and answer markup. The after capture uses this branch. Both use the existing
Inter font and the response in `fixture.md`.

The baseline has no paragraph margins, list markers, or heading size hierarchy
because Tailwind v4 does not load the typography plugin from the legacy config.
The updated answer uses Tailwind Typography, a 70ch limit on text blocks, and
horizontal scrolling for wide tables.
