
# Knowledge Pages

**Knowledge pages** are living documents a bank writes about itself. Each page answers one question — "What are the components here?", "What's our error-handling convention?" — and rewrites itself as the bank learns more. They're organized in folders, browsable, searchable, and can be projected onto disk as ordinary markdown files.

The shape is a wiki. The engine underneath is memory.

---

## Mental Models, Simplified

A knowledge page *is* a [mental model](./mental-models). Same synthesis, same background refresh, same provenance.

What's different is how much you have to know to use one. A mental model exposes its mechanics — what it reads, when it rebuilds, how it edits itself. Nobody should have to think about synthesis scope and refresh triggers to keep a wiki. So a page comes with those decisions already made:

- It's built from the bank's [observations](./observations) — consolidated, deduplicated beliefs — rather than raw conversational detail.
- It refreshes incrementally whenever consolidation produces new knowledge in its scope, editing the document rather than regenerating it.
- It never reads other pages, so pages can't cite each other into a feedback loop.
- It gets a larger content budget, because it's a document rather than an answer.

You supply a name and a question. Everything else is a default you can override if you need to — every mental model setting still applies. See the [Knowledge Pages API](./api/knowledge-pages) for the exact defaults and how to change them.

---

## Organized Like a Wiki

Pages live in a tree of folders. Nesting is arbitrary, page names are unique within their folder, and deleting a folder deletes its subtree. That's the whole structural model — a hierarchy, the way anyone would organize documents by hand.

The tree is what makes a knowledge base navigable rather than a flat list of synthesized blobs: `Architecture/`, `Runbooks/`, `Decisions/`, each holding pages that stay current on their own.

---

## Projected as Real Files

Agents already know how to work with a filesystem. So the CLI can mirror a bank's knowledge base onto disk:

```bash
hindsight fs mount --bank my-bank
```

The folder tree becomes real directories, each page a real markdown file with YAML frontmatter, kept current by a background refresh loop. From there, everything ordinary works — `ls`, `cat`, `grep`, `rg`, `fzf`, your editor, an agent's file tools. No SDK, no API client, no new vocabulary.

The same content is available as a portable markdown bundle over the API, for exporting or committing elsewhere.

---

## Searchable

Pages are searchable at the **document level**: a query returns whole pages, ranked, with snippets. It combines full-text and semantic matching, fused server-side, with no reranking step — fast enough to be the first thing an agent reaches for.

That last part matters. Search is a tool an agent *chooses* to call, visible in the transcript, rather than content pushed into its context on every turn. Retrieval the agent asked for informs what it's doing; retrieval it didn't ask for tends to derail it.

This is a different path from [recall](./retrieval), which searches individual memories. Use page search to pick a document; use recall for a specific fact.

---

## Why Not Just Raw Files?

If the answer is documents in a tree, haven't we reinvented files — the thing a memory system exists to replace?

The difference is what sits underneath. A file is where information goes to age. Whoever wrote it last wins, contradictions accumulate quietly, and nothing ever reconciles them. Left alone, a hand-maintained wiki becomes a beautifully formatted lie — not because anyone lied, but because keeping it true is a chore, and chores lose.

A knowledge page is a **projected view** over processed memory, the way a database view is not a table. Before a page is written, Hindsight has already done the work files can't do for themselves: extracted facts from the raw sessions, commits, and documents; deduplicated them; and reconciled their contradictions through consolidation. So when a team decided X and later amended it to Y, the page says Y — and can say why — instead of preserving both a paragraph apart.

Your raw documents remain the source of truth about *what was said*. The pages are the reconciled truth about *what holds*.

This is also why pages heal themselves rather than rot: they aren't the storage, they're the rendering. Delete a page and nothing is lost — it re-projects from memory. Try deleting a wiki and getting it back.

---

## Working With Pages

- **Control plane** — the Knowledge Base view renders the tree, page contents, and which pages have fallen behind.
- **HTTP API** — see [Knowledge Pages API](./api/knowledge-pages) for the full endpoint surface.
- **CLI** — `hindsight fs` mirrors a bank to a local folder of markdown files.
- **Agent tools** — the agent SDK exposes `agent_knowledge_*` tools so an agent can list, read, create, and update its own pages during a session.
