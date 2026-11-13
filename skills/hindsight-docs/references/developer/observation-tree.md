

# Observation Tree

The **observation tree** lays out everything a bank has learned as a folder of markdown files: one file per topic, every [observation](./observations.md) in exactly one file, and an `index.md` in each folder that says what is inside. It is the same bundle for a person browsing it in the control plane, an agent running `ls` and `grep` over it, or a tool importing it.

> **📝 Experimental**
>
The endpoint and the file layout may still change.
## Why a Tree

Recall answers a question you already know how to ask. A tree answers the one before it: *what does this bank know?* An agent starting work on a codebase can read the root `index.md`, open the folder that matters, and `grep` the topic files, the same way it explores source code, without guessing a query first. A person can skim the same files to audit what was learned, spot mistakes, and export it.

[Pages](./pages.md) cover a few questions in depth, written and refreshed by an LLM. The observation tree covers *everything*, verbatim: it shows each observation as stored, grouped and named by an algorithm, so it costs nothing to build and can be rebuilt on every request.

## What It Looks Like

```
index.md                 # the bank: its topics and best-supported observations
gemma/
  index.md               # what the gemma folder is about
  vllm.md                # every observation about Gemma on vLLM
  quantization.md
postgres.md
misc-2026-11.md          # observations that fit no topic, by month
```

Each topic file opens with frontmatter, then lists every observation with when it happened and how many sources back it:

```markdown
---
type: Memory Topic
title: "Gemma · vLLM"
description: "42 observations about Gemma, vLLM, NVFP4."
tags: ["Gemma", "vLLM", "NVFP4"]
timestamp: 2026-11-12T16:40:43+00:00
generated:
  by: hindsight/1.2.1
  at: 2026-11-12T16:40:43+00:00
---

# Gemma · vLLM

42 observations. Times are UTC.

**About:** Gemma, vLLM, NVFP4

**Related:** postgres

## Observations

- Gemma 4 runs on vLLM with NVFP4 weights, which is 1.6x faster than FP8. (2026-10-05 12:00 · 6 sources)
- ...
```

- **About** lists the entities that are much more common in this topic than in its folder. It is computed, not written by a model.
- **Related** links topics in other folders that share entities with this one.
- **Times are UTC**, matching the dated notes consolidation writes into observations.

## How Observations Are Grouped

Grouping uses the entities each observation is about (inherited from the facts it was built from) and no LLM:

1. Entities that say nothing about a subject are ignored: ids, hashes, IP addresses, speaker roles, and entities on more than a third of the bank.
2. Each folder starts its groups from its strongest entities, skipping one that mostly appears alongside one already chosen. An observation joins the group of the starting entity it carries most.
3. An observation that carries none of them joins the group it shares the most entities with, and only if that is at least a quarter of its weight. Otherwise it goes to the folder's `misc` file, rather than into a topic it has little to do with.
4. A bank with 120 observations or fewer is one topic file, `all.md` (plus `misc.md` for what shares nothing). A group with more than 120 observations becomes a folder and is split the same way, when it can be: a group with no two distinct entities to split on stays one file. A `misc` file that grows past 120 splits by month (`misc-2026-11.md`), then by ISO week (`misc-2026-w45.md`).

### Stable Paths

The tree is not stored: it is rebuilt from the bank on each request, so it can be filtered by any tag scope without keeping one copy per scope. Paths stay put anyway, because the build has no randomness: input order does not matter, groups start from entity counts (ties broken by name), and each path is named after the entity its group grew from. On a real 2,500-observation bank, a 1% change to the observations leaves over 90% of them in the same file. The test suite holds the build to this.

## Open Knowledge Format

The tree is an [Open Knowledge Format](https://okf.md/spec/) (OKF v0.2) bundle, so tools that read OKF can read it as-is:

- every topic file has YAML frontmatter with a `type` (`Memory Topic`), plus `title`, `description`, `tags`, `timestamp` and `generated`;
- folder overviews use the reserved `index.md`, with no frontmatter except the root's `okf_version`;
- links are bundle-relative (`/gemma/vllm.md`);
- timestamps carry a UTC offset, and are the newest observation's time, so the same bank always gives the same bundle.

## In the Control Plane

Open a bank, go to **Pages → Observations**:

- **Files** shows the tree. Click an observation to open it; **Raw .md** shows the exact markdown.
- **Find** filters the tree like `grep -i` (plain text or a regex) and highlights matches.
- **Filter by tag** builds the tree from the matching observations only.
- **Constellation** draws every observation as a star, clustered by topic, with related topics bridged.
- **Export .zip** downloads the bundle as shown.

## Using the API

`GET /v1/default/banks/{bank_id}/observations/tree` returns the root folder: each folder has `folders` and `files`, and each file has its `path`, `kind` (`index` or `topic`), `title`, `about`, `related`, `observations`, and its full markdown in `content`.

### Python

```python
# The whole bank: folders, topic files and each folder's index.md.
tree = await client.memory.get_observation_tree(bank_id=BANK_ID)
print(f"{tree.observation_count} observations")

def walk(folder):
    for f in folder.files:
        print(f"/{f.path} ({f.observation_count})")
    for sub in folder.folders:
        walk(sub)

walk(tree)

# Only one scope: the same tag filters as recall.
platform = await client.memory.get_observation_tree(bank_id=BANK_ID, tags=["team:platform"], tags_match="any_strict")
walk(platform)
```

### Node.js

```javascript
// The whole bank: folders, topic files and each folder's index.md.
const apiClient = createClient(createConfig({ baseUrl: HINDSIGHT_URL }));
const { data: tree } = await sdk.getObservationTree({ client: apiClient, path: { bank_id: BANK_ID } });
console.log(`${tree.observation_count} observations`);

const walk = (folder) => {
    for (const f of folder.files) console.log(`/${f.path} (${f.observation_count})`);
    folder.folders.forEach(walk);
};
walk(tree);

// Only one scope: the same tag filters as recall.
const { data: platform } = await sdk.getObservationTree({
    client: apiClient,
    path: { bank_id: BANK_ID },
    query: { tags: ['team:platform'], tags_match: 'any_strict' },
});
walk(platform);
```

### CLI

```bash
# The whole bank: folders, topic files and each folder's index.md (-o json for the full tree).
hindsight memory observation-tree "$BANK_ID"

# Only one scope: the same tag filters as recall.
hindsight memory observation-tree "$BANK_ID" --tags team:platform --tags-match any_strict
```

### Go

```go
// The whole bank: folders, topic files and each folder's index.md.
tree, _, err := client.MemoryAPI.GetObservationTree(ctx, mtBankID).Execute()
if err != nil {
	panic(err)
}
fmt.Printf("%d observations\n", tree.GetObservationCount())

var walk func(folder hindsight.ObservationTreeFolder)
walk = func(folder hindsight.ObservationTreeFolder) {
	for _, f := range folder.GetFiles() {
		fmt.Printf("/%s (%d)\n", f.GetPath(), f.GetObservationCount())
	}
	for _, sub := range folder.GetFolders() {
		walk(sub)
	}
}
walk(*tree)

// Only one scope: the same tag filters as recall.
platform, _, err := client.MemoryAPI.GetObservationTree(ctx, mtBankID).
	Tags([]string{"team:platform"}).TagsMatch("any_strict").Execute()
if err != nil {
	panic(err)
}
walk(*platform)
```

To write the bundle to disk, write each file's `content` to its `path`:

### Python

```python
# Write the bundle to disk: each file's markdown at its path.
def write(folder):
    for f in folder.files:
        path = os.path.join(out_dir, f.path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(f.content)
    for sub in folder.folders:
        write(sub)

write(tree)
```

### Node.js

```javascript
// Write the bundle to disk: each file's markdown at its path.
const write = (folder) => {
    for (const f of folder.files) {
        const path = join(outDir, f.path);
        mkdirSync(dirname(path), { recursive: true });
        writeFileSync(path, f.content, 'utf8');
    }
    folder.folders.forEach(write);
};
write(tree);
```

### CLI

```bash
# Write the bundle to disk: each file's markdown at its path.
hindsight memory observation-tree "$BANK_ID" --out "$OUT_DIR"
```

### Go

```go
// Write the bundle to disk: each file's markdown at its path.
var write func(folder hindsight.ObservationTreeFolder)
write = func(folder hindsight.ObservationTreeFolder) {
	for _, f := range folder.GetFiles() {
		path := filepath.Join(outDir, f.GetPath())
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			panic(err)
		}
		if err := os.WriteFile(path, []byte(f.GetContent()), 0o644); err != nil {
			panic(err)
		}
	}
	for _, sub := range folder.GetFolders() {
		write(sub)
	}
}
write(*tree)
```

### Parameters

| Parameter | Description |
|---|---|
| `tags` | Only group observations carrying these tags. Repeat for several. |
| `tags_match` | How `tags` match: `any`, `all`, `any_strict`, `all_strict`, `exact` (as in recall). |
| `tag_groups` | A JSON-encoded compound tag filter, as in recall's body. Not together with `tags`. |

The caller's own tag scope, when its API key has one, always applies on top.
