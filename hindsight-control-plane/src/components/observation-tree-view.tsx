"use client";

import { useEffect, useMemo, useState, type ReactNode } from "react";
import { useLocale, useTranslations } from "next-intl";
import {
  CalendarDays,
  ChevronDown,
  ChevronRight,
  FileText,
  Folder,
  FolderOpen,
  Download,
  Search,
  X,
} from "lucide-react";
import {
  client,
  type ObservationTreeFile,
  type ObservationTreeFolder,
  type ObservationTreeObservation,
} from "@/lib/api";
import { useBank } from "@/lib/bank-context";
import { Spinner } from "@/components/ui/spinner";
import { formatAbsoluteDateTime, formatRelativeTime } from "@/lib/relative-time";
import { MemoryDetailModal } from "./memory-detail-modal";
import { Constellation } from "./constellation";
import { TagFilterInput } from "./tag-filter-input";
import { zipFiles } from "@/lib/zip";
import type { GraphData } from "./graph-data";
import { UnderlineTabs } from "@/components/ui/underline-tabs";

// OKF reserves index.md for a folder's overview.
const INDEX = "index.md";

/** The find box works like grep -i: a regex when it parses as one, else the literal text. */
export function findPattern(query: string): RegExp | null {
  const q = query.trim();
  if (!q) return null;
  try {
    return new RegExp(q, "gi");
  } catch {
    return new RegExp(q.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "gi");
  }
}

function matches(re: RegExp, text: string): boolean {
  re.lastIndex = 0;
  return re.test(text);
}

/** Matches per topic file: observations whose text matches, plus the file's own title/path. */
export function findMatches(files: ObservationTreeFile[], re: RegExp): Map<string, number> {
  const hits = new Map<string, number>();
  for (const f of files) {
    if (f.kind !== "topic") continue;
    const n =
      f.observations.filter((o) => matches(re, o.text)).length +
      (matches(re, f.path) || matches(re, f.title) ? 1 : 0);
    if (n > 0) hits.set(f.path, n);
  }
  return hits;
}

function Highlight({ text, re }: { text: string; re: RegExp | null }) {
  if (!re) return <>{text}</>;
  const out: ReactNode[] = [];
  let last = 0;
  // A copy starts at lastIndex 0 without mutating the shared pattern.
  for (const m of text.matchAll(new RegExp(re))) {
    if (!m[0]) break;
    const at = m.index ?? 0;
    out.push(text.slice(last, at));
    out.push(
      <mark key={at} className="rounded-sm bg-amber-300/60 text-foreground dark:bg-amber-500/40">
        {m[0]}
      </mark>
    );
    last = at + m[0].length;
  }
  out.push(text.slice(last));
  return <>{out}</>;
}

const CLUSTER_COLORS = [
  "#0074d9",
  "#2ecc40",
  "#ff851b",
  "#b10dc9",
  "#f012be",
  "#39cccc",
  "#ffdc00",
  "#ff4136",
  "#85144b",
  "#3d9970",
];

/** One star per observation, clustered by its topic and sized by its sources; each topic's
 *  best-supported observation (its hub) links to the hubs of related topics, so related
 *  topics show as bridges. Topics are laid out in tree order, except that a topic's related
 *  topics are pulled right after it, so bridges join neighbours instead of crossing the map.
 *  (Every observation also used to link to its hub: the cluster's blob already shows that,
 *  and the spokes drew a bright star over every cluster.) */
export function treeConstellation(files: ObservationTreeFile[]): GraphData {
  const topics = files.filter((f) => f.kind === "topic" && f.observations.length > 0);
  const byPath = new Map(topics.map((f) => [f.path, f]));
  const ordered: ObservationTreeFile[] = [];
  const placed = new Set<string>();
  const place = (f: ObservationTreeFile) => {
    if (placed.has(f.path)) return;
    placed.add(f.path);
    ordered.push(f);
    for (const r of f.related) {
      const next = byPath.get(r);
      if (next) place(next);
    }
  };
  topics.forEach(place);
  const nodes: GraphData["nodes"] = [];
  const links: GraphData["links"] = [];
  // Related lists are not symmetric (each topic keeps its own top few), so dedupe by pair
  // rather than keeping only the lexically-first side, which dropped one-sided links.
  const bridged = new Set<string>();
  for (const f of ordered) {
    for (const o of f.observations) {
      nodes.push({ id: o.id, label: o.text, size: o.proof_count, group: f.path });
    }
    for (const r of f.related) {
      const other = byPath.get(r);
      const pair = [f.path, r].sort().join("|");
      if (!other || bridged.has(pair)) continue;
      bridged.add(pair);
      links.push({
        source: f.observations[0].id,
        target: other.observations[0].id,
        type: "related",
        color: "#94a3b8",
      });
    }
  }
  return { nodes, links };
}

/** Topic titles for the constellation; two topics with the same title (in different
 *  folders) get their folder appended, so no two clusters read the same. */
export function clusterLabels(files: ObservationTreeFile[]): Map<string, string> {
  const topics = files.filter((f) => f.kind === "topic");
  const count = new Map<string, number>();
  for (const f of topics) count.set(f.title, (count.get(f.title) ?? 0) + 1);
  return new Map(
    topics.map((f) => {
      const folder = f.path.includes("/") ? f.path.slice(0, f.path.lastIndexOf("/")) : "/";
      return [f.path, (count.get(f.title) ?? 0) > 1 ? `${f.title} · ${folder}` : f.title];
    })
  );
}

/** Topics in the same top-level folder share a color. */
function clusterColor(key: string): string {
  let h = 0;
  for (const c of key.split("/")[0]) h = (h * 31 + c.charCodeAt(0)) >>> 0;
  return CLUSTER_COLORS[h % CLUSTER_COLORS.length];
}

// In the viewer's language, in UTC like the markdown beside it ("Times are UTC").
function day(s: string | null | undefined, locale: string): string {
  return s
    ? new Date(s).toLocaleString(locale, {
        dateStyle: "medium",
        timeStyle: "short",
        timeZone: "UTC",
      }) + " UTC"
    : "";
}

/** When it happened if known, else when it was mentioned. */
function when(o: ObservationTreeObservation, locale: string): string {
  const start = day(o.occurred_start, locale);
  const end = day(o.occurred_end, locale);
  if (start && end && start !== end) return `${start} – ${end}`;
  return start || end || day(o.mentioned_at, locale);
}

function collect(
  folder: ObservationTreeFolder,
  files: ObservationTreeFile[],
  folders: Map<string, ObservationTreeFolder>
) {
  folders.set(folder.path, folder);
  files.push(...folder.files);
  folder.folders.forEach((sub) => collect(sub, files, folders));
}

function parentsOf(path: string): string[] {
  const parts = path.split("/").slice(0, -1);
  return ["", ...parts.map((_, i) => parts.slice(0, i + 1).join("/"))];
}

interface TreeProps {
  folder: ObservationTreeFolder;
  depth: number;
  open: Set<string>;
  close: (path: string) => void;
  selected: string;
  select: (path: string) => void;
  hits: Map<string, number> | null;
}

function folderHits(folder: ObservationTreeFolder, hits: Map<string, number>): number {
  return (
    folder.files.reduce((n, f) => n + (hits.get(f.path) ?? 0), 0) +
    folder.folders.reduce((n, sub) => n + folderHits(sub, hits), 0)
  );
}

function Count({ n, hit }: { n: number; hit?: boolean }) {
  return (
    <span
      className={`ml-auto shrink-0 rounded-full px-1.5 py-0.5 text-[10px] leading-none ${
        hit ? "bg-primary/15 text-primary font-medium" : "text-muted-foreground"
      }`}
    >
      {n}
    </span>
  );
}

function rowClass(active: boolean) {
  return `group flex items-center gap-1.5 pr-2 py-1.5 cursor-pointer border-l-2 transition-colors ${
    active
      ? "bg-primary/10 border-primary text-foreground"
      : "border-transparent hover:bg-muted text-foreground"
  }`;
}

function TreeItems({ folder, depth, open, close, selected, select, hits }: TreeProps) {
  const files = folder.files.filter((f) =>
    hits ? hits.has(f.path) : f.name !== INDEX || depth === 0
  );
  const folders = hits ? folder.folders.filter((sub) => folderHits(sub, hits) > 0) : folder.folders;
  const fileRow = (f: ObservationTreeFile) => (
    <li key={f.path}>
      <div
        className={rowClass(selected === f.path)}
        style={{ paddingLeft: `${depth * 18 + 10}px` }}
        onClick={() => select(f.path)}
        title={f.title}
      >
        <span className="w-3.5 flex-shrink-0" />
        <FileText className="w-4 h-4 flex-shrink-0 text-muted-foreground" />
        <span className="text-sm truncate">{f.name}</span>
        {f.kind === "topic" && (
          <Count n={hits?.get(f.path) ?? f.observation_count} hit={hits !== null} />
        )}
      </div>
    </li>
  );
  return (
    <>
      {files.filter((f) => f.kind === "index").map(fileRow)}
      {folders.map((sub) => {
        const isOpen = hits !== null || open.has(sub.path);
        const n = hits ? folderHits(sub, hits) : sub.observation_count;
        return (
          <li key={sub.path}>
            <div
              className={rowClass(selected === `${sub.path}/${INDEX}`)}
              style={{ paddingLeft: `${depth * 18 + 10}px` }}
              onClick={() =>
                // A folder opens on its index.md; clicking it again while there closes it.
                isOpen && selected === `${sub.path}/${INDEX}`
                  ? close(sub.path)
                  : select(`${sub.path}/${INDEX}`)
              }
              title={sub.title}
            >
              {isOpen ? (
                <ChevronDown className="w-3.5 h-3.5 flex-shrink-0 text-muted-foreground" />
              ) : (
                <ChevronRight className="w-3.5 h-3.5 flex-shrink-0 text-muted-foreground" />
              )}
              {isOpen ? (
                <FolderOpen className="w-4 h-4 flex-shrink-0 text-amber-500" />
              ) : (
                <Folder className="w-4 h-4 flex-shrink-0 text-amber-500" />
              )}
              <span className="text-sm truncate">{sub.name}</span>
              <Count n={n} hit={hits !== null} />
            </div>
            {isOpen && (
              <ul>
                <TreeItems
                  folder={sub}
                  depth={depth + 1}
                  open={open}
                  close={close}
                  selected={selected}
                  select={select}
                  hits={hits}
                />
              </ul>
            )}
          </li>
        );
      })}
      {files.filter((f) => f.kind === "topic").map(fileRow)}
    </>
  );
}

function ObservationRow({
  o,
  re,
  onOpen,
}: {
  o: ObservationTreeObservation;
  re: RegExp | null;
  onOpen: (id: string) => void;
}) {
  const t = useTranslations("observationTree");
  const locale = useLocale();
  const date = when(o, locale);
  return (
    <li
      className="group cursor-pointer border-b border-border py-3 last:border-b-0 hover:bg-muted/40 -mx-2 px-2 rounded-sm"
      onClick={() => onOpen(o.id)}
    >
      <p className="text-sm leading-6 text-foreground">
        <Highlight text={o.text} re={re} />
      </p>
      <div className="mt-1.5 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted-foreground">
        {date && (
          <span className="inline-flex items-center gap-1">
            <CalendarDays className="w-3 h-3" />
            {date}
          </span>
        )}
        <span>{t("sources", { count: o.proof_count })}</span>
        {o.updated_at && (
          <span title={formatAbsoluteDateTime(o.updated_at)}>
            {t("updated", { when: formatRelativeTime(o.updated_at) })}
          </span>
        )}
        {o.tags.map((tag) => (
          <span
            key={tag}
            className="rounded-full bg-muted-foreground/10 px-1.5 py-0.5 text-[10px] leading-none"
          >
            {tag}
          </span>
        ))}
      </div>
    </li>
  );
}

export function ObservationTreeView() {
  const t = useTranslations("observationTree");
  const { currentBank: bankId } = useBank();
  const [tree, setTree] = useState<ObservationTreeFolder | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [open, setOpen] = useState<Set<string>>(new Set());
  const [selected, setSelected] = useState(INDEX);
  const [query, setQuery] = useState("");
  const [openMemory, setOpenMemory] = useState<string | null>(null);
  const [mode, setMode] = useState<"files" | "constellation">("files");
  const [raw, setRaw] = useState(false);

  // Scope: the tree is rebuilt from the matching observations only, so a scope gets its own
  // topics. Strict modes, like the entities page: a tag should not pull in untagged memories.
  const [scopeTags, setScopeTags] = useState<string[]>([]);
  const [scopeMatch, setScopeMatch] = useState<"any" | "all">("any");

  useEffect(() => {
    if (!bankId) return;
    let stale = false;
    setTree(null);
    setError(null);
    // Topic paths differ per scope; start each scope at its root index.md.
    setSelected(INDEX);
    setOpen(new Set());
    client
      .getObservationTree(bankId, {
        tags: scopeTags,
        tags_match: scopeMatch === "all" ? "all_strict" : "any_strict",
      })
      .then((t) => !stale && setTree(t))
      .catch((e) => !stale && setError(e instanceof Error ? e.message : String(e)));
    return () => {
      stale = true;
    };
  }, [bankId, scopeTags, scopeMatch]);

  const { files, byPath, folders } = useMemo(() => {
    const all: ObservationTreeFile[] = [];
    const dirs = new Map<string, ObservationTreeFolder>();
    if (tree) collect(tree, all, dirs);
    return { files: all, byPath: new Map(all.map((f) => [f.path, f])), folders: dirs };
  }, [tree]);

  const graph = useMemo(() => treeConstellation(files), [files]);
  const labels = useMemo(() => clusterLabels(files), [files]);
  // Dots colored by when each observation was mentioned, like the memories view.
  const recency = useMemo(() => {
    const times = new Map<string, number>();
    for (const f of files)
      for (const o of f.observations) {
        const t = Date.parse(o.mentioned_at ?? o.updated_at ?? "");
        if (!Number.isNaN(t)) times.set(o.id, t);
      }
    const all = [...times.values()];
    return all.length ? { times, min: Math.min(...all), max: Math.max(...all) } : null;
  }, [files]);
  const re = useMemo(() => findPattern(query), [query]);
  const hits = useMemo(() => (re ? findMatches(files, re) : null), [files, re]);
  const totalHits = hits ? [...hits.values()].reduce((a, b) => a + b, 0) : 0;

  const select = (path: string) => {
    setSelected(path);
    setOpen((prev) => new Set([...prev, ...parentsOf(path)]));
  };
  const close = (path: string) =>
    setOpen((prev) => new Set([...prev].filter((p) => p !== path && !p.startsWith(`${path}/`))));

  const scopeBar = (
    <TagFilterInput
      value={scopeTags}
      onChange={setScopeTags}
      bankId={bankId}
      matchMode={scopeMatch}
      onMatchModeChange={setScopeMatch}
      className="mb-3 max-w-[480px]"
    />
  );
  if (error)
    return (
      <div>
        {scopeBar}
        <p className="text-sm text-destructive">{error}</p>
      </div>
    );
  if (!tree)
    return (
      <div>
        {scopeBar}
        <div className="flex items-center justify-center py-24">
          <Spinner size="lg" variant="jump" />
        </div>
      </div>
    );
  if (tree.observation_count === 0)
    return (
      <div>
        {scopeBar}
        <p className="text-muted-foreground">{t("empty")}</p>
      </div>
    );

  const file = byPath.get(selected);
  const dir =
    file?.kind === "index" ? folders.get(selected.split("/").slice(0, -1).join("/")) : null;
  const shown = file
    ? re && file.kind === "topic"
      ? file.observations.filter((o) => matches(re, o.text))
      : file.observations
    : [];

  // The bundle as shown (scope applied), with every file's markdown and frontmatter: an OKF
  // bundle another tool or agent can read as-is.
  const exportZip = () => {
    const blob = new Blob([zipFiles(files.map((f) => ({ path: f.path, content: f.content })))], {
      type: "application/zip",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${(bankId ?? "bank").replace(/[^\w.-]+/g, "_")}-observation-tree.zip`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const switcher = (
    <div className="mb-3 flex items-center justify-between gap-2">
      <UnderlineTabs
        className="flex-1"
        value={mode}
        onChange={setMode}
        tabs={[
          { id: "files", label: t("filesMode") },
          { id: "constellation", label: t("constellationMode") },
        ]}
      />
      <button
        onClick={exportZip}
        className="shrink-0 inline-flex items-center gap-1.5 rounded-md border border-border bg-background px-2.5 py-1 text-xs text-muted-foreground hover:text-foreground"
      >
        <Download className="w-3.5 h-3.5" />
        {t("exportZip")}
      </button>
    </div>
  );

  if (mode === "constellation")
    return (
      <div>
        {scopeBar}
        {switcher}
        <Constellation
          data={graph}
          height={640}
          nodeSizeFn={(node) => 2 + Math.sqrt(node.size ?? 1)}
          sizeLegendLabel={t("sourcesLegend")}
          clusterKeyFn={(node) => node.group ?? null}
          clusterColorFn={clusterColor}
          clusterLabelFn={(key) => labels.get(key) ?? key}
          nodeHeatFn={
            recency && recency.max > recency.min
              ? (node) => {
                  const t = recency.times.get(node.id);
                  return t === undefined ? 0 : (t - recency.min) / (recency.max - recency.min);
                }
              : undefined
          }
          heatLegendLabel={t("recencyLegend")}
          heatLegendEndpoints={
            recency
              ? [
                  new Date(recency.min).toISOString().slice(0, 10),
                  new Date(recency.max).toISOString().slice(0, 10),
                ]
              : undefined
          }
          onNodeClick={(node) => setOpenMemory(node.id)}
        />
        <MemoryDetailModal memoryId={openMemory} onClose={() => setOpenMemory(null)} />
      </div>
    );

  return (
    <div>
      {scopeBar}
      {switcher}
      <div className="flex flex-col md:flex-row items-stretch overflow-hidden h-[calc(100dvh-17rem)] min-h-[520px]">
        <aside className="w-full max-h-[45%] md:w-1/3 md:max-h-none flex-shrink-0 bg-muted/30 border-b md:border-b-0 md:border-r border-border flex flex-col">
          <div className="shrink-0 border-b border-border p-2">
            <div className="relative flex items-center pl-7 pr-7 py-1 rounded-md bg-background border border-border focus-within:ring-1 focus-within:ring-primary">
              <Search className="absolute left-2 top-2 w-3.5 h-3.5 text-muted-foreground pointer-events-none" />
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder={t("findPlaceholder")}
                aria-label={t("findPlaceholder")}
                className="flex-1 min-w-[6rem] py-0.5 text-sm bg-transparent focus:outline-none"
              />
              {query && (
                <button
                  onClick={() => setQuery("")}
                  className="absolute right-1.5 top-1.5 rounded p-0.5 text-muted-foreground hover:text-foreground hover:bg-muted"
                  aria-label={t("clearFind")}
                >
                  <X className="w-3.5 h-3.5" />
                </button>
              )}
            </div>
            {hits && (
              <p className="mt-1.5 px-1 text-xs text-muted-foreground">
                {hits.size === 0
                  ? t("noMatches")
                  : t("matchSummary", { matches: totalHits, files: hits.size })}
              </p>
            )}
          </div>
          <div className="flex-1 overflow-y-auto py-1">
            <ul>
              <TreeItems
                folder={tree}
                depth={0}
                open={open}
                close={close}
                selected={selected}
                select={select}
                hits={hits}
              />
            </ul>
          </div>
        </aside>

        <main className="flex-1 min-w-0 overflow-y-auto bg-background">
          <div className="sticky top-0 z-10 border-b border-border bg-muted px-4 py-2 font-mono text-xs text-muted-foreground flex items-center gap-2">
            <span className="truncate flex-1">/{selected}</span>
            {file && (
              <button
                onClick={() => setRaw((r) => !r)}
                className="shrink-0 rounded px-2 py-0.5 border border-border bg-background hover:text-foreground"
              >
                {raw ? t("showRendered") : t("showRaw")}
              </button>
            )}
          </div>
          {file && raw ? (
            // Exactly what the API returns for this file: the markdown an agent would read.
            <pre className="p-4 sm:p-8 text-xs font-mono whitespace-pre-wrap break-words text-foreground">
              {file.content}
            </pre>
          ) : file ? (
            <div className="p-4 sm:p-8 max-w-3xl mx-auto">
              <h2 className="text-2xl font-bold text-foreground">{file.title}</h2>
              <p className="mt-1 text-sm text-muted-foreground">
                {t("observationCount", { count: file.observation_count })}
              </p>
              {file.about.length > 0 && (
                <div className="mt-3 flex flex-wrap gap-1.5">
                  {file.about.map((e) => (
                    <span
                      key={e}
                      className="px-1.5 py-0.5 rounded text-xs bg-blue-500/10 text-blue-600 dark:text-blue-400"
                    >
                      {e}
                    </span>
                  ))}
                </div>
              )}
              {file.related.length > 0 && (
                <p className="mt-3 text-sm text-muted-foreground">
                  {t("related")}{" "}
                  {file.related.map((p, i) => (
                    <span key={p}>
                      {i > 0 && ", "}
                      <button className="text-primary hover:underline" onClick={() => select(p)}>
                        {p}
                      </button>
                    </span>
                  ))}
                </p>
              )}
              {dir && (
                <section className="border-t border-border mt-5 pt-4">
                  <h3 className="text-sm font-semibold text-foreground mb-2">{t("topics")}</h3>
                  <ul className="grid gap-1 sm:grid-cols-2">
                    {[
                      ...dir.folders.map((sub) => ({
                        path: `${sub.path}/${INDEX}`,
                        label: `${sub.name}/`,
                        title: sub.title,
                        n: sub.observation_count,
                        folder: true,
                      })),
                      ...dir.files
                        .filter((f) => f.kind === "topic")
                        .map((f) => ({
                          path: f.path,
                          label: f.name,
                          title: f.title,
                          n: f.observation_count,
                          folder: false,
                        })),
                    ]
                      .sort((a, b) => b.n - a.n)
                      .map((c) => (
                        <li key={c.path}>
                          <button
                            onClick={() => select(c.path)}
                            className="flex w-full items-center gap-2 rounded-md border border-border px-2.5 py-1.5 text-left text-sm hover:bg-muted"
                            title={c.title}
                          >
                            {c.folder ? (
                              <Folder className="w-4 h-4 shrink-0 text-amber-500" />
                            ) : (
                              <FileText className="w-4 h-4 shrink-0 text-muted-foreground" />
                            )}
                            <span className="truncate">{c.label}</span>
                            <Count n={c.n} />
                          </button>
                        </li>
                      ))}
                  </ul>
                </section>
              )}
              <section className="border-t border-border mt-5 pt-4">
                <h3 className="text-sm font-semibold text-foreground mb-1">
                  {file.kind === "index" ? t("bestSupported") : t("observations")}
                  {re && file.kind === "topic" && (
                    <span className="ml-2 font-normal text-muted-foreground">
                      {t("matchesInFile", { shown: shown.length, total: file.observations.length })}
                    </span>
                  )}
                </h3>
                <ul>
                  {shown.map((o) => (
                    <ObservationRow key={o.id} o={o} re={re} onOpen={setOpenMemory} />
                  ))}
                </ul>
              </section>
            </div>
          ) : (
            <p className="p-8 text-muted-foreground">{t("pickFile")}</p>
          )}
        </main>
        <MemoryDetailModal memoryId={openMemory} onClose={() => setOpenMemory(null)} />
      </div>
    </div>
  );
}
