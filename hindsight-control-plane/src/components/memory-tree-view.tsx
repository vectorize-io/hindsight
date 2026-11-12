"use client";

import { useEffect, useMemo, useState, type ReactNode } from "react";
import { useTranslations } from "next-intl";
import {
  CalendarDays,
  ChevronDown,
  ChevronRight,
  FileText,
  Folder,
  FolderOpen,
  Search,
  X,
} from "lucide-react";
import {
  client,
  type MemoryTreeFile,
  type MemoryTreeFolder,
  type MemoryTreeObservation,
} from "@/lib/api";
import { useBank } from "@/lib/bank-context";
import { Spinner } from "@/components/ui/spinner";
import { formatAbsoluteDateTime, formatRelativeTime } from "@/lib/relative-time";
import { MemoryDetailModal } from "./memory-detail-modal";
import { Constellation } from "./constellation";
import type { GraphData } from "./graph-data";
import { UnderlineTabs } from "@/components/ui/underline-tabs";

const README = "README.md";

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
export function findMatches(files: MemoryTreeFile[], re: RegExp): Map<string, number> {
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
  re.lastIndex = 0;
  for (const m of text.matchAll(re)) {
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

/** One star per observation, clustered by its topic and sized by its sources.
 *  Each observation links to its topic's best-supported one (the topic's hub), and hubs
 *  link to the hubs of related topics, so the related-topic links show up as bridges.
 *  (A first version drew one star per topic, which hid the observations entirely.) */
export function treeConstellation(files: MemoryTreeFile[]): GraphData {
  const topics = files.filter((f) => f.kind === "topic" && f.observations.length > 0);
  const hub = new Map(topics.map((f) => [f.path, f.observations[0].id]));
  const nodes: GraphData["nodes"] = [];
  const links: GraphData["links"] = [];
  // Related lists are not symmetric (each topic keeps its own top few), so dedupe by pair
  // rather than keeping only the lexically-first side, which dropped one-sided links.
  const bridged = new Set<string>();
  for (const f of topics) {
    const center = f.observations[0].id;
    for (const o of f.observations) {
      nodes.push({ id: o.id, label: o.text, size: o.proof_count, group: f.path });
      if (o.id !== center) links.push({ source: center, target: o.id, type: "topic" });
    }
    for (const r of f.related) {
      const other = hub.get(r);
      const pair = [f.path, r].sort().join("|");
      if (!other || bridged.has(pair)) continue;
      bridged.add(pair);
      links.push({ source: center, target: other, type: "related", color: "#94a3b8" });
    }
  }
  return { nodes, links };
}

/** Topics in the same top-level folder share a color. */
function clusterColor(key: string): string {
  let h = 0;
  for (const c of key.split("/")[0]) h = (h * 31 + c.charCodeAt(0)) >>> 0;
  return CLUSTER_COLORS[h % CLUSTER_COLORS.length];
}

function day(s?: string | null): string {
  return s
    ? new Date(s).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" })
    : "";
}

/** When it happened if known, else when it was mentioned. */
function when(o: MemoryTreeObservation): string {
  const start = day(o.occurred_start);
  const end = day(o.occurred_end);
  if (start && end && start !== end) return `${start} – ${end}`;
  return start || end || day(o.mentioned_at);
}

function collect(
  folder: MemoryTreeFolder,
  files: MemoryTreeFile[],
  folders: Map<string, MemoryTreeFolder>
) {
  folders.set(folder.path, folder);
  files.push(...folder.files);
  folder.folders.forEach((sub) => collect(sub, files, folders));
}

function parentsOf(path: string): string[] {
  const parts = path.split("/").slice(0, -1);
  return ["", ...parts.map((_, i) => parts.slice(0, i + 1).join("/"))];
}

function displayName(f: MemoryTreeFile): string {
  return f.name.replace(/\.md$/, "");
}

interface TreeProps {
  folder: MemoryTreeFolder;
  depth: number;
  open: Set<string>;
  close: (path: string) => void;
  selected: string;
  select: (path: string) => void;
  hits: Map<string, number> | null;
}

function folderHits(folder: MemoryTreeFolder, hits: Map<string, number>): number {
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
    hits ? hits.has(f.path) : f.name !== README || depth === 0
  );
  const folders = hits ? folder.folders.filter((sub) => folderHits(sub, hits) > 0) : folder.folders;
  const fileRow = (f: MemoryTreeFile) => (
    <li key={f.path}>
      <div
        className={rowClass(selected === f.path)}
        style={{ paddingLeft: `${depth * 18 + 10}px` }}
        onClick={() => select(f.path)}
        title={f.title}
      >
        <span className="w-3.5 flex-shrink-0" />
        <FileText className="w-4 h-4 flex-shrink-0 text-muted-foreground" />
        <span className="text-sm truncate">{f.kind === "readme" ? f.name : displayName(f)}</span>
        {f.kind === "topic" && (
          <Count n={hits?.get(f.path) ?? f.observation_count} hit={hits !== null} />
        )}
      </div>
    </li>
  );
  return (
    <>
      {files.filter((f) => f.kind === "readme").map(fileRow)}
      {folders.map((sub) => {
        const isOpen = hits !== null || open.has(sub.path);
        const n = hits ? folderHits(sub, hits) : sub.observation_count;
        return (
          <li key={sub.path}>
            <div
              className={rowClass(selected === `${sub.path}/${README}`)}
              style={{ paddingLeft: `${depth * 18 + 10}px` }}
              onClick={() =>
                // A folder opens on its README; clicking it again while there closes it.
                isOpen && selected === `${sub.path}/${README}`
                  ? close(sub.path)
                  : select(`${sub.path}/${README}`)
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
  o: MemoryTreeObservation;
  re: RegExp | null;
  onOpen: (id: string) => void;
}) {
  const t = useTranslations("memoryTree");
  const date = when(o);
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

export function MemoryTreeView() {
  const t = useTranslations("memoryTree");
  const { currentBank: bankId } = useBank();
  const [tree, setTree] = useState<MemoryTreeFolder | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [open, setOpen] = useState<Set<string>>(new Set());
  const [selected, setSelected] = useState(README);
  const [query, setQuery] = useState("");
  const [openMemory, setOpenMemory] = useState<string | null>(null);
  const [mode, setMode] = useState<"files" | "constellation">("files");

  useEffect(() => {
    if (!bankId) return;
    client
      .getMemoryTree(bankId)
      .then(setTree)
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, [bankId]);

  const { files, byPath, folders } = useMemo(() => {
    const all: MemoryTreeFile[] = [];
    const dirs = new Map<string, MemoryTreeFolder>();
    if (tree) collect(tree, all, dirs);
    return { files: all, byPath: new Map(all.map((f) => [f.path, f])), folders: dirs };
  }, [tree]);

  const graph = useMemo(() => treeConstellation(files), [files]);
  const re = useMemo(() => findPattern(query), [query]);
  const hits = useMemo(() => (re ? findMatches(files, re) : null), [files, re]);
  const totalHits = hits ? [...hits.values()].reduce((a, b) => a + b, 0) : 0;

  const select = (path: string) => {
    setSelected(path);
    setOpen((prev) => new Set([...prev, ...parentsOf(path)]));
  };
  const close = (path: string) =>
    setOpen((prev) => new Set([...prev].filter((p) => p !== path && !p.startsWith(`${path}/`))));

  if (error) return <p className="text-sm text-destructive">{error}</p>;
  if (!tree)
    return (
      <div className="flex items-center justify-center py-24">
        <Spinner size="lg" variant="jump" />
      </div>
    );
  if (tree.observation_count === 0) return <p className="text-muted-foreground">{t("empty")}</p>;

  const file = byPath.get(selected);
  const dir =
    file?.kind === "readme" ? folders.get(selected.split("/").slice(0, -1).join("/")) : null;
  const shown = file
    ? re && file.kind === "topic"
      ? file.observations.filter((o) => matches(re, o.text))
      : file.observations
    : [];

  const switcher = (
    <UnderlineTabs
      className="mb-3"
      value={mode}
      onChange={setMode}
      tabs={[
        { id: "files", label: t("filesMode") },
        { id: "constellation", label: t("constellationMode") },
      ]}
    />
  );

  if (mode === "constellation")
    return (
      <div>
        {switcher}
        <Constellation
          data={graph}
          height={640}
          nodeSizeFn={(node) => 2 + Math.sqrt(node.size ?? 1)}
          sizeLegendLabel={t("sourcesLegend")}
          clusterKeyFn={(node) => node.group ?? null}
          clusterColorFn={clusterColor}
          clusterLabelFn={(key) => byPath.get(key)?.title ?? key}
          onNodeClick={(node) => setOpenMemory(node.id)}
        />
        <MemoryDetailModal memoryId={openMemory} onClose={() => setOpenMemory(null)} />
      </div>
    );

  return (
    <div>
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
          <div className="sticky top-0 z-10 border-b border-border bg-muted px-4 py-2 font-mono text-xs text-muted-foreground truncate">
            /{selected}
          </div>
          {file ? (
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
                        {p.replace(/\.md$/, "")}
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
                        path: `${sub.path}/${README}`,
                        label: `${sub.name}/`,
                        title: sub.title,
                        n: sub.observation_count,
                        folder: true,
                      })),
                      ...dir.files
                        .filter((f) => f.kind === "topic")
                        .map((f) => ({
                          path: f.path,
                          label: displayName(f),
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
                  {file.kind === "readme" ? t("bestSupported") : t("observations")}
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
