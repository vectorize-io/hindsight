import Link from "next/link";

import { ExtractPanel } from "@/app/components/ExtractPanel";
import { getStatus, listProjects, projectsRoot } from "@/app/lib/project-context";

export const dynamic = "force-dynamic";

type StatusVersion = {
  n: number;
  bank: string;
  retainMission: string | null;
  observeMission: string | null;
  feedback: string[];
  notes: string;
  createdAt: string;
};

export default async function Page({
  searchParams,
}: {
  searchParams: Promise<{ project?: string }>;
}) {
  const { project } = await searchParams;
  return (
    <main className="mx-auto max-w-2xl px-6 py-12">
      {project ? await ProjectView({ name: project }) : await ProjectList()}
    </main>
  );
}

async function ProjectList() {
  const projects = await listProjects();
  return (
    <>
      <h1 className="text-xl font-semibold">Mission Sandbox</h1>
      <p className="mt-1 text-sm text-[var(--muted)]">
        Projects in <code>{projectsRoot()}</code> — driven from the CLI; this view is read-only.
      </p>
      {projects.length === 0 ? (
        <p className="mt-8 text-sm text-[var(--muted)]">
          No projects yet. Create one with <code>mission-sandbox init</code>.
        </p>
      ) : (
        <ul className="mt-8">
          {projects.map((p) => (
            <li key={p.name} className="border-t border-[var(--border)]">
              <Link
                href={`/?project=${encodeURIComponent(p.name)}`}
                className="flex items-baseline justify-between py-3 hover:text-[var(--accent)]"
              >
                <span className="font-medium">{p.name}</span>
                <span className="text-xs text-[var(--muted)]">
                  {p.versions} version{p.versions === 1 ? "" : "s"}
                  {p.currentBank ? ` · ${p.currentBank}` : ""}
                </span>
              </Link>
            </li>
          ))}
        </ul>
      )}
    </>
  );
}

async function ProjectView({ name }: { name: string }) {
  const s = await getStatus(name);
  if (!s) return <p className="text-sm text-[var(--bad)]">No project named “{name}”.</p>;
  const versions = [...s.versions].reverse();
  return (
    <>
      <Link href="/" className="text-xs text-[var(--muted)] hover:text-[var(--accent)]">
        ← all projects
      </Link>
      <h1 className="mt-2 text-xl font-semibold">{s.name}</h1>
      <p className="mt-1 text-xs text-[var(--muted)]">
        <code>{s.documents}</code> · {s.apiUrl} · current{" "}
        <code className="text-[var(--text)]">{s.currentBank ?? "none"}</code>
      </p>

      <Timeline steps={s.steps} />

      <GoldenPanel
        goldenCount={s.goldenCount}
        goldenAt={s.goldenAt}
        curations={s.curations}
        lastCheck={s.lastCheck}
      />

      {s.currentBank ? <ExtractPanel project={s.name} defaultMission={s.retainMission} /> : null}

      {versions.length === 0 ? (
        <p className="mt-10 text-sm text-[var(--muted)]">
          No versions yet. Set a mission (<code>retain mission</code>) and run{" "}
          <code>retain apply</code>.
        </p>
      ) : (
        <div className="mt-10">
          {versions.map((v) => (
            <VersionRow key={v.n} version={v} current={v.n === s.currentVersion} />
          ))}
        </div>
      )}

      <WorkingMissions
        retain={s.retainMission}
        observe={s.observeMission}
        retainFeedback={s.retainFeedback}
        observeFeedback={s.observeFeedback}
      />
    </>
  );
}

function VersionRow({ version, current }: { version: StatusVersion; current: boolean }) {
  return (
    <section className="border-t border-[var(--border)] py-5">
      <div className="flex items-baseline justify-between">
        <h2 className="text-lg font-semibold">
          v{version.n}
          {current ? (
            <span className="ml-2 align-middle text-xs text-[var(--accent)]">current</span>
          ) : null}
        </h2>
        <span className="text-xs text-[var(--muted)]">
          <code>{version.bank}</code> · {version.createdAt.slice(0, 16).replace("T", " ")}
        </span>
      </div>

      {version.feedback.length > 0 ? (
        <div className="mt-3">
          <Label>feedback</Label>
          <ul className="mt-1 space-y-1">
            {version.feedback.map((f, i) => (
              <li key={i} className="border-l-2 border-[var(--border)] pl-3 text-sm">
                {f}
              </li>
            ))}
          </ul>
        </div>
      ) : null}

      <div className="mt-3">
        <Label>notes</Label>
        <p className="mt-1 whitespace-pre-wrap border-l-2 border-[var(--accent)] pl-3 text-sm">
          {version.notes ? version.notes : <span className="text-[var(--muted)]">—</span>}
        </p>
      </div>

      {version.retainMission ? (
        <details className="mt-3 text-sm text-[var(--muted)]">
          <summary className="cursor-pointer">retain mission</summary>
          <p className="mt-1 whitespace-pre-wrap pl-4 text-[var(--text)]">
            {version.retainMission}
          </p>
        </details>
      ) : null}
      {version.observeMission ? (
        <details className="mt-1 text-sm text-[var(--muted)]">
          <summary className="cursor-pointer">observation mission</summary>
          <p className="mt-1 whitespace-pre-wrap pl-4 text-[var(--text)]">
            {version.observeMission}
          </p>
        </details>
      ) : null}
    </section>
  );
}

function WorkingMissions({
  retain,
  observe,
  retainFeedback,
  observeFeedback,
}: {
  retain: string | null;
  observe: string | null;
  retainFeedback: string[];
  observeFeedback: string[];
}) {
  return (
    <details className="mt-8 border-t border-[var(--border)] pt-5 text-sm text-[var(--muted)]">
      <summary className="cursor-pointer">working missions (used on next apply)</summary>
      <div className="mt-3 space-y-4">
        <MissionBlock label="retain" mission={retain} feedback={retainFeedback} />
        <MissionBlock label="observation" mission={observe} feedback={observeFeedback} />
      </div>
    </details>
  );
}

function MissionBlock({
  label,
  mission,
  feedback,
}: {
  label: string;
  mission: string | null;
  feedback: string[];
}) {
  return (
    <div>
      <Label>{label}</Label>
      <p className="mt-1 whitespace-pre-wrap text-[var(--text)]">{mission ?? "—"}</p>
      {feedback.length > 0 ? (
        <ol className="mt-1 list-decimal space-y-0.5 pl-5 text-xs">
          {feedback.map((f, i) => (
            <li key={i}>{f}</li>
          ))}
        </ol>
      ) : null}
    </div>
  );
}

const STEP_ICON: Record<string, string> = {
  init: "📥",
  "retain mission": "✎",
  "observe mission": "✎",
  "retain apply": "⚙",
  "observe apply": "⚙",
  trace: "🔍",
  curate: "✂",
  snapshot: "📌",
  "retain check": "✓",
  eval: "🎯",
  note: "🗒",
};

function Timeline({
  steps,
}: {
  steps: { id: string; at: string; kind: string; summary: string; detail: string | null }[];
}) {
  if (steps.length === 0) return null;
  return (
    <section className="mt-5">
      <h2 className="text-sm font-semibold uppercase tracking-wide text-[var(--muted)]">
        Activity ({steps.length})
      </h2>
      <ol className="mt-2">
        {steps.map((s, i) => (
          <li key={s.id} className="flex gap-3 border-l border-[var(--border)] pl-4 pb-4 relative">
            <span className="absolute -left-2 top-0 text-xs">{STEP_ICON[s.kind] ?? "•"}</span>
            <div className="min-w-0 flex-1">
              <div className="flex items-baseline justify-between gap-2">
                <span className="text-sm">
                  <span className="font-medium">
                    {i + 1}. {s.kind}
                  </span>{" "}
                  <span className="text-[var(--muted)]">— {s.summary}</span>
                </span>
                <span className="shrink-0 text-xs text-[var(--muted)]">{s.at.slice(11, 16)}</span>
              </div>
              {s.detail ? (
                <details className="mt-1 text-xs text-[var(--muted)]">
                  <summary className="cursor-pointer">detail</summary>
                  <pre className="mt-1 whitespace-pre-wrap font-sans text-[var(--text)]">
                    {s.detail}
                  </pre>
                </details>
              ) : null}
            </div>
          </li>
        ))}
      </ol>
    </section>
  );
}

function GoldenPanel({
  goldenCount,
  goldenAt,
  curations,
  lastCheck,
}: {
  goldenCount: number;
  goldenAt: string | null;
  curations: {
    id: string;
    memoryId: string;
    kind: string;
    before: string;
    after: string | null;
    reason: string | null;
  }[];
  lastCheck: { coverage: number; covered: number; total: number; docs: number; at: string } | null;
}) {
  if (goldenCount === 0 && curations.length === 0) return null;
  return (
    <div className="mt-5 rounded-lg border border-[var(--accent)] p-4">
      <div className="flex items-baseline justify-between">
        <span className="text-sm font-semibold">Golden snapshot (Phase 1 → 2)</span>
        <span className="text-xs text-[var(--muted)]">
          {goldenCount} memories
          {goldenAt ? ` · frozen ${goldenAt.slice(0, 16).replace("T", " ")}` : ""}
        </span>
      </div>

      {lastCheck ? (
        <p className="mt-2 text-sm">
          <Label>last mission check</Label> <b>{(lastCheck.coverage * 100).toFixed(0)}%</b> coverage
          ({lastCheck.covered}/{lastCheck.total} golden across {lastCheck.docs} doc
          {lastCheck.docs === 1 ? "" : "s"})
        </p>
      ) : (
        <p className="mt-2 text-xs text-[var(--muted)]">No `retain check` run yet.</p>
      )}

      {curations.length > 0 ? (
        <div className="mt-3">
          <Label>curations ({curations.length})</Label>
          <ul className="mt-1 space-y-1 text-sm">
            {curations.map((c) => (
              <li key={c.id} className="border-l-2 border-[var(--border)] pl-3">
                <span
                  className={c.kind === "invalidate" ? "text-[var(--bad)]" : "text-[var(--accent)]"}
                >
                  {c.kind}
                </span>{" "}
                {c.kind === "edit" ? (
                  <span className="text-[var(--muted)]">
                    “{c.before.slice(0, 50)}…” → “{(c.after ?? "").slice(0, 60)}…”
                  </span>
                ) : (
                  <span className="text-[var(--muted)]">“{c.before.slice(0, 70)}…”</span>
                )}
                {c.reason ? <div className="text-xs text-[var(--muted)]">— {c.reason}</div> : null}
              </li>
            ))}
          </ul>
        </div>
      ) : null}
    </div>
  );
}

function Label({ children }: { children: React.ReactNode }) {
  return (
    <span className="text-[0.7rem] uppercase tracking-wider text-[var(--muted)]">{children}</span>
  );
}
