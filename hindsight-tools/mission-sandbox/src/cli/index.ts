#!/usr/bin/env node
/** mission-sandbox CLI — headless driver for the validator-driven mission loop. */

import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { Command } from "commander";

import {
  loadProjectEnv,
  readStatus,
  runCurate,
  runInit,
  runInspect,
  runLog,
  runMission,
  runTrace,
  runNote,
  runObserveApply,
  runRetainApply,
  runRetainCheck,
  runSnapshot,
  type CurationKind,
  type MissionKind,
} from "../core/index.js";

// Pick up the Hindsight deployment's .env (LLM provider/model/key, API key) for headless runs.
loadProjectEnv();

const log = (msg: string) => process.stdout.write(`${msg}\n`);

function packageRoot(): string {
  // dist/cli/index.js or src/cli/index.ts -> two levels up is the package root
  return path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");
}

function launchUi(projectsDir: string, port: string): void {
  const root = packageRoot();
  const standalone = path.join(root, "standalone", "server.js");
  const env = {
    ...process.env,
    PORT: port,
    MISSION_SANDBOX_PROJECTS_DIR: path.resolve(projectsDir),
  };

  if (existsSync(standalone)) {
    log(`Starting Mission Sandbox UI (standalone) on http://localhost:${port}`);
    spawn("node", [standalone], { stdio: "inherit", env });
    return;
  }

  const nextBin = path.join(
    root,
    "node_modules",
    ".bin",
    process.platform === "win32" ? "next.cmd" : "next"
  );
  if (!existsSync(nextBin)) {
    throw new Error(
      "UI not built and Next.js not found. Run `npm run build` first, or `npm run dev` for development."
    );
  }
  log(`Starting Mission Sandbox UI (dev) on http://localhost:${port}`);
  spawn(nextBin, ["dev", "-p", port], { stdio: "inherit", env, cwd: root });
}

const program = new Command();
program
  .name("mission-sandbox")
  .description("Tune Hindsight retain/observation missions; verify with an external validator.");

program
  .command("init")
  .description("Bind a project to its documents path + API config (no ingest)")
  .argument("<project>", "Project directory to create")
  .requiredOption("--documents <path>", "Path to documents dir or file (bound for re-ingest)")
  .option("--api-url <url>", "Hindsight API URL", "http://localhost:8888")
  .option("--api-key <key>", "Hindsight API key (optional; or set HINDSIGHT_API_KEY)")
  .option("--name <name>", "Bank-id prefix (defaults to the project directory name)")
  .action(
    async (
      project: string,
      opts: { documents: string; apiUrl: string; apiKey?: string; name?: string }
    ) => {
      await runInit(
        {
          projectDir: project,
          documents: opts.documents,
          apiUrl: opts.apiUrl,
          apiKey: opts.apiKey,
          name: opts.name,
        },
        log
      );
    }
  );

const MODEL_HELP =
  "Gemini model for mission refinement (defaults to HINDSIGHT_API_LLM_MODEL or gemini-2.5-flash)";

function missionCommand(kind: MissionKind): Command {
  return new Command("mission")
    .description(`Refine the ${kind} mission from feedback (+ optional examples)`)
    .argument("<project>", "Project directory")
    .requiredOption("--feedback <text>", "What to change, based on your validator's results")
    .option("--example <text...>", "Failing example(s) to ground the refinement", [])
    .option("--model <model>", MODEL_HELP)
    .action(
      async (project: string, opts: { feedback: string; example: string[]; model?: string }) => {
        await runMission(
          {
            projectDir: project,
            kind,
            feedback: opts.feedback,
            examples: opts.example,
            model: opts.model,
          },
          log
        );
      }
    );
}

const retain = new Command("retain").description("Retain (extraction) loop — versioned banks");
retain.addCommand(missionCommand("retain"));
retain
  .command("apply")
  .description("Ingest documents into a NEW bank <project>-vN with the current missions")
  .argument("<project>", "Project directory")
  .action(async (project: string) => {
    await runRetainApply({ projectDir: project }, log);
  });
retain
  .command("check")
  .description("Phase 2: re-extract per-doc into a scratch bank, score coverage of the golden set")
  .argument("<project>", "Project directory")
  .option("--doc <id...>", "Limit to specific doc ids (default: all golden docs)")
  .option("--model <model>", MODEL_HELP)
  .action(async (project: string, opts: { doc?: string[]; model?: string }) => {
    const { perDoc } = await runRetainCheck(
      { projectDir: project, docs: opts.doc, model: opts.model },
      log
    );
    for (const d of perDoc) {
      if (d.missing.length) log(`  ${d.docId} missing: ${d.missing.slice(0, 3).join(" | ")}`);
    }
  });
program.addCommand(retain);

const observe = new Command("observe").description("Observation (consolidation) loop — in place");
observe.addCommand(missionCommand("observe"));
observe
  .command("apply")
  .description("Clear observations on the current bank and re-consolidate")
  .argument("<project>", "Project directory")
  .action(async (project: string) => {
    await runObserveApply({ projectDir: project }, log);
  });
program.addCommand(observe);

// -- Phase 1: curate the current bank to a golden snapshot ----------------------

program
  .command("inspect")
  .description("List facts in the current bank (filter by --doc / --grep) to trace a failure")
  .argument("<project>", "Project directory")
  .option("--doc <id>", "Filter by document id")
  .option("--grep <text>", "Full-text search")
  .action(async (project: string, opts: { doc?: string; grep?: string }) => {
    const rows = await runInspect({ projectDir: project, doc: opts.doc, grep: opts.grep });
    log(`${rows.length} fact(s):`);
    for (const r of rows) log(`  ${r.id}  [${r.docId ?? "?"}]  ${r.text}`);
  });

program
  .command("trace")
  .description("Recall what the bank retrieves for a question + show the evidence doc's memories")
  .argument("<project>", "Project directory")
  .requiredOption("--query <text>", "The (failing) eval question")
  .option("--doc <id...>", "Evidence document id(s) the answer should come from")
  .action(async (project: string, opts: { query: string; doc?: string[] }) => {
    const { retrieved, evidence } = await runTrace({
      projectDir: project,
      query: opts.query,
      docs: opts.doc,
    });
    log(`Retrieved for "${opts.query}":`);
    for (const r of retrieved.slice(0, 8)) log(`  ${r.id}  [${r.docId ?? "?"}]  ${r.text}`);
    for (const e of evidence) {
      log(`\nEvidence doc ${e.docId} (${e.facts.length} facts):`);
      for (const f of e.facts) log(`  ${f.id}  ${f.text}`);
    }
  });

program
  .command("curate")
  .description("Edit / invalidate / revert a memory in place (no re-ingest)")
  .argument("<project>", "Project directory")
  .argument("<memoryId>", "Memory id (from `inspect`)")
  .option("--edit <text>", "Replace the memory text")
  .option("--invalidate", "Soft-retire the memory")
  .option("--revert", "Restore an invalidated memory")
  .option("--reason <text>", "Reason (recorded)")
  .action(
    async (
      project: string,
      memoryId: string,
      opts: { edit?: string; invalidate?: boolean; revert?: boolean; reason?: string }
    ) => {
      const kind: CurationKind = opts.edit ? "edit" : opts.revert ? "revert" : "invalidate";
      await runCurate(
        { projectDir: project, memoryId, kind, text: opts.edit, reason: opts.reason },
        log
      );
    }
  );

program
  .command("log")
  .description("Record a free-form step in the activity log (e.g. an external eval result)")
  .argument("<project>", "Project directory")
  .argument("<summary>", "One-line summary, e.g. 'eval summer-plans → FAIL'")
  .option("--kind <kind>", "Step label", "eval")
  .option("--detail <text>", "Optional detail")
  .action(async (project: string, summary: string, opts: { kind: string; detail?: string }) => {
    await runLog({ projectDir: project, kind: opts.kind, summary, detail: opts.detail });
    log(`logged: ${summary}`);
  });

program
  .command("snapshot")
  .description("Freeze the current bank's memories as the golden target (Phase 1 output)")
  .argument("<project>", "Project directory")
  .action(async (project: string) => {
    await runSnapshot({ projectDir: project }, log);
  });

program
  .command("note")
  .description("Set free-text notes on a version (e.g. validator results)")
  .argument("<project>", "Project directory")
  .argument("<text>", "Note text")
  .option("--version <n>", "Version number (defaults to the current version)")
  .action(async (project: string, text: string, opts: { version?: string }) => {
    await runNote(
      {
        projectDir: project,
        notes: text,
        version: opts.version ? Number(opts.version) : undefined,
      },
      log
    );
  });

program
  .command("status")
  .description("Show bound docs, current missions, and versions")
  .argument("<project>", "Project directory")
  .action(async (project: string) => {
    const s = await readStatus(project);
    log(`Project: ${s.name}  (docs: ${s.documents})`);
    log(`API: ${s.apiUrl}`);
    log(`Current: ${s.currentBank ?? "(none — run `retain apply`)"}`);
    log(`\nRetain mission:\n${s.retainMission ?? "(none)"}`);
    log(`\nObservation mission:\n${s.observeMission ?? "(none)"}`);
    log(`\nVersions (${s.versions.length}):`);
    for (const v of s.versions) {
      const marker = v.n === s.currentVersion ? "*" : " ";
      log(`  ${marker} v${v.n}  ${v.bank}  ${v.createdAt}`);
      if (v.notes) log(`      notes: ${v.notes.replace(/\n/g, "\n             ")}`);
    }
    log(`\nGolden: ${s.goldenCount} memories${s.goldenAt ? ` (frozen ${s.goldenAt})` : ""}`);
    log(`Curations: ${s.curations.length}`);
    if (s.lastCheck) {
      log(
        `Last check: ${(s.lastCheck.coverage * 100).toFixed(0)}% coverage (${s.lastCheck.covered}/${s.lastCheck.total} golden, ${s.lastCheck.docs} docs)`
      );
    }
  });

program
  .command("ui")
  .description("Open the read-only UI to view project status + versions")
  .argument("[projects-dir]", "Directory holding projects", ".")
  .option("-p, --port <port>", "Port", "7777")
  .action((projectsDir: string, opts: { port: string }) => {
    launchUi(projectsDir, opts.port);
  });

program.parseAsync(process.argv).catch((err: unknown) => {
  process.stderr.write(`Error: ${err instanceof Error ? err.message : String(err)}\n`);
  process.exit(1);
});
