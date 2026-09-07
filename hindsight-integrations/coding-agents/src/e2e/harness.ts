import { execFileSync, spawn, spawnSync } from "node:child_process";
import { existsSync, mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { homedir, tmpdir } from "node:os";
import { basename, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { startStubModel, type StubModel } from "./stub-model";

export interface HarnessDockerSetup {
  name: string;
  hindsightHarness: string;
  installCommand: string;
  /**
   * Where the host keeps this CLI's subscription credentials, and where the CLI expects them in the
   * container. Omitted by harnesses driven through the stub model, which need no real account.
   */
  credentialPath?(): string;
  credentialTarget?: string;
  /**
   * Point this CLI at the local stub model instead of its vendor backend, given the stub's base
   * URL. Set for harnesses whose credentials live in a keyring or an account-bound session and so
   * cannot be handed to a container at all — see ./stub-model for what this does and doesn't prove.
   */
  stubModelEnv?(baseUrl: string): Record<string, string>;
  /**
   * False when the host cannot put Hindsight's context in front of the model, so the E2E asserts
   * retention only. Grok Build is the case: its prompt hook is passive — Grok ignores hook stdout,
   * so the memory block never reaches the conversation no matter what we do. Everything else about
   * the integration (bank setup, session write-back, MCP tools) still works and is still asserted.
   */
  injectsIntoModel?: boolean;
  /**
   * Set to WHY this harness cannot currently be driven end to end, and it is skipped with that
   * reason instead of failing. Kept in the list rather than deleted so the wiring, the credential
   * paths and the reason survive — re-enabling is a matter of clearing this field.
   */
  unsupported?: string;
  /** `ctx.stubUrl` is the stub model's base URL when one is in use, for CLIs that take it as a flag. */
  command(prompt: string, ctx: { stubUrl?: string }): string[];
}

interface RawConfig {
  apiUrl?: string;
  apiToken?: string;
}

export interface E2eRun {
  harness: string;
  bankId: string;
  apiUrl: string;
  apiToken?: string;
  output: string;
  diagnostics: string;
  /** Requests the stub model served; 0 means the CLI never reached it (undefined = no stub used). */
  stubRequests?: number;
}

// The symptom deliberately omits the answer. A passing harness can only produce these literals if
// its real hook lifecycle receives Hindsight's semantic context from the seeded bank.
export const seededDecisionStatuses = ["429", "408"];

const E2E = process.env.HINDSIGHT_HARNESS_E2E === "1";
const PACKAGE_ROOT = resolve(fileURLToPath(new URL("../..", import.meta.url)));
const IMAGE = "hindsight-coding-agents-e2e";
const RUN_MARKER = "HINDSIGHT_E2E_HARNESS_PROMPT";

function required(value: string | undefined, message: string): string {
  if (!value) throw new Error(message);
  return value;
}

function run(command: string, args: string[], options: { cwd?: string } = {}): string {
  const result = spawnSync(command, args, { cwd: options.cwd, encoding: "utf8" });
  if (result.status !== 0) {
    throw new Error(`${command} ${args.join(" ")} failed:\n${result.stderr || result.stdout}`);
  }
  return result.stdout;
}

/**
 * Run the harness container WITHOUT blocking this process's event loop.
 *
 * The stub model (./stub-model) is an HTTP server living in this very process, so a synchronous
 * child process here is fatal to it: the OS still completes the TCP handshake from the container,
 * but no JavaScript can run to answer the request, and the CLI inside hangs on a model call that is
 * never served until the whole docker run hits its timeout. That silently broke EVERY stub-model
 * harness — dcode, qwen-code, dsh — which is why this one call cannot be `spawnSync` like the rest.
 * The credential-mounted harnesses talk to their vendor's backend and never noticed.
 */
function runDocker(
  args: string[],
  timeoutMs: number
): Promise<{ status: number | null; stdout: string; stderr: string }> {
  return new Promise((resolveRun) => {
    const child = spawn("docker", args);
    let stdout = "";
    let stderr = "";
    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", (chunk: string) => (stdout += chunk));
    child.stderr.on("data", (chunk: string) => (stderr += chunk));
    const timer = setTimeout(() => {
      stderr += `\ndocker run exceeded ${timeoutMs}ms and was killed`;
      child.kill("SIGKILL");
    }, timeoutMs);
    child.on("error", (error) => {
      clearTimeout(timer);
      resolveRun({ status: null, stdout, stderr: `${stderr}\n${String(error)}` });
    });
    child.on("close", (status) => {
      clearTimeout(timer);
      resolveRun({ status, stdout, stderr });
    });
  });
}

function loadHindsightConfig(): RawConfig {
  const path =
    process.env.HINDSIGHT_E2E_CONFIG || join(homedir(), ".hindsight", "coding-agent.json");
  if (!existsSync(path)) {
    throw new Error(
      `Hindsight config not found at ${path}; set HINDSIGHT_E2E_CONFIG to a local config file`
    );
  }
  try {
    return JSON.parse(readFileSync(path, "utf8")) as RawConfig;
  } catch {
    throw new Error(`Hindsight config at ${path} is not valid JSON`);
  }
}

function dockerApiUrl(apiUrl: string): string {
  const url = new URL(apiUrl);
  if (["localhost", "127.0.0.1", "::1", "[::1]"].includes(url.hostname)) {
    url.hostname = "host.docker.internal";
  }
  return url.toString().replace(/\/$/, "");
}

function makeTestConfig(bankId: string): {
  hostApiUrl: string;
  apiToken?: string;
  containerConfig: RawConfig & {
    bankId: string;
    autoReflect: boolean;
    autoSeed: boolean;
    codebaseSurvey: boolean;
    gitIngest: "none";
  };
} {
  const source = loadHindsightConfig();
  const hostApiUrl = required(
    process.env.HINDSIGHT_E2E_API_URL || source.apiUrl,
    "Hindsight apiUrl is required"
  );
  const apiToken = process.env.HINDSIGHT_E2E_API_TOKEN || source.apiToken;
  return {
    hostApiUrl,
    apiToken,
    containerConfig: {
      apiUrl: dockerApiUrl(hostApiUrl),
      apiToken,
      bankId,
      // The Docker run must exercise semantic injection, not merely record an empty session.
      autoReflect: true,
      autoSeed: false,
      codebaseSurvey: false,
      gitIngest: "none",
    },
  };
}

function packageTarball(dir: string): string {
  execFileSync("npm", ["pack", "--pack-destination", dir], {
    cwd: PACKAGE_ROOT,
    stdio: "pipe",
  });
  const packed = readdirSync(dir).find((name) => name.endsWith(".tgz"));
  if (!packed) throw new Error("npm pack did not produce a tarball");
  return join(dir, packed);
}

const imageFor = (harness: HarnessDockerSetup): string => `${IMAGE}-${harness.name}`;

/**
 * Build the shared base, then this harness's image on top. One CLI per image: a vendor's broken
 * installer can then only fail its own harness instead of the whole matrix, and rebuilds stay cheap
 * because the base layer is cached across all of them.
 */
function buildImage(harness: HarnessDockerSetup): void {
  run("docker", ["build", "--tag", `${IMAGE}-base`, "--file", "e2e/Dockerfile.base", "e2e"], {
    cwd: PACKAGE_ROOT,
  });
  run(
    "docker",
    ["build", "--tag", imageFor(harness), "--file", `e2e/Dockerfile.${harness.name}`, "e2e"],
    { cwd: PACKAGE_ROOT }
  );
}

function assertPrerequisites(harness: HarnessDockerSetup): void {
  if (!E2E) throw new Error("HINDSIGHT_HARNESS_E2E=1 is required to run Docker harness tests");
  const credentials = harness.credentialPath?.();
  if (credentials && !existsSync(credentials)) {
    throw new Error(`Missing ${harness.name} credentials at ${credentials}`);
  }
  run("docker", ["version", "--format", "{{.Server.Version}}"]); // fail early with a direct message
}

/**
 * Whether this machine can drive `harness` — i.e. its subscription credentials are present.
 *
 * These are real subscription logins, and nobody holds all of them at once, so a missing one must
 * SKIP that harness rather than fail the suite. Callers report the reason so a skip is never
 * mistaken for a pass.
 */
export function harnessCredentialStatus(harness: HarnessDockerSetup): {
  available: boolean;
  reason: string;
} {
  if (harness.unsupported)
    return { available: false, reason: `unsupported — ${harness.unsupported}` };
  // Stub-model harnesses need no account, so they always run — that is the point of the stub.
  if (harness.stubModelEnv) return { available: true, reason: "stub model (no account needed)" };
  const path = harness.credentialPath?.();
  if (!path) return { available: false, reason: `${harness.name} declares no credential source` };
  return existsSync(path)
    ? { available: true, reason: `credentials at ${path}` }
    : { available: false, reason: `no ${harness.name} credentials at ${path}` };
}

export async function runHarnessE2e(harness: HarnessDockerSetup): Promise<E2eRun> {
  assertPrerequisites(harness);
  const root = mkdtempSync(join(tmpdir(), `hindsight-e2e-${harness.name}-`));
  const packageDir = join(root, "package");
  const workDir = join(root, "workspace");
  const resultDir = join(root, "results");
  const configPath = join(root, "hindsight-config.json");
  const bankId = `e2e-${harness.name}-${basename(root)}`.replace(/[^a-zA-Z0-9:_-]/g, "-");
  let stub: StubModel | undefined;

  try {
    run("mkdir", ["-p", packageDir, workDir, resultDir]);
    const tarball = packageTarball(packageDir);
    run("git", ["init", "-q"], { cwd: workDir });
    writeFileSync(join(workDir, "README.md"), "# Hindsight harness E2E fixture\n");
    run("git", ["add", "README.md"], { cwd: workDir });
    run(
      "git",
      [
        "-c",
        "user.email=e2e@example.test",
        "-c",
        "user.name=E2E",
        "commit",
        "-qm",
        "fix: retry only transient failures",
        "-m",
        "Retry every 5xx plus EXACTLY 429 and 408 from the 4xx range; every other 4xx is permanent and must fail fast.",
      ],
      {
        cwd: workDir,
      }
    );

    const config = makeTestConfig(bankId);
    writeFileSync(configPath, JSON.stringify(config.containerConfig));
    const hostConfigPath = join(root, "hindsight-host-config.json");
    writeFileSync(
      hostConfigPath,
      JSON.stringify({ ...config.containerConfig, apiUrl: config.hostApiUrl, gitIngest: "full" })
    );
    // This is the same production ingestion engine used by SessionStart, run in the foreground so
    // the test has a deterministic semantic fixture before it starts the real CLI harness.
    execFileSync(
      "node",
      [
        join(PACKAGE_ROOT, "dist", "deepen.js"),
        "--repo",
        workDir,
        "--bank",
        bankId,
        "--config",
        hostConfigPath,
        "--git-ingest",
        "full",
      ],
      { encoding: "utf8", timeout: 600_000 }
    );
    buildImage(harness);

    const prompt =
      `${RUN_MARKER}: Our HTTP client keeps retrying requests that will never succeed and hammers ` +
      "the auth endpoint after failures. Which 4xx failures are actually safe to retry?";

    // Harnesses whose credentials can't leave the host keyring are pointed at a local echo server
    // instead of their vendor backend; nothing is bypassed, the CLI simply talks to our model.
    const credentials = harness.credentialPath?.();
    stub = harness.stubModelEnv ? await startStubModel() : undefined;
    const stubEnv = stub ? harness.stubModelEnv!(stub.containerUrl) : {};

    const result = await runDocker(
      [
        "run",
        "--rm",
        "--add-host",
        "host.docker.internal:host-gateway",
        "--mount",
        `type=bind,src=${tarball},dst=/plugin/${basename(tarball)},readonly`,
        // Staged read-only; the entrypoint copies it to credentialTarget. Mounting straight onto
        // the target breaks harnesses whose credentials are a directory holding a live SQLite
        // store the CLI opens read-write (copilot, cline), and copying keeps the host's real
        // subscription credentials unwritable by the run.
        ...(credentials && harness.credentialTarget
          ? [
              "--mount",
              `type=bind,src=${credentials},dst=/hindsight-credentials/source,readonly`,
              "--env",
              `HINDSIGHT_E2E_CREDENTIAL_TARGET=${harness.credentialTarget}`,
            ]
          : []),
        ...Object.entries(stubEnv).flatMap(([key, value]) => ["--env", `${key}=${value}`]),
        "--mount",
        `type=bind,src=${configPath},dst=/hindsight/config.json,readonly`,
        "--mount",
        `type=bind,src=${workDir},dst=/workspace`,
        "--mount",
        `type=bind,src=${resultDir},dst=/results`,
        "--env",
        `HINDSIGHT_CONFIG=/hindsight/config.json`,
        "--env",
        "HINDSIGHT_DIAG_FILE=/results/diagnostics.jsonl",
        "--env",
        `HINDSIGHT_E2E_INSTALL_COMMAND=${harness.installCommand}`,
        // Every harness operates on the fixture repo, so start there. Without this the container's
        // cwd is `/`, which makes a CLI resolve the wrong project (Devin refused to run at all).
        "--workdir",
        "/workspace",
        imageFor(harness),
        ...harness.command(prompt, { stubUrl: stub?.containerUrl }),
      ],
      300_000
    );
    if (result.status !== 0) {
      throw new Error(
        `Docker ${harness.name} run failed` +
          (stub ? ` (stub model served ${stub.requests()} requests)` : "") +
          `:\n${result.stderr || result.stdout}`
      );
    }
    const outputPath = join(resultDir, "last-message.txt");
    const diagnosticsPath = join(resultDir, "diagnostics.jsonl");
    const output = existsSync(outputPath) ? readFileSync(outputPath, "utf8") : result.stdout;
    const diagnostics = existsSync(diagnosticsPath) ? readFileSync(diagnosticsPath, "utf8") : "";
    return {
      harness: harness.hindsightHarness,
      bankId,
      apiUrl: config.hostApiUrl,
      apiToken: config.apiToken,
      output,
      diagnostics,
      stubRequests: stub?.requests(),
    };
  } finally {
    await stub?.close();
    rmSync(root, { recursive: true, force: true });
  }
}

export async function getRetainedDocument(run: E2eRun): Promise<unknown> {
  const headers: Record<string, string> = {};
  if (run.apiToken) headers.Authorization = `Bearer ${run.apiToken}`;
  const bankUrl = `${run.apiUrl.replace(/\/$/, "")}/v1/default/banks/${encodeURIComponent(run.bankId)}`;
  let id: string | undefined;
  let lastError: unknown;
  // Retain is intentionally asynchronous. A successful Stop hook can precede document visibility
  // by a few seconds, especially when the configured server has a queue behind extraction work.
  //
  // A CONNECTION error is retried on the same footing as "not visible yet". A whole harness run
  // costs minutes of real CLI and LLM work, so throwing it away because one poll hit the server
  // mid-hiccup is pure waste — and it is not hypothetical: a saturated machine (several harness
  // containers plus their ingest jobs) made this fail twice with `fetch failed` while the API was
  // healthy before and after.
  for (let attempt = 0; attempt < 30 && !id; attempt++) {
    const tag = encodeURIComponent(`harness:${run.harness}`);
    try {
      const list = await fetch(`${bankUrl}/documents?tags=${tag}&tags_match=all`, {
        headers,
      });
      if (!list.ok) throw new Error(`Could not list E2E documents: ${list.status}`);
      const listed = (await list.json()) as { items?: Array<{ id?: string }> };
      id = listed.items?.find((item) => item.id?.startsWith("conversation:"))?.id;
    } catch (error) {
      lastError = error; // reported below only if every attempt fails
    }
    if (!id) await new Promise((resolve) => setTimeout(resolve, 1_000));
  }
  if (!id) {
    const diagnostics = run.diagnostics.trim() || "no hook diagnostics were written";
    // Distinguish "the bank never got the document" from "we could never ask" — otherwise a
    // server that was unreachable throughout reads as a retention bug in the harness.
    const cause = lastError ? `\nLast error reaching the API: ${String(lastError)}` : "";
    throw new Error(
      `${run.harness} did not retain a conversation document in the Hindsight bank.${cause}\nDiagnostics:\n${diagnostics}`
    );
  }
  const document = await fetch(`${bankUrl}/documents/${encodeURIComponent(id)}`, { headers });
  if (!document.ok) throw new Error(`Could not read retained E2E document: ${document.status}`);
  return document.json();
}

export const e2eEnabled = E2E;
export const e2ePromptMarker = RUN_MARKER;
