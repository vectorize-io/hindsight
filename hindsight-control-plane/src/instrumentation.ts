/**
 * Next.js instrumentation file - runs exactly once at server startup.
 * https://nextjs.org/docs/app/building-your-application/optimizing/instrumentation
 */
export async function register() {
  const pidFile = process.env.HINDSIGHT_EMBED_UI_PID_FILE;
  if (pidFile) {
    const { mkdirSync, readFileSync, writeFileSync } = await import("node:fs");
    const { dirname } = await import("node:path");
    const { execFileSync } = await import("node:child_process");
    let birthMarker: string | null = null;
    try {
      if (process.platform === "linux") {
        const stat = readFileSync(`/proc/${process.pid}/stat`, "utf8");
        const fields = stat.slice(stat.lastIndexOf(")") + 2).split(/\s+/);
        birthMarker = fields.length > 19 ? `linux:${fields[19]}` : null;
      } else {
        const args =
          process.platform === "win32"
            ? [
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                `(Get-CimInstance Win32_Process -Filter 'ProcessId = ${process.pid}').CreationDate`,
              ]
            : ["-o", "lstart=", "-p", String(process.pid)];
        const executable = process.platform === "win32" ? "powershell.exe" : "ps";
        birthMarker = execFileSync(executable, args, { encoding: "utf8", windowsHide: true })
          .trim()
          .replace(/\s+/g, " ");
      }
    } catch {
      console.warn(
        "[Control Plane] Could not determine process creation time; ownership receipt not written"
      );
    }
    mkdirSync(dirname(pidFile), { recursive: true });
    if (birthMarker) {
      writeFileSync(
        pidFile,
        JSON.stringify({ version: 1, pid: process.pid, birth_marker: birthMarker })
      );
    }
  }

  const dataplaneUrl = process.env.HINDSIGHT_CP_DATAPLANE_API_URL || "http://localhost:8888";
  const apiKey = process.env.HINDSIGHT_CP_DATAPLANE_API_KEY || "";

  console.log(`[Control Plane] Connecting to dataplane at: ${dataplaneUrl}`);
  if (apiKey) {
    console.log("[Control Plane] Using API key authentication");
  } else {
    console.log("[Control Plane] No API key configured (public access)");
  }
}
