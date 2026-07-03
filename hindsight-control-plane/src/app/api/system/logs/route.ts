import { NextRequest, NextResponse } from "next/server";
import { readFileSync, existsSync } from "fs";
import { execSync } from "child_process";

// Determine project root (try to find it from git or use fallback)
function getProjectRoot(): string {
  try {
    const root = execSync("git rev-parse --show-toplevel", { encoding: "utf-8" }).trim();
    return root;
  } catch {
    // Fallback to hardcoded path if git not available
    return "/Users/oliververmeulen/hindsight";
  }
}

const PROJECT_ROOT = getProjectRoot();
const LOGS_DIR = `${PROJECT_ROOT}/logs`;

const LOG_FILES: Record<string, string> = {
  api: `${LOGS_DIR}/hindsight-api.log`,
  "control-plane": `${LOGS_DIR}/hindsight-control-plane.log`,
  "ollama-embeddings": "/tmp/ollama-embeddings.log",
  "ollama-llm": "/tmp/ollama-llm.log",
  "worker-0": `${LOGS_DIR}/worker-0.log`,
  "worker-1": `${LOGS_DIR}/worker-1.log`,
  "worker-2": `${LOGS_DIR}/worker-2.log`,
  "worker-3": `${LOGS_DIR}/worker-3.log`,
  "worker-4": `${LOGS_DIR}/worker-4.log`,
};

function tailLog(filePath: string, lines: number = 100): string[] {
  if (!existsSync(filePath)) {
    return [];
  }

  try {
    const output = execSync(`tail -n ${lines} "${filePath}"`, { encoding: "utf-8" });
    return output.trim().split("\n").filter(Boolean);
  } catch (error) {
    return [];
  }
}

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const service = searchParams.get("service");
  const lines = parseInt(searchParams.get("lines") || "100");
  const follow = searchParams.get("follow") === "true";

  if (!service) {
    // Return available log files
    const available = Object.keys(LOG_FILES).filter((key) => existsSync(LOG_FILES[key]));
    return NextResponse.json({ available });
  }

  const logFile = LOG_FILES[service];
  if (!logFile || !existsSync(logFile)) {
    return NextResponse.json(
      { error: `Log file not found for service: ${service}` },
      { status: 404 }
    );
  }

  if (follow) {
    // For streaming logs, use Server-Sent Events
    const encoder = new TextEncoder();
    const stream = new ReadableStream({
      async start(controller) {
        try {
          // Send initial logs
          const initialLogs = tailLog(logFile, lines);
          controller.enqueue(encoder.encode(`data: ${JSON.stringify({ lines: initialLogs })}\n\n`));

          // Note: Real-time streaming would require a tail -f implementation
          // For now, we'll close after sending initial logs
          controller.enqueue(encoder.encode("data: [DONE]\n\n"));
          controller.close();
        } catch (error) {
          controller.error(error);
        }
      },
    });

    return new Response(stream, {
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        Connection: "keep-alive",
      },
    });
  }

  // Return static log snapshot
  const logLines = tailLog(logFile, lines);
  return NextResponse.json({
    service,
    file: logFile,
    lines: logLines,
    count: logLines.length,
  });
}
