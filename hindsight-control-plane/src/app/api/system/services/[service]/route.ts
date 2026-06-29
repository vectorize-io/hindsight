import { NextRequest, NextResponse } from "next/server";
import { spawn, execSync } from "child_process";
import { existsSync } from "fs";
import path from "path";

const SCRIPTS_DIR = "/Users/oliververmeulen/hindsight/scripts/dev";

interface ServiceConfig {
  name: string;
  startScript?: string;
  stopScript?: string;
  restartScript?: string;
  startCommand?: string;
  stopCommand?: string;
}

const SERVICE_CONFIGS: Record<string, ServiceConfig> = {
  "hindsight-api": {
    name: "Hindsight API",
    startScript: path.join(SCRIPTS_DIR, "start-api.sh"),
    stopCommand: `pkill -f "hindsight-api"`,
  },
  "ollama-embeddings": {
    name: "Ollama Embeddings",
    startCommand: `OLLAMA_HOST=127.0.0.1:11434 /opt/homebrew/opt/ollama/bin/ollama serve > /tmp/ollama-embeddings.log 2>&1 &`,
    stopCommand: `pkill -f "OLLAMA_HOST=127.0.0.1:11434"`,
  },
  "ollama-llm": {
    name: "Ollama LLM",
    startCommand: `OLLAMA_HOST=127.0.0.1:11435 OLLAMA_MODELS=/Volumes/Mac/Users/oliververmeulen/.ollama/models /opt/homebrew/opt/ollama/bin/ollama serve > /tmp/ollama-llm.log 2>&1 &`,
    stopCommand: `pkill -f "OLLAMA_HOST=127.0.0.1:11435"`,
  },
  workers: {
    name: "Workers",
    startScript: path.join(SCRIPTS_DIR, "scale-workers.sh"),
    stopScript: path.join(SCRIPTS_DIR, "scale-workers.sh"),
  },
};

function executeCommand(command: string): { success: boolean; output: string; error?: string } {
  try {
    const output = execSync(command, {
      encoding: "utf-8",
      stdio: ["pipe", "pipe", "pipe"],
    });
    return { success: true, output };
  } catch (error: any) {
    return {
      success: false,
      output: error.stdout || "",
      error: error.stderr || error.message,
    };
  }
}

export async function POST(
  request: NextRequest,
  context: { params: Promise<{ service: string }> }
) {
  const { service } = await context.params;
  const body = await request.json();
  const { action, count } = body;

  const config = SERVICE_CONFIGS[service];
  if (!config) {
    return NextResponse.json({ error: `Unknown service: ${service}` }, { status: 404 });
  }

  let result: { success: boolean; output: string; error?: string };

  switch (action) {
    case "start":
      if (config.startScript && existsSync(config.startScript)) {
        result = executeCommand(`bash ${config.startScript}`);
      } else if (config.startCommand) {
        result = executeCommand(config.startCommand);
      } else {
        return NextResponse.json(
          { error: `No start method configured for ${config.name}` },
          { status: 400 }
        );
      }
      break;

    case "stop":
      if (config.stopScript && existsSync(config.stopScript)) {
        // For workers, use stop command
        if (service === "workers") {
          result = executeCommand(`bash ${config.stopScript} stop`);
        } else {
          result = executeCommand(`bash ${config.stopScript}`);
        }
      } else if (config.stopCommand) {
        result = executeCommand(config.stopCommand);
      } else {
        return NextResponse.json(
          { error: `No stop method configured for ${config.name}` },
          { status: 400 }
        );
      }
      break;

    case "restart":
      if (config.restartScript && existsSync(config.restartScript)) {
        result = executeCommand(`bash ${config.restartScript}`);
      } else {
        // Stop then start
        if (config.stopCommand) {
          executeCommand(config.stopCommand);
        }
        await new Promise((resolve) => setTimeout(resolve, 2000));

        if (config.startScript && existsSync(config.startScript)) {
          result = executeCommand(`bash ${config.startScript}`);
        } else if (config.startCommand) {
          result = executeCommand(config.startCommand);
        } else {
          return NextResponse.json(
            { error: `No restart method configured for ${config.name}` },
            { status: 400 }
          );
        }
      }
      break;

    case "scale":
      // Worker scaling
      if (service === "workers") {
        if (count === undefined || count < 0 || count > 10) {
          return NextResponse.json(
            { error: "Worker count must be between 0 and 10" },
            { status: 400 }
          );
        }

        const scaleScript = path.join(SCRIPTS_DIR, "scale-workers.sh");
        if (!existsSync(scaleScript)) {
          return NextResponse.json({ error: "Worker scaling script not found" }, { status: 500 });
        }

        result = executeCommand(`bash ${scaleScript} ${count}`);
      } else {
        return NextResponse.json(
          { error: `Scaling not supported for ${config.name}` },
          { status: 400 }
        );
      }
      break;

    default:
      return NextResponse.json({ error: `Unknown action: ${action}` }, { status: 400 });
  }

  return NextResponse.json({
    service: config.name,
    action,
    success: result.success,
    output: result.output,
    error: result.error,
  });
}
