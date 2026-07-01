import { NextRequest, NextResponse } from "next/server";
import { execSync } from "child_process";
import { readFileSync, existsSync } from "fs";

interface ServiceStatus {
  name: string;
  port: number;
  status: "running" | "stopped" | "error";
  pid?: number;
  uptime?: string;
  health?: string;
  cpu?: number;
  memory?: number;
}

function checkPort(port: number): { running: boolean; pid?: number } {
  try {
    const output = execSync(`lsof -ti:${port}`, { encoding: "utf-8" }).trim();
    const pid = parseInt(output.split("\n")[0]);
    return { running: !!output, pid };
  } catch {
    return { running: false };
  }
}

function getProcessStats(pid: number): { cpu: number; memory: number; uptime: string } | null {
  try {
    // Get process stats using ps
    const output = execSync(`ps -p ${pid} -o %cpu,%mem,etime`, { encoding: "utf-8" });
    const lines = output.trim().split("\n");
    if (lines.length < 2) return null;

    const [cpu, mem, etime] = lines[1].trim().split(/\s+/);
    return {
      cpu: parseFloat(cpu),
      memory: parseFloat(mem),
      uptime: etime,
    };
  } catch {
    return null;
  }
}

async function checkApiHealth(): Promise<{ status: string; database: string } | null> {
  try {
    const response = await fetch("http://localhost:8888/health", {
      signal: AbortSignal.timeout(2000),
    });
    return await response.json();
  } catch {
    return null;
  }
}

async function checkOllamaHealth(port: number): Promise<boolean> {
  try {
    const response = await fetch(`http://localhost:${port}/api/tags`, {
      signal: AbortSignal.timeout(2000),
    });
    return response.ok;
  } catch {
    return false;
  }
}

async function checkGenericHealth(url: string): Promise<{ ok: boolean; body?: string }> {
  try {
    const response = await fetch(url, { signal: AbortSignal.timeout(2000) });
    const text = await response.text().catch(() => "");
    return { ok: response.ok, body: text };
  } catch {
    return { ok: false };
  }
}

function getWorkerCount(): number {
  const stateFile = "/tmp/hindsight-workers.state";
  if (!existsSync(stateFile)) return 0;

  try {
    const content = readFileSync(stateFile, "utf-8");
    const workers = content
      .trim()
      .split("\n")
      .filter((line) => line.includes("worker_"));
    return workers.length;
  } catch {
    return 0;
  }
}

export async function GET(request: NextRequest) {
  const services: ServiceStatus[] = [];

  // Check Hindsight API
  const apiPort = checkPort(8888);
  const apiHealth = apiPort.running ? await checkApiHealth() : null;
  const apiStats = apiPort.pid ? getProcessStats(apiPort.pid) : null;

  services.push({
    name: "Hindsight API",
    port: 8888,
    status: apiPort.running ? "running" : "stopped",
    pid: apiPort.pid,
    health: apiHealth?.status || "unknown",
    uptime: apiStats?.uptime,
    cpu: apiStats?.cpu,
    memory: apiStats?.memory,
  });

  // Check Control Plane (always running if this API responds)
  const cpPort = checkPort(
    process.env.CONTROL_PLANE_PORT ? parseInt(process.env.CONTROL_PLANE_PORT) : 9998
  );
  const cpStats = cpPort.pid ? getProcessStats(cpPort.pid) : null;

  services.push({
    name: "Control Plane",
    port: process.env.CONTROL_PLANE_PORT ? parseInt(process.env.CONTROL_PLANE_PORT) : 9998,
    status: "running",
    pid: cpPort.pid,
    health: "healthy",
    uptime: cpStats?.uptime,
    cpu: cpStats?.cpu,
    memory: cpStats?.memory,
  });

  // Check Ollama Embeddings Lane
  const ollamaEmbPort = checkPort(11434);
  const ollamaEmbHealth = ollamaEmbPort.running ? await checkOllamaHealth(11434) : false;
  const ollamaEmbStats = ollamaEmbPort.pid ? getProcessStats(ollamaEmbPort.pid) : null;

  services.push({
    name: "Ollama Embeddings",
    port: 11434,
    status: ollamaEmbPort.running ? "running" : "stopped",
    pid: ollamaEmbPort.pid,
    health: ollamaEmbHealth ? "healthy" : "unknown",
    uptime: ollamaEmbStats?.uptime,
    cpu: ollamaEmbStats?.cpu,
    memory: ollamaEmbStats?.memory,
  });

  // Check Ollama LLM Lane
  const ollamaLlmPort = checkPort(11435);
  const ollamaLlmHealth = ollamaLlmPort.running ? await checkOllamaHealth(11435) : false;
  const ollamaLlmStats = ollamaLlmPort.pid ? getProcessStats(ollamaLlmPort.pid) : null;

  services.push({
    name: "Ollama LLM",
    port: 11435,
    status: ollamaLlmPort.running ? "running" : "stopped",
    pid: ollamaLlmPort.pid,
    health: ollamaLlmHealth ? "healthy" : "unknown",
    uptime: ollamaLlmStats?.uptime,
    cpu: ollamaLlmStats?.cpu,
    memory: ollamaLlmStats?.memory,
  });

  // Check PostgreSQL
  const pgPort = checkPort(5433);
  const pgStats = pgPort.pid ? getProcessStats(pgPort.pid) : null;

  services.push({
    name: "PostgreSQL",
    port: 5433,
    status: pgPort.running ? "running" : "stopped",
    pid: pgPort.pid,
    health: apiHealth?.database === "connected" ? "connected" : "unknown",
    uptime: pgStats?.uptime,
    cpu: pgStats?.cpu,
    memory: pgStats?.memory,
  });

  // Check Workers
  const workerCount = getWorkerCount();
  services.push({
    name: "Workers",
    port: 9001, // First worker port
    status: workerCount > 0 ? "running" : "stopped",
    health: `${workerCount} active`,
    uptime: undefined,
    cpu: undefined,
    memory: undefined,
  });

  // Check Central API (port 8000)
  const centralApiPort = checkPort(8000);
  const centralApiHealth = await checkGenericHealth("http://localhost:8000/health");
  const centralApiStats = centralApiPort.pid ? getProcessStats(centralApiPort.pid) : null;
  services.push({
    name: "Central API",
    port: 8000,
    status: centralApiPort.running ? "running" : "stopped",
    pid: centralApiPort.pid,
    health: centralApiHealth.ok ? "healthy" : "unknown",
    uptime: centralApiStats?.uptime,
    cpu: centralApiStats?.cpu,
    memory: centralApiStats?.memory,
  });

  // Check Memlord (port 8005)
  const memlordPort = checkPort(8005);
  const memlordHealth = await checkGenericHealth("http://localhost:8005/health");
  const memlordStats = memlordPort.pid ? getProcessStats(memlordPort.pid) : null;
  services.push({
    name: "Memlord",
    port: 8005,
    status: memlordPort.running ? "running" : "stopped",
    pid: memlordPort.pid,
    health: memlordHealth.ok ? "healthy" : "unknown",
    uptime: memlordStats?.uptime,
    cpu: memlordStats?.cpu,
    memory: memlordStats?.memory,
  });

  // Check Cockpit (port 8999) — we are Cockpit, so report our own status
  const cockpitPort = checkPort(8999);
  const cockpitStats = cockpitPort.pid ? getProcessStats(cockpitPort.pid) : null;
  services.push({
    name: "Cockpit",
    port: 8999,
    status: cockpitPort.running ? "running" : "stopped",
    pid: cockpitPort.pid,
    health: "healthy",
    uptime: cockpitStats?.uptime,
    cpu: cockpitStats?.cpu,
    memory: cockpitStats?.memory,
  });

  // Check LM Studio (port 1234 on 192.168.1.144)
  const lmStudioHealth = await checkGenericHealth("http://192.168.1.144:1234/health");
  services.push({
    name: "LM Studio",
    port: 1234,
    status: lmStudioHealth.ok ? "running" : "stopped",
    health: lmStudioHealth.ok ? "healthy" : "unknown",
  });

  // Check Grafana (port 3000)
  const grafanaPort = checkPort(3000);
  const grafanaStats = grafanaPort.pid ? getProcessStats(grafanaPort.pid) : null;
  services.push({
    name: "Grafana",
    port: 3000,
    status: grafanaPort.running ? "running" : "stopped",
    pid: grafanaPort.pid,
    health: grafanaPort.running ? "healthy" : "unknown",
    uptime: grafanaStats?.uptime,
    cpu: grafanaStats?.cpu,
    memory: grafanaStats?.memory,
  });

  return NextResponse.json({ services });
}
