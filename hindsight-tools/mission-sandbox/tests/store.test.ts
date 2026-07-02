import { promises as fs } from "node:fs";
import os from "node:os";
import path from "node:path";

import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { Project } from "../src/core/store.js";
import type { ProjectMeta } from "../src/core/types.js";

let dir: string;

beforeEach(async () => {
  dir = await fs.mkdtemp(path.join(os.tmpdir(), "mission-sandbox-"));
});

afterEach(async () => {
  await fs.rm(dir, { recursive: true, force: true });
});

describe("Project store", () => {
  it("creates, persists and reloads a project", async () => {
    const projDir = path.join(dir, "proj");
    const proj = await Project.create(projDir, {
      name: "conv26",
      documents: "/docs",
      apiUrl: "http://localhost:8888",
      apiKey: "secret-token",
    });
    proj.retain.mission = "Extract durable life facts.";
    proj.retain.feedback.push("capture dates");
    await proj.save();

    const reloaded = await Project.load(projDir);
    expect(reloaded.name).toBe("conv26");
    expect(reloaded.apiUrl).toBe("http://localhost:8888");
    expect(reloaded.apiKey).toBe("secret-token");
    expect(reloaded.retain.mission).toBe("Extract durable life facts.");
    expect(reloaded.retain.feedback).toEqual(["capture dates"]);
    expect(reloaded.currentVersion).toBeNull();
    expect(reloaded.currentBank()).toBeNull();
  });

  it("allocates sequential versioned banks and tracks the current one", async () => {
    const projDir = path.join(dir, "proj");
    const proj = await Project.create(projDir, {
      name: "conv26",
      documents: "/docs",
      apiUrl: "http://localhost:8888",
    });

    const v1 = proj.addVersion({ retainMission: "m1", observeMission: null });
    expect(v1.n).toBe(1);
    expect(v1.bank).toBe("conv26-v1");
    expect(proj.currentBank()).toBe("conv26-v1");

    const v2 = proj.addVersion({ retainMission: "m2", observeMission: "o2" });
    expect(v2.n).toBe(2);
    expect(v2.bank).toBe("conv26-v2");
    await proj.save();

    const reloaded = await Project.load(projDir);
    expect(reloaded.versions).toHaveLength(2);
    expect(reloaded.currentVersion).toBe(2);
    expect(reloaded.currentBank()).toBe("conv26-v2");
    expect(reloaded.versions[0]).toMatchObject({ n: 1, bank: "conv26-v1", retainMission: "m1" });
  });

  it("persists project.json as camelCase ProjectMeta", async () => {
    const projDir = path.join(dir, "proj");
    const proj = await Project.create(projDir, {
      name: "conv26",
      documents: "/docs",
      apiUrl: "http://localhost:8888",
    });
    proj.observe.mission = "Aggregate per-person profiles.";
    proj.addVersion({ retainMission: "m1", observeMission: "Aggregate per-person profiles." });
    await proj.save();

    const raw = JSON.parse(
      await fs.readFile(path.join(projDir, "project.json"), "utf8")
    ) as ProjectMeta;
    expect(raw.name).toBe("conv26");
    expect(raw.documents).toBe("/docs");
    expect(raw.observe.mission).toBe("Aggregate per-person profiles.");
    expect(raw.versions[0]).toMatchObject({ n: 1, bank: "conv26-v1" });
    expect(raw.currentVersion).toBe(1);
  });
});
