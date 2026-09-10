#!/usr/bin/env node
/**
 * Hold the codebase-survey lease for the lifetime of the survey agent. Spawned DETACHED by
 * core/survey.ts with one argument — the JSON `SurveySupervisorSpec` — because the hook that won
 * the lease exits immediately and the agent itself cannot heartbeat. See core/survey-lease.ts.
 */
import { superviseSurvey, type SurveySupervisorSpec } from "./core/survey-lease";

const run = superviseSurvey(JSON.parse(process.argv[2]) as SurveySupervisorSpec);
for (const signal of ["SIGTERM", "SIGINT", "SIGHUP"] as const) process.once(signal, run.stop);
void run.done; // the agent and the heartbeat keep this process alive until it resolves
