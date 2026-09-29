#!/usr/bin/env node
/**
 * Hold the codebase-survey lease for the lifetime of the survey agent. Spawned DETACHED by
 * core/survey.ts, which hands over the `SurveySupervisorSpec` in the `SURVEY_SPEC_ENV` variable
 * (not argv — see there), because the hook that won the lease exits immediately and the agent
 * itself cannot heartbeat. See core/survey-lease.ts.
 */
import { SURVEY_SPEC_ENV, superviseSurvey, type SurveySupervisorSpec } from "./core/survey-lease";

const spec = JSON.parse(process.env[SURVEY_SPEC_ENV] ?? "") as SurveySupervisorSpec;
delete process.env[SURVEY_SPEC_ENV]; // the agent (and its MCP server) inherit this environment
const run = superviseSurvey(spec);
for (const signal of ["SIGTERM", "SIGINT", "SIGHUP"] as const) process.once(signal, run.stop);
void run.done; // the agent and the heartbeat keep this process alive until it resolves
