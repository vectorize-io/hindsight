#!/usr/bin/env node
/**
 * Cursor CLI `sessionStart` hook. It runs the shared session-start lifecycle: configures the
 * bank's five knowledge pages, starts cold-repo seeding/deepening, and supplies the page roster.
 * Cursor's hook protocol uses `additional_context`, unlike the Claude-compatible hook envelope.
 */
import { runHarnessSessionStart } from "./harness/hook-lifecycle";

void runHarnessSessionStart("cursor-cli");
