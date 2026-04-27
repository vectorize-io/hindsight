/**
 * Preload script for running Jest-style tests under Deno.
 * Injects Jest-compatible globals (describe, test, beforeAll, expect)
 * using Deno's standard library BDD and expect modules.
 *
 * Usage:
 *   deno test --allow-env --allow-net --unstable-sloppy-imports \
 *             --preload=tests/deno_setup.ts tests/
 */

import { beforeAll, beforeEach, afterAll, afterEach, describe, it } from "jsr:@std/testing/bdd";
import { expect } from "jsr:@std/expect";

Object.assign(globalThis, {
  describe,
  test: it,
  it,
  beforeAll,
  beforeEach,
  afterAll,
  afterEach,
  expect,
});
