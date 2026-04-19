/**
 * Vitest compatibility shim for running tests under Deno.
 *
 * Redirected from 'vitest' via deno.json import map.
 * Implements vi.fn() / vi.spyOn() / vi.mocked() using the @std/expect
 * MockCall interface so that @std/expect matchers (toHaveBeenCalledWith, etc.) work.
 */

import { beforeAll, beforeEach, afterAll, afterEach, describe, it } from "jsr:@std/testing/bdd";
import { expect } from "jsr:@std/expect";

// @std/expect recognises mock functions via this well-known symbol (@std/expect 1.0.18+)
const MOCK_SYMBOL = Symbol.for("@MOCK");

type MockCall = {
  args: unknown[];
  returned?: unknown;
  thrown?: unknown;
  timestamp: number;
  returns: boolean;
  throws: boolean;
};

function createMock(impl?: (...args: unknown[]) => unknown) {
  let currentImpl = impl;
  const calls: MockCall[] = [];
  const mockInfo = { calls };

  const mockFn = function (this: unknown, ...args: unknown[]) {
    const call: MockCall = {
      args,
      timestamp: Date.now(),
      returns: false,
      throws: false,
    };
    calls.push(call);
    try {
      const result = currentImpl ? currentImpl.apply(this, args) : undefined;
      call.returned = result;
      call.returns = true;
      return result;
    } catch (err) {
      call.thrown = err;
      call.throws = true;
      throw err;
    }
  };

  // @std/expect mock interface — required for toHaveBeenCalledWith etc.
  (mockFn as any)[MOCK_SYMBOL] = mockInfo;

  // vitest-style chainable methods
  (mockFn as any).mockReturnValue = (val: unknown) => {
    currentImpl = () => val;
    return mockFn;
  };
  (mockFn as any).mockResolvedValue = (val: unknown) => {
    currentImpl = () => Promise.resolve(val);
    return mockFn;
  };
  (mockFn as any).mockRejectedValue = (val: unknown) => {
    currentImpl = () => Promise.reject(val);
    return mockFn;
  };
  (mockFn as any).mockImplementation = (fn: (...args: unknown[]) => unknown) => {
    currentImpl = fn;
    return mockFn;
  };
  (mockFn as any).mockReset = () => {
    calls.length = 0;
    currentImpl = undefined;
    return mockFn;
  };
  (mockFn as any).mockClear = () => {
    calls.length = 0;
    return mockFn;
  };
  (mockFn as any).mockRestore = () => {};

  return mockFn;
}

export const vi = {
  fn: (impl?: (...args: unknown[]) => unknown) => createMock(impl),

  // vi.mocked() is a TypeScript type cast — just return the same value at runtime
  mocked: <T>(fn: T): T => fn,

  spyOn: <T extends Record<string, unknown>>(obj: T, method: keyof T) => {
    const original = obj[method];
    const mock = createMock(
      typeof original === "function" ? (original as (...args: unknown[]) => unknown) : undefined
    );
    const restore = () => {
      obj[method] = original;
    };
    (mock as any).mockRestore = restore;
    obj[method] = mock as unknown as T[keyof T];
    return mock;
  },
};

export { describe, it, it as test, beforeAll, beforeEach, afterAll, afterEach, expect };
