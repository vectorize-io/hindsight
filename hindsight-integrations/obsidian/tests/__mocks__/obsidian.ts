/**
 * Minimal mock of the `obsidian` module for unit tests. Vitest aliases
 * "obsidian" → this file (see vitest.config.ts). Only `requestUrl` has real
 * behaviour the tests drive; the rest are harmless stubs so any import resolves.
 */
import { vi } from "vitest";

export interface RequestUrlParam {
  url: string;
  method?: string;
  headers?: Record<string, string>;
  body?: string;
  throw?: boolean;
}

export interface RequestUrlResponse {
  status: number;
  text: string;
  json: unknown;
}

export const requestUrl = vi.fn(
  async (_param: RequestUrlParam): Promise<RequestUrlResponse> => ({
    status: 200,
    text: "{}",
    json: {},
  })
);

export class Notice {
  constructor(_message?: string) {}
}
export class Plugin {}
export class ItemView {}
export class PluginSettingTab {}
export class Setting {}
export class TFile {}
export const MarkdownRenderer = { render: vi.fn() };
export function debounce<T extends unknown[]>(fn: (...args: T) => unknown): (...args: T) => void {
  return (...args: T) => void fn(...args);
}
