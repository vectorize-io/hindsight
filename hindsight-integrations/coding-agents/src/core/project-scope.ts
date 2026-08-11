import type { Config } from "./config";
import { buildRetainStamp, type RetainStamp, type RetainStampContext } from "./retain-stamp";

export interface ProjectScope {
  projectTag: string;
  globalTags: string[];
}

export function resolveProjectScope(
  cfg: Config,
  cwd: string,
  harness: string,
  bankId: string
): ProjectScope | undefined {
  if (cfg.projectScope !== "tags") return undefined;
  const projectTag = buildRetainStamp(
    { retainTags: [cfg.projectTagTemplate] },
    { directory: cwd, harness, bankId }
  ).tags[0];
  return projectTag ? { projectTag, globalTags: cfg.globalTags } : undefined;
}

export function scopeTagGroups(scope: ProjectScope): unknown[] {
  const leaves = [scope.projectTag, ...scope.globalTags].map((tag) => ({
    tags: [tag],
    match: "any_strict",
  }));
  return leaves.length === 1 ? leaves : [{ or: leaves }];
}

export function buildScopedRetainStamp(cfg: Config, ctx: RetainStampContext): RetainStamp {
  const stamp = buildRetainStamp(cfg, ctx);
  const scope = resolveProjectScope(cfg, ctx.directory, ctx.harness, ctx.bankId);
  return scope ? { ...stamp, tags: [...new Set([...stamp.tags, scope.projectTag])] } : stamp;
}
