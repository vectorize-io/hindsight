import { basename } from "node:path";
import { projectNameOf } from "./bank";

export interface RetainAttributionConfig {
  retainTags?: string[];
  retainMetadata?: Record<string, string>;
}

export interface RetainAttribution {
  tags: string[];
  metadata: Record<string, string>;
}

const PLACEHOLDER = /\{([a-zA-Z]+)\}/g;

/** Resolve user-configured provenance once, then merge it into every document written for a repo. */
export function resolveRetainAttribution(
  config: RetainAttributionConfig,
  directory: string,
  harness: string,
  bankId: string
): RetainAttribution {
  const resolvers: Record<string, () => string> = {
    bankId: () => bankId,
    cwd: () => directory,
    gitProject: () => projectNameOf(directory),
    harness: () => harness,
    project: () => (directory ? basename(directory) : "unknown"),
  };
  const warned = new Set<string>();
  const render = (template: string): string | undefined => {
    let valid = true;
    const rendered = template.replace(PLACEHOLDER, (_, name: string) => {
      const resolver = resolvers[name];
      if (!resolver) {
        valid = false;
        if (!warned.has(name)) {
          warned.add(name);
          console.error(
            `hindsight: unknown retain template placeholder "{${name}}" — valid: ` +
              Object.keys(resolvers)
                .sort()
                .map((key) => `{${key}}`)
                .join(", ")
          );
        }
        return "";
      }
      return resolver();
    });
    return valid ? rendered : undefined;
  };

  const tags = [
    ...new Set(
      (config.retainTags ?? [])
        .map(render)
        .filter((tag): tag is string => tag !== undefined)
        .map((tag) => tag.trim())
    ),
  ].filter((tag) => tag && !tag.endsWith(":"));
  const metadata = Object.fromEntries(
    Object.entries(config.retainMetadata ?? {})
      .map(([key, value]) => [key, render(value)?.trim()])
      .filter((entry): entry is [string, string] => entry[1] !== undefined)
      .filter(([, value]) => value !== "")
  );
  return { tags, metadata };
}

export function mergeTags(base: string[], extra: string[] = []): string[] {
  return [...new Set([...base, ...extra])];
}

export function mergeMetadata(
  base: Record<string, string>,
  extra: Record<string, string> = {}
): Record<string, string> {
  return { ...extra, ...base };
}
