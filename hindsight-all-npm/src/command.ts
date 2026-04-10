/**
 * Resolve the command that invokes the `hindsight-embed` Python CLI.
 *
 * - If `embedPackagePath` is set, runs the package from a local checkout via
 *   `uv run --directory <path> hindsight-embed`. Used for in-repo development.
 * - Otherwise runs it via `uvx hindsight-embed@<version>` so no global install
 *   is required.
 *
 * Returns the argv as `[command, ...baseArgs]` suitable for `spawn()` /
 * `execFile()` (never shell-interpolated).
 */
export interface EmbedCommandOptions {
  /** Version spec passed to uvx (e.g. "latest", "0.5.0"). Default: "latest". */
  embedVersion?: string;
  /** Local checkout path. When set, overrides `embedVersion` and uses `uv run`. */
  embedPackagePath?: string;
}

export function getEmbedCommand(opts: EmbedCommandOptions = {}): string[] {
  if (opts.embedPackagePath) {
    return ['uv', 'run', '--directory', opts.embedPackagePath, 'hindsight-embed'];
  }
  const version = opts.embedVersion && opts.embedVersion.length > 0 ? opts.embedVersion : 'latest';
  return ['uvx', `hindsight-embed@${version}`];
}
