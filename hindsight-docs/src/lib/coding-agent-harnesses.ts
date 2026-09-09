/**
 * The coding agents covered by the Coding Agents plugin — the docs site's SINGLE source of truth.
 *
 * Two surfaces draw this row of logos: the integrations gallery card
 * (src/pages/integrations/index.tsx) and the coding-agent group preview in both sidebars
 * (src/lib/integration-groups.ts). They used to keep separate hand-maintained copies, which drifted
 * the moment a harness was added — `dcode`, `opencode2`, `pi` and `prime-agent` were each missing
 * from one list or the other. Add a harness HERE and both surfaces pick it up.
 *
 * `id` is the harness id the plugin itself emits (see that package's `src/harness/registry.ts`);
 * `src/docs-harness-roster.test.ts` over there fails if this list and that one drift apart, so a new
 * harness cannot ship without its logo.
 *
 * Icons live in static/img/harness/, named by the same id — except opencode 2, which ships as its
 * own binary and reports its own harness id but is the same product, so it deliberately shares
 * v1's brand mark rather than inventing a second one.
 *
 * Deliberately free of the `@site/` alias and of any JSON import: integration-groups.ts pulls this
 * in from sidebars-integrations.ts, which is evaluated at config load time where the alias does not
 * exist yet.
 */
export interface CodingAgentHarness {
  /** Canonical harness id, e.g. "claude-code". */
  id: string;
  /** Display name, a proper noun — deliberately not translated. */
  label: string;
  /** File name under static/img/harness/. */
  file: string;
}

/** Presentation order, matching the per-agent sections of the plugin's README. */
export const CODING_AGENT_HARNESSES: CodingAgentHarness[] = [
  {id: 'claude-code', label: 'Claude Code', file: 'claude-code.png'},
  {id: 'codex', label: 'Codex CLI', file: 'codex.svg'},
  {id: 'dcode', label: 'DeepAgents Dcode', file: 'dcode.svg'},
  {id: 'opencode', label: 'opencode', file: 'opencode.png'},
  {id: 'opencode2', label: 'opencode 2', file: 'opencode.png'},
  {id: 'kilo', label: 'Kilo CLI', file: 'kilo.svg'},
  {id: 'cursor-cli', label: 'Cursor CLI', file: 'cursor-cli.svg'},
  {id: 'copilot-cli', label: 'GitHub Copilot CLI', file: 'copilot-cli.svg'},
  {id: 'grok-build', label: 'Grok Build', file: 'grok-build.svg'},
  {id: 'qwen-code', label: 'Qwen Code', file: 'qwen-code.svg'},
  {id: 'factory-droid', label: 'Factory Droid', file: 'factory-droid.svg'},
  {id: 'zcode', label: 'ZCode', file: 'zcode.svg'},
  {id: 'antigravity-cli', label: 'Antigravity CLI', file: 'antigravity-cli.png'},
  {id: 'devin-cli', label: 'Devin CLI', file: 'devin-cli.svg'},
  {id: 'cline-cli', label: 'Cline CLI', file: 'cline-cli.svg'},
  {id: 'pi', label: 'pi', file: 'pi.svg'},
  {id: 'prime-agent', label: 'Prime Agent', file: 'prime-agent.svg'},
  {id: 'dsh', label: 'DeepSeek Harness', file: 'dsh.svg'},
];

/** Public path of a harness icon, for surfaces that want the URL rather than the file name. */
export const harnessIconPath = (harness: CodingAgentHarness): string =>
  `/img/harness/${harness.file}`;
