/**
 * "Hindsight" in the brand gradient (#0074d9 → #009296) — the same colors the API server's
 * banner uses (hindsight_api/banner.py gradient_text). ANSI foreground-only.
 *
 * `color: false` gives the plain word. Hook output is not a terminal: a host that does not paint
 * color codes shows them as text (#6502), so only hosts known to paint them ask for color
 * (`HookSpec.color` / `SessionStartHookSpec.color`).
 */
export function brandWord(color = true): string {
  if (!color) return "Hindsight";
  const start = [0, 116, 217];
  const end = [0, 146, 150];
  const word = "Hindsight";
  let out = "";
  for (let i = 0; i < word.length; i++) {
    const t = i / (word.length - 1);
    const [r, g, b] = start.map((s, k) => Math.round(s + (end[k] - s) * t));
    out += `\x1b[38;2;${r};${g};${b}m${word[i]}`;
  }
  return `${out}\x1b[0m`;
}
