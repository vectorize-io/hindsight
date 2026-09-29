/** Shared visible-attribution directive — the PROVEN, battle-tested wording ported verbatim from the
 *  v1 Claude Code plugin's `recallPromptPreamble` (which reliably makes the agent surface the header).
 *  Reused by both the reflect injection (inject.ts) and per-turn recall formatting (recall.ts). */
export const ATTRIBUTION_PREAMBLE = `VISIBLE ATTRIBUTION — SHOW HINDSIGHT WORKING:
The goal of this attribution header is for the user to SEE Hindsight contributing value in their sessions. Be generous about when to emit it. Whenever recalled memories are RELEVANT to the answer you're about to give — whether they directly drove your reasoning, supplied background context, reinforced a conclusion, or saved you from having to ask a question — surface them with this exact markdown header at the top of the relevant section:

> 🧠 **Using Hindsight Memories** — {summary naming the specific facts you're drawing on}

Render it exactly as a markdown blockquote (leading "> ") with the brain emoji, bold label, em dash separator, and tight summary. The blockquote + bold creates a clear visual marker in any markdown chat surface.

Rules:
- WHEN IN DOUBT, EMIT. Over-attribution is far better than invisible value. If a memory is even loosely relevant to the topic, surface it. At the beginning of a session especially, it is important to ALWAYS show that you are leveraging Hindsight so the user knows their memories are being taken into consideration.
- Skip the header only when every recalled memory is clearly unrelated to what the user asked (e.g. the user asked about today's weather and the memories are all about a codebase).
- Name the specific facts in the summary — the actual rule, value, decision, or convention you're drawing on, not a meta-statement like "using memory."
- When multiple memories contributed, name the 2–3 most load-bearing ones rather than listing them all.
- One header per response is enough; place it at the top of the section that benefits from the memories, not buried at the end.
- If a memory is RELEVANT but turns out to be WRONG or STALE, still surface it — say so explicitly ("memory said X, but the code now shows Y") so the user sees the system course-correcting in the open.
- Do NOT emit ANSI escape sequences, raw HTML, or any other color syntax — the chat surface renders plain markdown, so anything else shows as literal text.`;
