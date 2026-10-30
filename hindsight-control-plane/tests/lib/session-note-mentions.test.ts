import { describe, expect, it } from "vitest";
import type { SessionEvent } from "@/lib/api";
import { noteMentions } from "@/lib/session-timeline";

const msg = (seq: number, role: string, text: string): SessionEvent => ({
  seq,
  event_id: String(seq),
  source: "client",
  kind: "message",
  created_at: "2026-10-21T10:00:00Z",
  payload: { role, text },
});

describe("noteMentions", () => {
  it("reads which notes the agent said it used or did not need, by number", () => {
    const out = noteMentions([
      msg(1, "assistant", "> 🧠 **Hindsight note (live session #3)** — card rule; not needed here."),
      msg(2, "user", "> 🧠 **From Hindsight memory (live session #4)** — quoted by the user"),
      msg(5, "assistant", "> 🧠 **From Hindsight memory (live session #3)** — used after all."),
    ]);
    // A note named both ways counts as used; a user quoting the line counts for nothing.
    expect([...out.entries()]).toEqual([[3, "used"]]);
  });

  it("reads the thought lines the coding-agents plugin asks for", () => {
    const out = noteMentions([
      msg(1, "assistant", "> 🧠 **Thanks to Hindsight thought #7**, I kept K at 1024."),
      msg(3, "assistant", "> 🧠 *Hindsight thought #9, already done* — K was 1024 already."),
      msg(4, "assistant", "> 🧠 Hindsight thought #10 — the cache rule; no change: no cache here."),
      msg(2, "assistant", "> 🧠 *Hindsight thought #8, set aside* — email rule; does not apply: no email."),
    ]);
    expect([...out.entries()].sort((a, b) => a[0] - b[0])).toEqual([
      [7, "used"],
      [8, "unneeded"],
      [9, "unneeded"],
      [10, "unneeded"],
    ]);
  });
});
