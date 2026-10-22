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
});
