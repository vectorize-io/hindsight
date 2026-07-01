# Sample briefs

These are reference outputs showing the expected shape of `oma market render` results. Citation URLs are synthetic for documentation; live runs cite real handles, threads, and posts.

---

## Example 1 — pain intent: VS Code performance pain points (last 30 days)

```
oma-market v0.1.0 · intent: pain · synced 2026-05-14 · window: 30d
```

What we learned: Extension host crashes and startup latency dominate the VS Code pain conversation this month, with remote SSH workflows amplifying both issues.

**Extension host instability is the top complaint cluster.** Users on [r/vscode](https://reddit.com/r/vscode/comments/abc123/extension_host_crashes_python/) report the Python and Pylance extensions crashing the host 2-3 times per session on large monorepos. A [HN thread](https://news.ycombinator.com/item?id=38471200) from last week accumulated 140 comments on the same pattern, with many switching to Neovim LSP as a workaround.

**Startup time degrades sharply beyond 40 extensions.** A [GitHub issue](https://github.com/microsoft/vscode/issues/200001) tracking extension activation order has 380 upvotes. Community benchmarks shared on [r/programming](https://reddit.com/r/programming/comments/def456/vscode_startup_benchmark_2026/) show median cold-start at 4.2 s with 45+ extensions, versus 0.9 s for a clean profile — a 4.7x gap that users call "unacceptable for a daily driver."

KEY PATTERNS from the research:

1. Extension host crashes are triggered most often by language server extensions (Python, Go, Rust Analyzer) on files > 5 MB.
2. Remote SSH sessions compound latency: file-save round-trips average 800 ms vs. 120 ms local, per a [community benchmark repo](https://github.com/vscode-bench/remote-latency).
3. Memory footprint complaints cluster around the 16 GB RAM threshold — users with less RAM report swap-induced freezes during indexing.
4. Settings Sync conflicts after team upgrades generate a secondary pain cluster, particularly around keybinding merges.

## SWOT

**Strengths**
- Extension ecosystem depth is unmatched; users tolerate pain because no alternative covers all languages.
- Remote development (SSH, Containers, Codespaces) has no comparable open-source rival.

**Weaknesses**
- Extension host is a single point of failure; one bad extension crashes the whole environment.
- Startup performance scales poorly with extension count, with no built-in lazy-load enforcement.

**Opportunities**
- Profiling tooling for extensions (similar to Chrome DevTools for extensions) would capture this pain directly.
- A curated "Lite Profile" for users with < 20 extensions could win back churned users.

**Threats**
- Zed and Neovim LSP setups are cited as direct alternatives in 23% of sampled complaints.
- JetBrains Fleet's remote story is maturing and targets the SSH pain cluster explicitly.

---
Engine: oma-market v0.1.0 · sources: reddit, hn, github · window: 30d · clusters: 4 · trust >= 0.6

---

## Example 2 — trend intent: RAG framework adoption trend (last 30 days)

```
oma-market v0.1.0 · intent: trend · synced 2026-05-14 · window: 30d
```

What we learned: LlamaIndex and LangChain are losing ground to lighter orchestration layers as teams prioritize latency and cost over feature breadth.

**Minimal orchestration is the dominant emerging pattern.** Posts on [r/LocalLLaMA](https://reddit.com/r/LocalLLaMA/comments/ghi789/ditching_langchain_for_raw_openai_calls/) and [r/MachineLearning](https://reddit.com/r/MachineLearning/comments/jkl012/rag_without_frameworks_benchmark/) show teams stripping LangChain from production and replacing it with direct API calls plus a single vector store client. The cited reason in 70% of cases is debugging difficulty — framework abstractions hide retrieval failures.

**Hybrid retrieval (dense + sparse) is crossing from research to production.** A [HN thread](https://news.ycombinator.com/item?id=39100450) on a Pinecone blog post about BM25 + ADA-002 hybrid search generated 200 comments, most positive. Teams report 12-18% precision gains on domain-specific corpora over pure dense retrieval.

KEY PATTERNS from the research:

1. LlamaIndex GitHub stars grew 8% month-over-month but issue volume grew 22%, suggesting adoption is outpacing maintainer capacity.
2. "RAG evaluation" is a rising search term — teams are investing in offline evals (RAGAS, TruLens) before production rollout.
3. Chunking strategy discussions dominate practitioner forums; semantic chunking libraries saw 3 new releases this month.
4. Cost-driven architectural shifts: teams moving retrieval to local embeddings (nomic-embed, mxbai) to cut OpenAI embedding spend by 60-80%.

## SWOT

**Strengths**
- RAG is now a well-understood pattern; onboarding friction has dropped significantly since 2024.
- Retrieval-augmented generation outperforms fine-tuning for knowledge-update use cases, a validated claim driving enterprise adoption.

**Weaknesses**
- Evaluation tooling is fragmented; no single standard for measuring retrieval quality.
- Framework churn creates maintenance debt — teams upgrading LangChain face breaking changes every 3-4 months.

**Opportunities**
- Evaluation-as-a-service is an open market gap; no dominant player yet.
- Local embedding providers could build IDE plugins and attract the "no cloud data" enterprise segment.

**Threats**
- LLM providers building retrieval natively (OpenAI file search, Gemini grounding) reduce the need for custom RAG pipelines.
- Hallucination in retrieved context — a fundamental trust issue — could trigger regulatory scrutiny in healthcare and finance verticals.

---
Engine: oma-market v0.1.0 · sources: reddit, hn, github · window: 30d · clusters: 4 · trust >= 0.6

---

## Example 3 — competitor intent: Cursor vs Windsurf market signal

```
oma-market v0.1.0 · intent: competitor · synced 2026-05-14 · window: 30d
```

# Cursor vs Windsurf: 시장 신호

## Quick Verdict

Cursor holds mindshare with experienced developers who value tab-completion quality and VS Code compatibility. Windsurf is gaining momentum with users who want a more autonomous "write it for me" flow, and its Cascade agent is generating the most organic word-of-mouth this month.

## Cursor

**Tab completion quality remains Cursor's strongest differentiator.** A [r/cursor](https://reddit.com/r/cursor/comments/mno345/why_i_keep_coming_back_to_cursor/) thread with 600 upvotes attributes retention to multi-line completions that "feel predictive, not reactive." Users migrating from GitHub Copilot cite this as the primary pull factor in a [HN discussion](https://news.ycombinator.com/item?id=39200100).

**Pricing friction is the top churn signal.** The $20/month Pro tier generates complaints when users hit the fast-model usage cap mid-month. [r/cursor](https://reddit.com/r/cursor/comments/pqr678/cursor_usage_limits_frustrating/) surfaces this monthly; the workaround community around BYOK (bring your own key) is growing, suggesting willingness to pay but not at current limits.

## Windsurf

**Cascade's autonomous multi-file editing is the breakout feature.** A [product launch thread](https://news.ycombinator.com/item?id=39301500) reached the HN front page and stayed in the top 10 for 18 hours. Users describe Cascade as "closer to a junior dev than an autocomplete," with the ability to plan, edit, and run terminal commands in sequence.

**Onboarding friction is lower than Cursor's.** Multiple [r/webdev](https://reddit.com/r/webdev/comments/stu901/tried_windsurf_for_a_week/) posts note that Windsurf works well out-of-the-box without the VS Code extension configuration that Cursor sometimes requires. However, extension compatibility gaps with niche language plugins are a recurring complaint.

## Head-to-Head

| Dimension | Cursor | Windsurf |
|-----------|--------|----------|
| Completion style | Inline tab, multi-line | Agent-first (Cascade) |
| VS Code compatibility | Near-complete | Partial (Codeium fork) |
| Pricing sentiment | Mixed (cap frustration) | Positive (free tier generous) |
| Mindshare source | r/cursor, HN power users | r/webdev, indie hackers |
| Primary pain | Usage limits | Extension gaps |

## The Bottom Line

Cursor wins on depth for power users; Windsurf wins on accessibility for newcomers. If Windsurf closes the extension compatibility gap, it poses a credible threat to Cursor's mid-market position. The autonomous agent narrative (Cascade) is resonating in a way that tab-completion improvements no longer do.

## SWOT

From Cursor's perspective:

**Strengths**
- Best-in-class inline completion quality, validated by repeat user testimony.
- Deep VS Code ecosystem compatibility; drops into existing workflows with no config.

**Weaknesses**
- Fast-model usage caps create predictable monthly churn spikes.
- Perception of being "just an autocomplete" limits appeal to the agentic-workflow segment.

**Opportunities**
- Shipping a competitive autonomous agent mode would neutralize Windsurf's primary differentiation.
- A teams/enterprise tier with higher limits could convert the BYOK workaround community.

**Threats**
- Windsurf's Cascade is generating stronger organic advocacy in growth-stage developer communities.
- GitHub Copilot Workspace targets the same agentic segment with Microsoft distribution behind it.

## Porter's 5 Forces

_Porter's 5 Forces analysis is available in v1.1 (planned)._

---
Engine: oma-market v0.1.0 · sources: reddit, hn · window: 30d · clusters: 5 · trust >= 0.6
