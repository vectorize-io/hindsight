# Maintenance

## Background

Maintained fork: `0xble/hindsight` of `vectorize-io/hindsight`, branch `main`.
Canonical checkout: `/Users/brianle/Repos/hindsight`. Accepted upstream baseline:
`4cc131c0b238c8f206def60804d7b6591f6a45e7`. Publish only to `origin`; never
push to `upstream`. Source synchronization, publication, installation, and
runtime activation are separate stages.

## Preserve

Keep evidence admission, generated-language policy, operation status, and retained
service hardening compatible across the API and clients. Fork CI must cover the
retained patches and upstream recovery without enabling upstream deployment,
signing, release, or publishing. Source inclusion never implies runtime adoption.

## Required maintenance support

Evaluate these support units every maintenance run, including no-change
runs; these extend this sole enrollment and scheduling unit:

- [OCR admission and typed failures](maintenance/ocr.md): HINDSIGHT-001, HINDSIGHT-004 and HINDSIGHT-006.
- [Service hardening](maintenance/service-hardening.md): HINDSIGHT-002 and HINDSIGHT-007.
- [Fork CI governance](maintenance/ci.md): HINDSIGHT-003.
- [Generated-language integrity](maintenance/language-integrity.md): HINDSIGHT-005 and HINDSIGHT-008.
- [Bounded consolidation schema correction](maintenance/schema-correction.md): consolidation-only divergence.
- [Codex injection budget](maintenance/codex-injection-budget.md): every run, preserve the complete-context cap and offline tokenizer recovery.
- [Codex session attribution](maintenance/codex-session-attribution.md): when adopting Codex retain/config changes, preserve explicit strategy scope and source-status safeguards.

## Upstream-owned recovery

Codex quota deferral is provided by upstream [#4161](https://github.com/vectorize-io/hindsight/pull/4161)
(`31da16737d98e808e21c4708379eee8c52aff546`). Processing-operation cancellation
is provided by [#4177](https://github.com/vectorize-io/hindsight/pull/4177)
(`b1de1b941857c62eb4963185450e93dcee96be70`). Both are included in the accepted
baseline. Keep these paths upstream-owned rather than adding parallel fork fixes.

- A validated provider quota-reset time defers work. Previously failed facts are
  not automatically recovered by adopting this code. Diagnose and recover those
  separately through the supported operation or consolidation APIs.
- Cancel pending or processing operations through the normal cancellation API,
  not direct database status updates. Cancellation is cooperative. Read back the
  operation and affected batch parent before declaring recovery complete.
- Per-bank consolidation claim serialization is already upstream-owned and was
  omitted from the original HINDSIGHT-002 port. Do not restore that local layer.
- Source inclusion is not runtime activation. Verify installed source identity,
  migrations, health, and recovery behavior before claiming a service adopted it.

Fork CI runs the upstream Codex/provider quota-deferral, cancellation, worker,
and operation-status regressions alongside retained-patch regressions.

## Update and verify

Every maintenance run fetches `origin` and `upstream` separately and reconciles
`main` with the latest stable upstream release (the GitHub latest release,
excluding drafts and prereleases) by merging its tag, preserving each
intentional registered patch. Unreleased `upstream/main` commits are never a
sync target. When `main` already contains that release tag, source sync is
current. Evaluate all support-file adoption and retirement
conditions; update the responsible record in the same delivery as any patch
addition, change, or retirement. Missing or stale patch coverage blocks publication.

Run all retained-patch and upstream-recovery regressions on the exact candidate.
From `hindsight-api-slim`, also run `uv run --frozen ruff check .`,
`uv run --frozen ruff format --check .`, and `uv run --frozen ty check hindsight_api`;
run the HINDSIGHT-003 regression from the repository root.

Publish to owned `origin` when authorized, or report `Blocked` with the exact
failed stage, refs, and evidence. Before `Updated` or `Already current`, fetch
upstream tags again and require zero release-only commits with
`git rev-list --left-right --count <latest-release-tag>...main`, then read back
local/`origin/main` SHA parity. Require exact installed/runtime SHA proof only
when those separately authorized stages are in scope.
