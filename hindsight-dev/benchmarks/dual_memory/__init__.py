"""D-experiment harness: three-arm comparison of memory deployments.

Stage-1 evidence for the Hindsight × Graphiti integration decision gates
(design docs: HINDSIGHT_GRAPHITI_EVAL_4WAY.md). Arms: Hindsight-only
(per-agent private banks), Graphiti-only (one shared graph), and dual-mount
(both). The multi-agent task set is derived mechanically from LoCoMo by
splitting each conversation's sessions between two agents.
"""
