# Operator pack used when intent=competitor or --vs is set — surfaces switching signals and direct comparisons.

## English OR Clause

```
(vs OR versus OR alternative OR replaced OR switched OR "migrating from" OR comparison OR compare OR benchmark OR "better than" OR "worse than" OR "switch from" OR "moved to")
```

## Korean OR Clause

```
(대안 OR 대체 OR 비교 OR 옮겨 OR 갈아탔 OR 전환 OR 스위치 OR 비교군 OR 옮겼다 OR 넘어갔다)
```

## Fan-Out Behavior with `--vs`

When `--vs <entity>` is provided, the harvest stage runs a separate query per entity:

```
# Single --vs:
"<topic> <entity> (vs OR alternative OR switched ...)"

# Multiple --vs entities (fan-out):
"<topic> NotionAI (vs OR alternative ...)"
"<topic> Asana (vs OR alternative ...)"
"<topic> Linear (vs OR alternative ...)"
```

Each fan-out result is scored and clustered independently, then merged by the fuse stage into a side-by-side comparison structure.

## Notes

- Without `--vs`, the competitor pack returns generic switching signals — useful for discovering undocumented competitors.
- With `--vs`, the pack targets directed comparison — useful for positioning and win/loss analysis.
- Noise reduction: `-is:retweet` recommended for X sources to avoid amplified hot-takes.
