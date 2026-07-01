# Operator pack used when intent=discovery — surfaces unmet needs, underserved gaps, and wish-list signals.

## English OR Clause

```
(wish OR need OR missing OR underrated OR underserved OR "I want" OR "if only" OR "why doesn't" OR gap OR overlooked OR "nobody builds" OR unmet OR "would love" OR "please add" OR "feature request")
```

## Korean OR Clause

```
(있었으면 OR 필요하다 OR 아쉽다 OR "왜 없지" OR 부족하다 OR 못 찾겠다 OR 니즈 OR 발굴 OR "추가해줘" OR "기능 요청")
```

## Usage

Discovery intent is only triggered by explicit `--intent discovery` flag or by keyword scan with confidence >= 2. It is NOT triggered by fallback chain.

```
# Example invocation:
oma market research "async standup tools" --intent discovery
```

## Notes

- Discovery signals are forward-looking; they reveal what users want to exist, not what exists and fails.
- Combine with trend pack results to distinguish "gap that nobody has filled" from "gap that is being filled but is unknown."
- Signal quality is lower than pain signals — apply higher `--min-trust` threshold in score stage.
- Sources: Reddit (r/entrepreneur, r/startups, r/productivity) and HN "Ask HN" threads are highest-yield for discovery signals.
