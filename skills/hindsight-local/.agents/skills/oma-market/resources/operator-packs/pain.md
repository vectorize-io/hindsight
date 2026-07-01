# Operator pack injected as query preamble when intent=pain — surfaces complaints, churn signals, and migration stories.

## English OR Clause

```
(broken OR bug OR crash OR slow OR freeze OR migrate OR migrating OR ditched OR quit OR ditch OR alternative OR replacing OR painful OR frustrating OR hate OR worst)
```

## Korean OR Clause

```
(불편 OR 버그 OR 느림 OR 이탈 OR 떠났다 OR 마이그레이션 OR 짜증 OR 답답 OR 최악 OR 문제)
```

## Noise Reduction Suffixes

For X (Twitter) and X-like sources, append:

```
-is:retweet -is:reply
```

This suppresses retweet amplification and reduces thread noise that lacks primary signal.

## Rationale

Pain signals are the highest-value leading indicator for product-market gaps. Users venting about a tool are actively seeking alternatives and are primed for acquisition. The OR clause is deliberately broad to capture the linguistic diversity of complaint expression — from technical terms (`crash`, `bug`) to emotional language (`frustrating`, `hate`) to intent-revealing actions (`ditched`, `migrating`). The Korean clause mirrors the same emotional spectrum for KR-market harvests. Noise reduction strips amplified content so the fuse stage operates on original-expression signals only.

## Usage

This pack is auto-injected when `--intent pain` is resolved. It can be suppressed with `--no-operators`.

```
# Query built by execution-protocol Step 3:
"<topic> (broken OR bug OR crash ... )"
```
