# Optional positive operator pack — used as a sentiment comparator against pain signals.

## English OR Clause

```
(love OR shipped OR fast OR clean OR delight OR sticky OR best OR favorite OR smooth OR reliable OR recommend)
```

## Korean OR Clause

```
(좋다 OR 사랑 OR 빠르다 OR 깔끔 OR 최고 OR 추천 OR 만족 OR 편하다 OR 훌륭)
```

## Usage

This pack is NOT injected by default. It is used when the research brief requests a pain/positive contrast — e.g., to compute a sentiment ratio or to populate the "What's working" column of a comparison table.

```
# Manual invocation example:
oma market harvest "<topic>" --operator-pack positive
```

It can also be composed with the pain pack via a dual-run:

```
# Run 1 (pain signals):
oma market harvest "<topic>" --operator-pack pain

# Run 2 (positive signals, same topic):
oma market harvest "<topic>" --operator-pack positive
```

The fuse stage merges both result sets and tags each cluster with a `sentiment` field (`pain | positive | mixed`).

## Notes

- Positive signals alone are weak market research inputs; they reveal satisfaction but not opportunity.
- Use this pack primarily when the output format requires a contrast section or NPS-style polarity breakdown.
