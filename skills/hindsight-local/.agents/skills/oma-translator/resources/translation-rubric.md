# Translation Quality Rubric

5-point scale per criterion. Total 50 points.

## Criteria

### 1. Naturalness (10 pts)
Does it read like it was originally written in the target language?

| Score | Description |
|-------|-------------|
| 9-10 | Native speaker cannot tell it's a translation |
| 7-8 | Reads naturally with minor awkwardness |
| 5-6 | Understandable but clearly "translated" |
| 3-4 | Awkward phrasing, unnatural word order |
| 1-2 | Word-for-word translation, painful to read |

Red flags:
- Source language word order leaking through
- Unnatural particles or prepositions
- Overly long sentences that should be split
- Forced subjects where omission is natural
- Europeanized patterns: unnecessary connectives (따라서/그러나/또한), passive voice abuse, noun pile-up, over-nominalization, cleft sentence calques

### 2. Accuracy (10 pts)
Is the meaning fully preserved?

| Score | Description |
|-------|-------------|
| 9-10 | Meaning, nuance, and intent fully preserved |
| 7-8 | Core meaning preserved, minor nuance loss |
| 5-6 | Main idea correct but some details lost |
| 3-4 | Partial meaning distortion |
| 1-2 | Meaning significantly changed or wrong |

Red flags:
- Added meaning not in source
- Omitted important qualifiers
- Reversed logic (negation errors)
- Hallucinated content
- Emotional connotations flattened (e.g., "alarming" → neutral "놀라운" instead of urgent "우려되는")
- Figurative language translated literally when the image doesn't work in target language

### 3. Register Consistency (10 pts)
Does the tone match the configured register throughout?

| Score | Description |
|-------|-------------|
| 9-10 | Perfect register match, consistent throughout |
| 7-8 | Mostly consistent with rare slips |
| 5-6 | Noticeable register shifts |
| 3-4 | Inconsistent: mixes formal and casual |
| 1-2 | Wrong register entirely |

Red flags:
- Mixing 합니다체 and 해요체 in Korean
- Switching between です/ます and 普通体 in Japanese
- Formal vocabulary in casual context or vice versa

### 4. Terminology (10 pts)
Are domain terms consistent and correct?

| Score | Description |
|-------|-------------|
| 9-10 | All glossary terms applied, consistent throughout |
| 7-8 | Glossary followed with 1-2 misses |
| 5-6 | Some terms inconsistent across strings |
| 3-4 | Multiple glossary violations |
| 1-2 | Glossary ignored |

Red flags:
- Same source term translated differently in same batch
- Glossary term overridden without justification
- Technical term translated when it should stay in English

### 5. Technical Integrity (10 pts)
Are placeholders, formatting, and structure preserved?

| Score | Description |
|-------|-------------|
| 9-10 | All placeholders, tags, and structure intact |
| 7-8 | Intact with minor formatting differences |
| 5-6 | One placeholder or tag issue |
| 3-4 | Multiple broken placeholders |
| 1-2 | Structure significantly damaged |

Red flags:
- Modified or deleted `{variables}`
- Broken HTML/markdown tags
- Changed JSON/YAML key names
- Added or removed line breaks that affect rendering

## Scoring Guide

| Total | Grade | Action |
|-------|-------|--------|
| 45-50 | Excellent | Ship as-is |
| 38-44 | Good | Minor polish recommended |
| 30-37 | Acceptable | Review flagged items before shipping |
| 20-29 | Needs work | Re-translate problem sections |
| < 20 | Reject | Full re-translation required |

## Quick Self-Check (for translator)

Before submitting, verify:
- [ ] Read the full translation aloud: does it flow?
- [ ] Glossary terms are consistent across all strings
- [ ] Register matches config from start to finish
- [ ] All placeholders survived intact
- [ ] No source language word order leaking through
- [ ] Emotional connotations preserved (not flattened into neutral words)
- [ ] Figurative language handled naturally (interpret/substitute/retain, not literal calques)
- [ ] No Europeanized patterns (passive voice, noun pile-up, forced pronouns, unnecessary connectives)
- [ ] Translator's notes are concise, accurate, and calibrated to target audience
