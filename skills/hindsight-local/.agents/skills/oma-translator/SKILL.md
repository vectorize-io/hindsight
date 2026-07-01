---
name: oma-translator
description: Context-aware translation that preserves tone, style, and natural word order. Use when translating UI strings, documentation, marketing copy, or any multilingual content. Infers register, domain, and style from the source text and surrounding codebase context.
---

# Translator - Context-Aware Translation

## Scheduling

### Goal
Translate, review, or adapt multilingual content while preserving meaning, register, placeholders, structure, domain terminology, and natural target-language word order.

### Intent signature
- User asks to translate, localize, review translation quality, create a glossary, or adapt UI/docs/marketing copy.
- User needs context-aware translation rather than mechanical word substitution.

### When to use

- Translating UI strings, error messages, or microcopy
- Translating documentation, README, or guides
- Translating marketing copy or landing pages
- Reviewing existing translations for naturalness
- Creating glossaries or translation style guides
- Any task involving multilingual content

### When NOT to use

- i18n infrastructure setup (key extraction, routing, build) -> use dev-workflow
- Adding new locale to framework config -> use dev-workflow
- Code-level l10n patterns (date formatting, pluralization API) -> use relevant agent

### Expected inputs
- Source text, target language, and optional locale or audience
- Existing locale files, glossary, code context, or style constraints
- Optional user/author writing sample for voice matching in prose, marketing, dialogue, or adaptation tasks
- Placeholder syntax, formatting constraints, and output mode

### Expected outputs
- Natural target-language translation or review findings
- Preserved placeholders, code spans, links, headings, lists, and file structure
- Translator notes when source concepts need explanation
- Batch-safe output for i18n files when requested

### Dependencies
- Existing translations and surrounding code for register and terminology
- `resources/translation-rubric.md` and `resources/anti-ai-patterns.md`
- Project locale files when translating UI strings
- User-provided voice samples when the task asks to preserve or match a specific author's style

### Control-flow features
- Branches by content type, target language, batch size, register uncertainty, and placeholder/structure requirements
- Branches by whether style-sample calibration is available and appropriate for the content type
- Reads locale files and source context; may write translated content only when explicitly editing files
- Blocks output until mechanical verification passes

## Structural Flow

### Entry
1. Confirm source text, target language, content type, and output mode.
2. Load existing translations, glossary, file context, or code context when available.
3. Identify placeholders, formatting constraints, and ambiguity.

### Scenes
1. **PREPARE**: Determine language, register, domain, and structure constraints.
2. **ACQUIRE**: Read existing translations and surrounding context.
3. **REASON**: Analyze source meaning, connotations, figurative language, and terminology.
4. **ACT**: Reconstruct natural target-language output.
5. **VERIFY**: Run mechanical checks and translation rubric.
6. **FINALIZE**: Emit translation, review notes, or file changes.

### Transitions
- If context is insufficient, ask one targeted question.
- If batch size is greater than 10 strings, verification is mandatory before output.
- If CJK output contains em dashes or source-language artifacts, rewrite before final output.
- If placeholders or structure do not match, revise and rerun verification.

### Failure and recovery
- If source meaning is ambiguous, flag ambiguity rather than guessing.
- If project conventions conflict with literal translation, follow project conventions and explain if needed.
- If file structure is risky to modify, preserve structure and limit edits to values.

### Exit
- Success: target text is natural, faithful, structurally equivalent, and verified.
- Partial success: ambiguous source text or missing context is explicit.

### Context Inference

No config file required. Instead, infer translation context from:

1. **Existing translations in the project**: scan sibling locale files to match register, terminology, and style already in use
2. **File location**: `messages/`, `locales/`, `.arb` files reveal the framework and format
3. **Surrounding code**: component names, comments, and variable names hint at domain and audience
4. **Source text itself**: register, formality, sentence structure reveal intent

If context is insufficient to make a confident decision, ask the user. Prefer one targeted question over a batch of questions.

### Translation Method

### Stage 1: Analyze Source

Read the source text and identify:
- **Register**: Formal, casual, conversational, technical, literary
- **Intent**: Inform, persuade, instruct, entertain
- **Domain terms**: Words that need consistent translation (check existing translations first)
- **Cultural references**: Idioms, metaphors, humor that won't transfer directly
- **Sentence rhythm**: Short/punchy vs. long/flowing; note parallel structures, intentional repetition, and emphasis patterns
- **Comprehension challenges**: Terms or references target readers may struggle with, such as domain jargon lacking standard translations, cultural references (pop culture, history, social norms), implicit knowledge the author assumes, wordplay or puns, named concepts (e.g., "Dunning-Kruger effect"). For each, note: the original term, why it may confuse, and a concise plain-language explanation for a potential translator's note
- **Figurative language mapping**: For each metaphor, simile, idiom, or figurative expression, classify the handling approach:
  - **Interpret**: Discard source image entirely, express the intended meaning directly in natural target language
  - **Substitute**: Replace with a target-language idiom or image that conveys the same idea and emotional effect
  - **Retain**: Keep the original image if it works equally well in the target language
- **Emotional connotations**: Words carrying subjective feeling beyond dictionary meaning (e.g., "alarming" = urgency, "haunting" = lingering unease); note the emotional effect to preserve in translation

### Stage 2: Extract Meaning

Strip away source language structure. Ask yourself:
- What is the author actually trying to say?
- What emotion or tone should the reader feel?
- What action should the reader take?

Do NOT start forming target sentences yet.

### Stage 2.5: Persona Assignment

Persona resolution has two layers: **content-type** (what kind of text) and **voice** (how punchy or formal the rhythm). Both are needed.

#### Layer 1: Read `translation_voice` from `.agents/oma-config.yaml`

The `translation_voice` field controls global rhythm/formality. Three values:

| Voice | Style override applied on top of content-type |
|---|---|
| `formal` | complete sentences only, no fragments, strict 합니다체/です・ます, no padding cuts |
| `balanced` (default) | content-type defaults; fragments allowed only in label/cell positions |
| `interpreter` | interpreter mindset across all content types: punchy, audience-first, spoken cadence, fragments allowed when natural in target, drops formal padding ("을 받았습니다" → "받음" / "을 모두" → drop) |

If the field is missing, default to `balanced`. If `oma-config.yaml` is unreadable, also `balanced`.

#### Layer 2: Content-type persona table

| Content type | Persona | Base style markers |
|---|---|---|
| UI strings / microcopy | UX copywriter | concise, imperative, user-friendly |
| Docs / README / API reference | technical writer | data + commentary, expanded explanations |
| Benchmark / report / changelog | technical reporter | data + commentary, objective tone |
| Marketing / landing / hero copy | brand copywriter | concise impact, audience-first, aggressive transcreation |
| Blog post / essay | essayist | preserve cadence and rhythm, retain author voice |
| Literary / prose | literary translator | preserve imagery, style consistency, narrative voice |
| Dialogue / subtitle / interview | interpreter | immediacy, audience-first, spoken register, cultural context inline |

Classification heuristics:
- File location `messages/`, `locales/`, `*.arb` → UX copywriter
- Filename `README*`, `docs/*`, or `.md` with frequent code blocks → technical writer
- Score tables, benchmark stats, changelog rows → technical reporter
- Page/section hero copy → brand copywriter
- Quote marks, em-dashes, speaker labels in source → interpreter

When unclear, default to **technical writer** for code-adjacent content and **essayist** for prose. Never use a generic "translator" persona.

#### Combining layers

Voice is applied **on top** of the content-type persona. Examples:

- Content-type = `technical reporter` + voice = `formal` → fully expanded sentences, no fragments anywhere, strict 합니다체.
- Content-type = `technical reporter` + voice = `balanced` → complete sentences in body, fragments allowed in table cells (current default).
- Content-type = `technical reporter` + voice = `interpreter` → punchier rhythm, list-item fragments allowed (e.g., "39턴 / 8m 13s / $1.28 (파일당 $0.14)" instead of "39턴, 8m 13s, 총 $1.28을 썼습니다(파일당 약 $0.14)"), drops "을 모두 받았습니다" padding.

The persona is then **localized to the target language** at execution time. Translating into Korean as a "technical reporter" with `interpreter` voice means thinking as a Korean technical reporter who values rhythm and audience scan-speed over formal completeness.

#### Optional Layer 3: Voice sample calibration

If the user provides an author/user writing sample, analyze it before drafting. Use it as a style constraint, not as permission to alter meaning.

Extract:
- Sentence length pattern: short/punchy, long/flowing, or mixed
- Paragraph entry habit: immediate claim, context first, anecdote, question, or contrast
- Word choice level: casual, technical, academic, literary, blunt, or polished
- Punctuation habits: parentheses, colons, commas, semicolons, dashes, sparse punctuation
- Transition style: explicit connectors, abrupt turns, numbered logic, or minimal signposting
- Recurring phrases or verbal tics that are appropriate to preserve

Apply only where style matters:
- ON: blog posts, essays, speeches, interviews, marketing copy, narrative prose, adaptation requests, and user-authored documentation where preserving author voice is requested
- LIMITED: technical documentation and reports; match rhythm and terminology, but do not add personal stance
- OFF: UI strings, locale key batches, legal/official text, exact policy text, or any text where structure and fidelity outrank authorial style

Guardrail: Voice matching may adjust rhythm, diction, and sentence shape. It must not add new opinions, first-person perspective, humor, facts, examples, or emotional color that is absent from the source.

### Stage 3: Reconstruct in Target Language

Rebuild from meaning **as the assigned persona**, following target language norms:

**Word order**: Follow target language's natural structure.
- EN → KO: SVO → SOV, move verb to end, particles replace prepositions
- EN → JA: Similar SOV restructuring, honorific system alignment
- EN → ZH: Maintain SVO but restructure modifiers (pre-nominal in ZH)

**Register matching**:
- Infer from existing translations in the project, or from source text tone
- Adjust formality markers (honorifics, sentence endings, vocabulary level)

**Sentence splitting/merging**:
- English compound sentences often split into shorter Korean/Japanese sentences
- English bullet points may merge into flowing paragraphs in some languages

**Omission of the obvious**:
- Many languages (Korean, Japanese, Chinese, etc.) allow subject or pronoun omission when contextually clear
- Don't force subjects or pronouns that feel unnatural in the target language

### Stage 4: Verification Gate (blocking; do not emit output until every item is confirmed)

This stage is mandatory. Skipping any item is a bug, not a shortcut. Before producing the final translation, run the mechanical checks first, then the rubric.

**A. Mechanical checks (run before rubric, must all pass):**

- **CJK em dash scan**: For Korean, Japanese, or Chinese targets, search the draft output for `—`. Every occurrence must be **structurally restructured**, never simply substituted with `:` / `(` / `,`. Em dash usually signals a definitional `X — Y` pattern that maps to coordinated noun phrases, relative clauses, or separate sentences in CJK. Zero em dashes AND zero mechanical-substitution survivors in the emitted output. (See anti-AI rules 14 and 14a.)
- **Curly quote scan**: Search the draft output for `“`, `”`, `‘`, `’`. Replace with straight quotes (`"`, `'`) unless the source explicitly uses curly quotes, the target language convention requires them (e.g., Japanese 「」/『』, French «»), or the surrounding file format mandates them.
- **Placeholder integrity**: Every `{name}`, `{{count}}`, `%s`, `<tag>`, and `` `code` `` from the source appears unchanged in the target.
- **Structure parity**: Headings, list bullets, table rows, code blocks, and links match the source count and nesting.
- **Register consistency**: One sentence-ending style throughout (don't mix `-ㅂ니다` with `-다`, formal with casual).
- **Sibling-pattern match (when applicable)**: If the target lives in a context that already contains target-language siblings (markdown table rows, locale file with sibling values, glossary entries, list items in a doc), read at least 3 siblings and identify (a) separator style: comma vs `및`/`와`/`과` vs em dash vs colon vs newline, (b) action-verb form: noun-phrase fragments vs full verb phrases vs imperative, (c) loanword density, (d) register and sentence-ending style. Your draft MUST match the dominant pattern. If the draft uses a separator/verb form/register absent from siblings, BLOCK and revise. Example failure: siblings use comma-separated noun phrases without colons; your draft uses `X: Y and Z` colon syntax. → revise to comma form.

If any mechanical check fails, revise and re-run. Do not proceed to the rubric until all pass.

**B. Translation rubric (see `resources/translation-rubric.md`):**
1. Does it read like it was originally written in the target language?
2. Are domain terms consistent with existing translations in the project?
3. Is the register consistent throughout?
4. Is the meaning preserved (not just words)?
5. Are cultural references adapted appropriately?
6. Are emotional connotations preserved (not flattened into neutral descriptions)?

**C. Anti-AI patterns (see `resources/anti-ai-patterns.md`):**
7. No AI vocabulary clustering or inflated significance
8. No promotional tone upgrade beyond the source
9. No synonym cycling; use consistent terminology
10. No source-language word order leaking through
11. No unnecessary bold or formatting artifacts (em dashes already covered in mechanical check A)
12. No Europeanized patterns (unnecessary connectives, passive voice, noun pile-up, over-nominalization, forced pronouns, cleft calques)
13. No humanizer-pattern leftovers: generic positive conclusions, "let's dive in" signposting, persuasive-authority tropes, formulaic "challenges/future prospects" sections, title-restating warmups, emoji decoration, or vague media/notability padding

**D. Figurative language handling:**
14. Were all metaphors/idioms handled per the classify decision (interpret/substitute/retain)?
15. Do figurative expressions read naturally in the target language, not as literal calques?

**E. Pre-emit gate (must answer in writing before output):**

Before emitting the translation, write 1–2 sentences answering each:

1. **"Why is Stage 5 reflection ON or OFF for this content?"**: must cite the specific classification rule from the "When to run Stage 5–7" section. If the target qualifies for both ON and OFF lists (e.g., README table cell as both a short string AND documentation), default ON wins.
2. **"Does my draft match the sibling patterns in the target context?"**: must reference at least one specific sibling and the matched (or unmatched) pattern dimension.
3. **"Is any source-language structural artifact (em dash, colon-after-X, parentheses-after-noun) merely substituted rather than restructured?"**: must answer No, with evidence.

If any answer is missing, hand-wavy, or "I think so" without evidence, run Stage 5 anyway before emitting.

### Translator's Notes Guidelines

When adding explanatory notes for terms, cultural references, or concepts that target readers may struggle with:

**Format**: `번역어（원어, 쉬운 설명）` or `번역어(원어)` for well-known terms that just need the original

**Calibration by audience**:
- **Technical readers**: Skip annotation on common tech terms (API, deploy, refactor). Only annotate domain-specific or coined terms
- **General readers**: More generous annotation. Explain jargon, cultural references, and domain concepts in plain language
- **Short texts** (< 5 sentences): Minimize annotations; only annotate terms the target audience is unlikely to know

**Rules**:
- Annotate on first occurrence only; don't repeat the note
- Keep notes concise (aim for under 10 words)
- Explain *what it means*, not just provide the English original
- Don't annotate self-explanatory terms or widely recognized loanwords
- If a comprehension challenge was identified in Stage 1, use the pre-planned explanation

### Reflection Mode (default for non-trivial content)

Reflection passes (Stage 5–7) are the default (not optional) for any content that is more than a short snippet. Empirical evidence (Slator 2024, Self-Refine paper) shows a single polish pass cuts translationese rates roughly in half. Skipping reflection on non-trivial content is the most common cause of translationese complaints.

### When to run Stage 5–7

Default ON for:
- Documentation (README, guides, API reference)
- Reports, benchmarks, changelogs, blog posts
- Marketing copy and landing pages
- Any prose longer than ~3 sentences
- Anything containing tables, bullet lists, or code blocks mixed with prose
- Translation review mode

Default OFF (Stage 4 verification only) for:
- Single short UI string (< 10 words) **in a UI locale file** (i18n keys, `.arb`, `.json`, `messages/`) with established glossary
- Batch UI key translations where each value is independent and < 1 sentence
- User explicitly requests "fast translation", "skip reflection", or "직역"

**Tie-breaker rule**: When a target qualifies for BOTH ON and OFF categories, default ON wins. Common conflict cases:

| Situation | Why both | Resolution |
|---|---|---|
| README table cell (short AND documentation) | <10 words but lives in `README*.md` | ON: README is documentation |
| CHANGELOG line entry | <10 words but lives in changelog | ON: changelog is documentation |
| Skill description in registry | short noun phrase but commits to git-tracked source | ON: any git-tracked text |
| Tooltip in i18n file | <10 words AND in `messages/` | OFF: UI string in locale file |

Reflection cost is acceptable; post-merge revision cost is not.

When in doubt, run reflection. The cost is roughly 1.5–2× tokens; the quality gain on body-text fragments and Europeanized patterns is large.

### Extended workflow

After completing Stage 1–4, continue with:

**Stage 5: Critical Review**

Re-read the translation against the source with fresh eyes. Produce a diagnostic review (no rewriting yet).

Start the review by explicitly answering this question first: **"What makes the draft below still feel obviously machine-translated or AI-generated?"** Write 3–7 short bullets naming the remaining tells (e.g., "register suddenly shifts to formal in the final paragraph", "the same connective construction repeats three times", "noun-ending fragments survive in body text outside label/cell positions", "a metaphor was kept literal where the target language would interpret it"). Then continue with the structured checklist:

- **Accuracy**: Compare paragraph by paragraph. Any facts, numbers, or qualifiers altered?
- **Europeanized language**: Scan for unnecessary connectives, passive voice, noun pile-up, over-nominalization, forced pronouns (see `resources/anti-ai-patterns.md`)
- **Figurative language fidelity**: Cross-check metaphor mapping from Stage 1. Were all handled per the classify decision? Any literal calques that sound unnatural?
- **Emotional fidelity**: Were subjective/emotional word choices flattened into neutral descriptions?
- **Tone drift**: Does the register stay consistent from start to finish, or does it shift mid-document (e.g., formal intro drifting into casual explanation)?
- **Expression & flow**: Flag sentences that still read like "translation-ese" (stiff phrasing, unnatural word order, awkward transitions)
- **Humanization patterns**: For prose, marketing, blog, report, and adaptation tasks, scan for sterile rhythm, evenly shaped paragraphs, signposting, generic conclusions, persuasive-authority tropes, formulaic challenge/future sections, emoji decoration, title-restating warmups, and filler phrases
- **Voice sample fit**: If a sample was provided, check whether sentence rhythm, paragraph openings, diction, punctuation, and transition style match the sample without adding unsupported meaning
- **Translator's notes quality**: Too many? Too few? Accurate and concise?

**Stage 6: Revision**

Apply all findings from Stage 5 to produce a revised translation:
- Fix accuracy issues
- Rewrite Europeanized expressions into native patterns
- Re-interpret literally translated metaphors per the mapping
- Restore flattened emotional connotations
- Restructure stiff sentences for fluency
- Adjust translator's notes per review recommendations

**Stage 7: Polish**

Final pass for publication quality:
- Read as a standalone piece: does it flow as native content?
- Smooth remaining rough transitions between paragraphs
- Ensure narrative voice is consistent throughout
- Final scan for surviving literal metaphors or translation-ese
- Verify formatting preservation (headings, bold, links, code blocks)

### Batch Translation Rules

When translating multiple strings (e.g., UI keys):

1. **Read all strings first** before translating any; context matters
2. **Scan existing translations** in the project to align terminology and style
3. **Maintain terminology consistency** across the batch
4. **Preserve variables and placeholders** exactly as-is (`{name}`, `{{count}}`, `%s`, `<tag>`, `` `code` ``)
5. **Keep key structure**: only translate values, never keys
6. **Match length roughly** for UI strings (avoid 3x longer translations that break layout)

### Diff-Sync Mode (patch existing translation against source diff)

Use when the English source has changed and one or more existing target-language translations need to be brought back in sync. Triggered by `oma-docs` v2 multilingual sync, manual i18n catch-up after a docs PR, or any "the source moved, the translation didn't" scenario.

**Inputs**:
- A unified diff of the English source (`/tmp/oma-en-diff.patch` or git diff snippet)
- One or more target-language file paths (existing translations of the same source)
- Optional: per-locale glossary or terminology hints

**Stages override**:
1. **PREPARE**: Read the diff. Identify added, modified, removed sections.
2. **ACQUIRE**: Read each target file. Map source positions to target positions by **heading anchors and surrounding context**, not by line number (line numbers will not match across translations).
3. **REASON**: For each diff hunk, decide:
   - *Added section*: translate fresh, splice in at the equivalent position
   - *Modified text*: localize the modification, replace target equivalent
   - *Removed text*: delete the target equivalent
   - *Touched-but-cosmetic* (whitespace, formatting): skip; don't churn translation
4. **ACT**: Apply patches via Edit tool. Match the existing translation's register, terminology, and voice (re-read at least 3 sibling sections in the target file before writing).
5. **VERIFY**: Run Stage 4 mechanical checks (em-dash, placeholder integrity, structure parity) AND ensure no untouched sections were modified.

**Hard rules for diff-sync**:
- **Touch only what the diff touched.** Other sections of the target file must remain byte-identical. If you find drift outside the diff, flag it but do NOT auto-fix in the same patch.
- **Preserve structural fidelity.** The target file's heading hierarchy, table count, list structure must match the post-patch source.
- **No line-number assumptions.** Always navigate by heading text and anchor, never by absolute line.
- **Code/regex/identifiers in English.** Per i18n-guide rules, code blocks, JSON keys, file paths, regex patterns, workflow names, and system markers like `[OMA WORKFLOW: ...]` stay verbatim.

**Output format (per target file)**:
```
Target: <path>
Sections updated: <list of heading paths>
Sections skipped: <list with reason, e.g. "no semantic change">
Ambiguities resolved: <terminology decisions made>
```

**Parallelization**: When multiple target locales need the same source diff, dispatch one agent per locale in parallel. Each agent gets the same diff but different target-file path. No coordination needed since target files are disjoint.

### Output Format

### Single text
```
Source (EN):
> original text

Translation (KO):
> translated text

Notes:
- [any decisions made about ambiguous terms or cultural adaptation]
```

### Batch (i18n files)
Output in the same format as input (JSON, ARB, YAML, etc.) with only values translated.

### Review mode
```
Original translation:
> existing translation

Suggested revision:
> improved translation

Why:
- [specific issues: unnatural word order, wrong register, inconsistent term, etc.]
```

### Troubleshooting

| Issue | Solution |
|-------|----------|
| Ambiguous source meaning | Flag and ask for context before translating |
| No precedent for a term | Propose a translation, confirm with user before applying |
| Register conflict in source | Follow project's existing register, note the inconsistency |
| Placeholder in middle of sentence | Restructure around it; never break placeholder syntax |
| Translation too long for UI | Provide a shorter alternative with note |
| Multiple valid translations for a term | Pick the one most consistent with project's existing translations; note alternatives |
| Target language requires gendered forms | Follow source text intent; prefer gender-neutral forms when available in target language |
| Tone shifts across a long document | Re-read end-to-end after translating; normalize register to the dominant tone |

### How to Execute

Follow the translation method (Stage 1-4) step by step.
Before submitting, verify against `resources/translation-rubric.md` and `resources/anti-ai-patterns.md`.

### Execution Protocol (CLI Mode)

Vendor-specific execution protocols are injected automatically by `oma agent:spawn`.
Source files live under `../_shared/runtime/execution-protocols/{vendor}.md`.

## Logical Operations

### Actions
| Action | SSL primitive | Evidence |
|--------|---------------|----------|
| Read source and context | `READ` | Text, locale files, code context |
| Select register and terminology | `SELECT` | Existing translations and domain terms |
| Infer intended meaning | `INFER` | Meaning extraction stage |
| Write translation | `WRITE` | Target-language reconstruction |
| Validate placeholders/structure | `VALIDATE` | Verification gate |
| Compare against rubric | `COMPARE` | Translation rubric |
| Report translation or notes | `NOTIFY` | Final output |

### Tools and instruments
- Existing locale files and surrounding code
- Translation rubric, anti-AI-pattern rules, glossary/style references
- File editing tools only when the user requests file changes

### Canonical workflow path
```text
1. Analyze source register, intent, domain terms, placeholders, and structure.
2. Reconstruct meaning in the target language, not word-for-word.
3. Run mechanical checks and `resources/translation-rubric.md` before emitting output.
4. For non-trivial prose, run Stage 5 humanization review before final polish; apply voice-sample calibration only when provided and appropriate.
```

For UI files, scan sibling locale files first:
```bash
rg "<source-key-or-term>" .
```

### Resource scope
| Scope | Resource target |
|-------|-----------------|
| `LOCAL_FS` | Locale files, docs, README, source text files |
| `CODEBASE` | Components and code context around UI strings |
| `MEMORY` | Register, glossary, ambiguity, verification notes |
| `USER_DATA` | User-provided text and target-language requirements |

### Preconditions
- Source text and target language are known.
- Placeholder and structure constraints are identifiable.
- Ambiguities are resolved or explicitly flagged.

### Effects and side effects
- Produces translated text or translation review.
- May modify locale/docs files only when requested.
- Preserves source structure and placeholders.

### Guardrails

1. Scan existing locale files before translating to align with project conventions
2. Preserve placeholders and interpolation syntax
3. Translate meaning, not words
4. Preserve emotional connotations: translate the feeling, not just the dictionary meaning (e.g., "alarming" carries urgency/concern, not merely "surprising")
5. Match register consistently throughout a single piece
6. Split, merge, or restructure sentences for target language naturalness
7. Flag ambiguous source text rather than guessing
8. Preserve domain terminology: if a term has established meaning in the field (e.g., harness, scaffold, shim, polyfill, middleware), keep it even if a "simpler" native word exists
9. Never produce literal word-for-word translations
10. Never mix registers within a single piece (formal + casual)
11. Never replace domain-specific terms with generic equivalents (e.g., "harness" → "framework", "shim" → "wrapper")
12. Never translate proper nouns unless existing translations do so
13. Never change the meaning to "sound better"
14. Never skip verification stage for batches > 10 strings
15. Never modify source file structure (keys, nesting, comments)
16. Never preserve source-language formatting artifacts that are unnatural in the target language. For CJK targets (Korean, Japanese, Chinese), em dashes (—), title case in headings, and trailing "-ing" participle clauses must be restructured, even when the source uses them. See `resources/anti-ai-patterns.md` rules 2 (-ing phrases), 14–15 (em dash, title case), and 25 (CJK typography & fragments).
17. Never "humanize" by inventing personality. Do not add first person, jokes, opinions, examples, facts, citations, stronger emotion, or messiness unless the source or user explicitly calls for adaptation.
18. When a voice sample is provided, match observable style traits only: rhythm, diction level, punctuation habits, transitions, and paragraph shape. Preserve source meaning and target-language naturalness above mimicry.

## References

- Translation rubric: `resources/translation-rubric.md` (5-criterion scoring: naturalness, accuracy, register, terminology, technical integrity)
- Anti-AI patterns: `resources/anti-ai-patterns.md` (AI output patterns + Europeanized/translation-ese patterns to avoid)
- Context loading: `../_shared/core/context-loading.md`
- Quality principles: `../_shared/core/quality-principles.md`
