# Anti-AI Writing Patterns for Translation

Translated text should read like a human wrote it in the target language from scratch.
These patterns are common in AI-generated or AI-translated text. Avoid all of them.

---

## Content Patterns

### 1. Inflated Significance

**Avoid:** *stands/serves as, is a testament/reminder, a vital/significant/crucial/pivotal role, underscores/highlights its importance, reflects broader, symbolizing its enduring, setting the stage for, key turning point, indelible mark*

AI inflates importance of mundane subjects. In translation, this manifests as adding emphasis that wasn't in the source.

- Don't add "중요한", "핵심적인", "획기적인" where the source doesn't emphasize
- Don't turn a simple description into a grand statement
- Translate the weight of the original, not more

### 2. Superficial Analysis via -ing Phrases

**Avoid:** *highlighting/underscoring/emphasizing ..., ensuring ..., reflecting/symbolizing ..., contributing to ..., fostering ...*

AI appends shallow analysis as participle phrases. In Korean/Japanese translation, these become awkward trailing clauses.

- EN: "The update improves performance, **ensuring a seamless experience**"
- Bad KO: "업데이트는 성능을 향상시키며, **원활한 경험을 보장합니다**" (직역)
- Good KO: "업데이트로 성능이 좋아졌어요" (의미만 살림)

### 3. Promotional Tone

**Avoid:** *boasts a, vibrant, rich (figurative), profound, enhancing, showcasing, exemplifies, commitment to, groundbreaking, renowned*

AI defaults to positive, promotional language. Translation should match the source's actual tone. If the source is neutral, the translation must be neutral.

- Don't upgrade "good" to "excellent" during translation
- Don't add marketing flair that wasn't there

### 4. Vague Attribution

**Avoid:** *Experts argue, Some critics argue, Industry reports suggest, Observers have cited*

If the source has a specific attribution, keep it specific. If the source is vague, don't make it vaguer.

### 4a. Notability and Media Padding

**Avoid:** unsupported authority padding such as *covered by major media*, *widely recognized*, *leading expert*, *active social presence*, or long publication-name lists that do not add a concrete claim.

In translation and adaptation, do not make a weak source sound more notable than it is. If the source lists authority markers without substance, preserve the factual claim plainly or flag that the sentence needs a source.

### 4b. Formulaic Challenges/Future and Generic Conclusions

**Avoid:** formulaic endings such as *Despite these challenges*, *future outlook*, *exciting times ahead*, *the future looks bright*, *major step in the right direction*, or *continues its journey toward excellence*.

These closers sound assembled. Translate the actual next step, risk, or conclusion instead. If the source itself is generic, keep it restrained rather than making it more polished.

---

## Language Patterns

### 5. AI Vocabulary Overuse

Words that appear far more in AI text than human text. Avoid overusing these in translated output:

**English:** *Additionally, align with, crucial, delve, emphasizing, enduring, enhance, fostering, garner, highlight (verb), interplay, intricate, key (adjective), landscape (abstract), pivotal, showcase, tapestry (abstract), testament, underscore (verb), valuable, vibrant*

**Korean equivalents to watch:**
- 또한 (sentence-initial) → vary with 그리고, 게다가, or restructure
- 핵심적인/중요한/필수적인 → use only when source genuinely emphasizes
- 활용하다 → 쓰다, 사용하다 are often more natural
- 다양한 → often unnecessary, remove if it adds no meaning
- ~를 통해 → restructure to avoid this construction when possible
- 극대화하다/최적화하다 → don't escalate the source's intent

### 6. Copula Avoidance

AI replaces simple "is/are/has" with fancier alternatives:
- "serves as a" → just "is a"
- "boasts/features/offers" → just "has"

In translation, use the simplest natural equivalent. Korean ~이다/~있다 are fine.

### 7. Rule of Three

AI overuses triple constructions: "adjective, adjective, and adjective."

- EN: "a fast, reliable, and intuitive experience"
- Bad KO: "빠르고, 안정적이며, 직관적인 경험" (mechanical triple)
- Good KO: "빠르고 쓰기 편한 경험" (natural compression)

If the source uses rule of three, you may compress or keep it. Follow what sounds natural in the target language.

### 8. Elegant Variation (Synonym Cycling)

AI avoids repeating words by cycling synonyms: user → participant → key player → stakeholder.

In translation, **consistent terminology matters more** than variety. If "사용자" is the right word, use "사용자" every time. Don't cycle to "이용자", "참여자", "유저" for variety's sake.

### 9. Negative Parallelisms

AI loves "Not only ... but also ..." and "It's not just about X, it's about Y."

These structures feel unnatural when translated literally. Restructure:
- Bad KO: "X뿐만 아니라 Y도" (every time)
- Good KO: vary the structure, or simplify to just state both facts

### 10. Hyphenated Compound Adjectives

AI overuses hyphenated compound adjectives as modifiers:

**Common offenders:** *AI-powered, data-driven, cloud-based, user-friendly, enterprise-grade, production-ready, privacy-focused, community-driven, developer-friendly, mobile-first, cross-platform, open-source, real-time, end-to-end, high-performance, next-generation*

These stack up quickly and create jargon-heavy text:
- Bad EN: "an AI-powered, cloud-based, enterprise-grade solution"
- Bad KO: "AI 기반의, 클라우드 기반의, 엔터프라이즈급 솔루션" (직역 나열)
- Good KO: "클라우드에서 돌아가는 AI 솔루션" (풀어서 자연스럽게)

Rules:
- One per sentence is acceptable; two or more stacked is a red flag
- In Korean/Japanese, unpack the compound into a natural clause rather than calque-translating
- Not every hyphenated adjective needs to survive translation; drop if redundant

### 11. Adjective-Noun Compound Stacking

AI heavily relies on "adjective + noun" or "noun + adjective" compound phrases:

**English:** *seamless integration, robust solution, intuitive design, comprehensive overview, scalable architecture, streamlined workflow, cutting-edge technology, holistic approach*

**Korean equivalents:**
- 원활한 통합, 강력한 솔루션, 직관적인 디자인, 포괄적인 개요
- 확장 가능한 아키텍처, 간소화된 워크플로우, 최첨단 기술

These compounds stack up and create a dense, unnatural rhythm:
- Bad: "직관적인 UI와 강력한 성능, 원활한 연동을 제공합니다"
- Good: "UI가 쓰기 편하고, 빠르고, 연동도 잘 돼요"

Rules:
- Break compounds into simpler, spoken-style expressions
- If a compound feels like it came from a press release, rewrite it
- One compound per sentence is fine; three in a row is a red flag

### 12. False Ranges

AI uses "from X to Y" with loosely related endpoints.

- Bad: "from cutting-edge technology to heartfelt stories"
- These are meaningless in any language. Drop or restructure.

---

## Style Patterns

### 13. Boldface Overuse

AI bolds key terms mechanically, especially in lists: "**Feature Name**: description."

In translation output:
- Don't add bold that wasn't in the source
- Don't format lists as "**bold header**: description" unless the source does

### 14. Em Dash Overuse

AI uses em dashes (—) where commas, colons, or parentheses are more natural.

In Korean, em dashes are rare. Use:
- Commas or restructured sentences instead
- Parentheses for asides

### 14a. Mechanical Punctuation Swap (Anti-Pattern)

When source uses an `X — Y and Z` em-dash pattern and the target language doesn't use em dash naturally in that position, **AI tends to swap the em dash for `:` / `(` / parens and call it done**. This is not translation; it's punctuation substitution that preserves source-language structure.

Example failure (Korean):
- Source: `Documentation drift checks — broken refs and diff-affected docs`
- Lazy swap: `문서 drift 체크: 깨진 참조와 diff 영향받는 docs` (em dash → colon, structure unchanged, "체크"/"diff 영향받는" Konglish)
- Restructured: `참조 무결성 검사, 변경 영향 문서 식별` (matches sibling style: comma-coordinated noun phrases, native vocabulary)

The em dash separator implies a definitional `definiendum — definiens` structure that may map to:
- Coordinated noun phrases joined by commas / `및` / `와/과`
- Relative clauses (target-language pre-nominal modifiers)
- Separate sentences
- A different grammatical pivot entirely

Run the **Sibling-pattern match** check (Stage 4 mechanical) before emitting: if siblings use commas and your draft uses `:`, BLOCK and revise.

### 15. Title Case in Headings

AI capitalizes all main words in headings. This is an English convention. Korean/Japanese headings should not mimic this pattern; just write naturally.

### 16. Unnecessary Tables

AI creates small tables that would be better as prose. In translation, don't introduce tabular format that wasn't in the source.

### 16a. Inline-Header Vertical Lists

AI often writes bullets as bold mini-headings followed by colons. Do not introduce this style unless the source already uses it or the target format requires it.

- Bad: `- **Performance:** Load times were improved.`
- Better: `Load times improved.` or a normal bullet matching sibling style

### 16b. Emoji Decoration

Do not add emoji to headings, bullets, or status labels. Preserve emoji only when they are part of the source content or an established UI convention in the target file.

### 16c. Fragmented Heading Warmups

AI often places a heading, then a one-line paragraph that merely restates the heading before the real content starts. Remove or rewrite these warmups in adaptation/review mode. In strict translation mode, preserve structure but avoid adding a new warmup sentence.

---

## Communication Artifacts

### 17. Chatbot Phrases

**Never include in translated output:**
- "I hope this helps"
- "Let me know if you need anything else"
- "Here is a breakdown of..."
- "Of course!", "Certainly!"

These are chatbot artifacts, not content.

### 18. Hedging and Disclaimers

**Avoid:** *it's important to note, worth noting, it's crucial to remember, may vary*

If the source doesn't hedge, the translation shouldn't either.

### 18a. Signposting and Announcements

**Avoid:** *let's dive in*, *let's explore*, *let's break this down*, *here's what you need to know*, *now let's look at*, *without further ado*.

These phrases announce the writing instead of doing the writing. In translation, usually drop them or replace them with the actual claim.

### 18b. Persuasive-Authority Tropes

**Avoid:** *the real question is*, *at its core*, *in reality*, *what really matters*, *fundamentally*, *the heart of the matter*, *the deeper issue*.

Use these only if the source author genuinely uses that rhetorical stance. Otherwise, translate the concrete claim directly.

### 18c. Knowledge-Cutoff and Availability Disclaimers

**Avoid:** *as of my last update*, *based on available information*, *specific details are limited*, *readily available sources*.

These are usually chatbot artifacts. Do not preserve them unless the source text is explicitly about uncertainty or source limitations.

---

## Filler and Rhythm Patterns

### 18d. Filler Phrase Compression

Compress empty setup phrases when reviewing or adapting prose:

- `in order to` -> `to`
- `due to the fact that` -> `because`
- `at this point in time` -> `now`
- `has the ability to` -> `can`
- `it is important to note that` -> usually delete

In strict translation mode, preserve the author's intended emphasis but do not add filler.

### 18e. Sterile Rhythm

For prose, marketing, blogs, interviews, and adaptation tasks, scan for writing that is technically correct but too evenly shaped: same-length sentences, identical paragraph arcs, neutral summary without a stance where the genre expects one, or transitions that feel like a template.

Fix by matching the source/author voice: vary sentence rhythm, keep concrete details, and let the target language use its natural cadence. Do not add first person, opinions, humor, or stronger emotion unless the source or user asks for adaptation.

---

## Europeanized / Translation-ese Patterns

Patterns where target-language output mimics source-language (typically English) grammar instead of following native structure. Especially critical for CJK translations (Korean, Japanese, Chinese).

### 19. Unnecessary Connectives

AI over-inserts logical connectives where context already implies the relationship:

**English connectives that leak into CJK:**
- Therefore / However / Additionally / Furthermore / Moreover

**Korean equivalents to watch:**
- 따라서, 그러므로, 하지만, 게다가, 또한, 더 나아가
- If the previous sentence already implies the logical relationship, drop the connective
- Bad KO: "성능이 향상되었다. **따라서** 사용자 경험이 좋아졌다."
- Good KO: "성능이 향상되면서 사용자 경험도 좋아졌다."

**Chinese equivalents:**
- 因此, 然而, 此外, 另外, 同时
- Bad ZH: "性能提升了。**因此**，用户体验变好了。"
- Good ZH: "性能提升了，用户体验也跟着变好。"

### 20. Passive Voice Abuse

English uses passive voice frequently. CJK languages strongly prefer active voice.

- **Korean:** ~에 의해, ~(으)로 인해, ~되어지다 → restructure to active
  - Bad: "이 기능은 팀에 **의해 개발되었다**"
  - Good: "팀이 이 기능을 개발했다"
- **Chinese:** 被, 由, 受到 overuse
  - Bad: "这个功能**被团队开发了**"
  - Good: "团队开发了这个功能"
- **Japanese:** ~される, ~られる overuse where能動態 is natural

### 21. Noun Pile-up (Long Modifier Chains)

English stacks modifiers before nouns. CJK languages read better when modifier chains are broken into shorter clauses.

- Bad KO: "AI 기반의 클라우드 지원 실시간 데이터 모니터링 시스템"
- Good KO: "AI를 활용해 클라우드에서 실시간으로 데이터를 모니터링하는 시스템"
- Bad ZH: "基于AI的云端支持的实时数据监控系统"
- Good ZH: "一个用AI在云端做实时数据监控的系统"

Rule: 3+ stacked modifiers before a noun → break into clauses.

### 22. Over-nominalization

English often uses abstract nouns where CJK languages prefer verbs or adjectives.

- Bad KO: "논의를 **진행했다**" → Good: "**논의했다**"
- Bad KO: "개선을 **실시하다**" → Good: "**개선하다**"
- Bad ZH: "进行了**讨论**" → Good: "**讨论了**"
- Bad ZH: "做出了**改进**" → Good: "**改进了**"

Watch for: ~를 진행하다, ~를 실시하다, ~에 대한 검토, 进行~, 做出~, 实施~

### 23. Awkward Pronoun Insertion

Korean, Japanese, and Chinese allow (and prefer) pronoun omission when the subject is contextually clear. English requires explicit subjects.

- Bad KO: "**우리는** 이 기능을 출시했고, **우리는** 좋은 반응을 얻었다"
- Good KO: "이 기능을 출시했고, 반응이 좋았다"
- Bad ZH: "**我们**推出了这个功能，**我们**得到了好的反馈"
- Good ZH: "推出这个功能后，反馈不错"

Rule: If the subject hasn't changed and is clear from context, omit it.

### 24. Cleft Sentence Calques

English "It is X that..." structures should not be directly calqued.

- Bad KO: "중요한 것**은** 사용자 경험**이다**" (직역)
- Good KO: "사용자 경험이 가장 중요하다" (자연스러운 어순)
- Bad ZH: "重要的**是**用户体验" (是...的 calque)
- Good ZH: "用户体验最重要"

### 25. CJK Typography & Fragment Sentences

English technical writing tolerates noun-only fragments ("Lint failed.", "Run conditions: X, Y, Z."). Korean and Japanese readers expect complete sentences with verbs in body text. AI translations calque the source's terse rhythm directly, producing reports that feel choppy and unfinished.

**Body fragments (noun-ending sentences in body text) → rewrite as full sentences:**

- Bad KO: "`ANTHROPIC_API_KEY` 미설정."
- Good KO: "`ANTHROPIC_API_KEY`는 설정하지 않았고, OAuth로 대신 인증합니다."
- Bad KO: "lint 실패."
- Good KO: "lint가 실패했습니다." (standalone) / "결과: lint 실패" (inside a label)
- Bad KO: "anti-pattern 1개."
- Good KO: "anti-pattern이 하나 발견되었습니다."

**Tailing negation calque (do not directly translate English trailing negation fragments):**

English commonly tacks short negation fragments onto the end of a sentence ("..., no guessing", "..., no wasted motion"). Calquing this into CJK produces awkward noun-ending fragments ("..., 추측 없이", "..., 낭비 없이"). In body text, expand the fragment into a full clause.

- Bad EN→KO: "선택된 항목에서 옵션이 나옵니다, 추측 없이."
- Good KO: "선택된 항목에서 바로 옵션을 가져오므로 사용자가 추측할 필요가 없습니다."
- Bad EN→KO: "한 번에 처리합니다, 낭비 없이."
- Good KO: "낭비되는 동작 없이 한 번에 처리합니다."

Inside label/table-cell positions, short negation fragments are acceptable (e.g., `결과: 추측 불필요`).

**Fragments are allowed inside label/table positions:**
- Fragments are natural inside table cells, bullet headers, and `key: value` labels (e.g., `실행 조건:`, `Pass:`).
- Only rewrite fragments that appear abruptly inside narrative body text.

**Spacing between code spans and Korean particles:**

Korean publishing and technical-writing convention attaches the trailing particle directly to an inline code span (`backtick`) with no space. Calquing the English convention of spacing around inline code looks visually broken in Korean.

- Bad: "`prompt` 로 5 개 하네스를 비교합니다"
- Good: "`prompt`로 5개 하네스를 비교합니다"
- Bad: "`oh-my-agent` 소스를 프로젝트에 시드"
- Good: "`oh-my-agent` 소스를 프로젝트에 심습니다" (also avoid Sino-Korean noun transliteration of English verbs; see the next sub-rule)

**No Sino-Korean noun transliteration of English verbs:**

Pinning English verbs into Korean as Sino-Korean nouns ("시드", "로드", "발동", "진행") strips out the action and reads as jargon. Unpack them into natural Korean verbs.

- Bad KO: "skill을 **로드**" → Good KO: "skill을 **불러옵니다**"
- Bad KO: "프로젝트에 **시드**" → Good KO: "프로젝트에 **심습니다**"
- Bad KO: "`<HARD-GATE>` **발동**" → Good KO: "`<HARD-GATE>`가 **트리거됐습니다**" or "`<HARD-GATE>`가 **걸렸습니다**"

---

## Self-Check

Before finalizing any translation, scan for:

- [ ] No AI vocabulary clustering (5+ flagged words in one paragraph)
- [ ] No inflated significance added beyond source
- [ ] No promotional tone upgrade
- [ ] Consistent terminology (no synonym cycling)
- [ ] Natural sentence structure for target language (not source-language word order)
- [ ] No unnecessary bold, em dashes, or formatting artifacts
- [ ] No chatbot communication artifacts
- [ ] No signposting or announcement phrases such as "let's dive in" unless the source intentionally uses them
- [ ] No persuasive-authority tropes such as "the real question is" or "at its core" unless the source voice requires them
- [ ] No generic positive conclusions or formulaic challenge/future sections
- [ ] No unsupported media/notability padding
- [ ] No emoji decoration or inline-header vertical lists introduced by the translation
- [ ] No filler phrases that can be compressed without changing meaning
- [ ] For prose/adaptation tasks, rhythm matches the source or provided voice sample without invented personality
- [ ] No unnecessary connectives where context already implies the relationship
- [ ] No passive voice where active voice is more natural in target language
- [ ] No long modifier chains (3+) stacked before a noun; break into clauses
- [ ] No over-nominalization (verbs/adjectives turned into nouns where unnecessary)
- [ ] No forced pronouns where omission is natural in target language
- [ ] No cleft sentence calques from English
- [ ] No body-text fragments (noun-ending sentences appearing abruptly in CJK body text); labels and table cells are exempt
- [ ] CJK typography: no space between inline code and a trailing Korean particle
- [ ] No Sino-Korean transliteration of English verbs (e.g., 시드 / 로드 / 발동 / 진행); unpack into natural Korean verbs
