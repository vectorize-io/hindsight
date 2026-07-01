# Prompt Writing Rules for TTS

These rules apply to the text that the agent sends to Voicebox. STT input is audio, so these tips cover only the TTS side.

## Core rules

1. **Plain text only.** Do not include SSML tags. Voicebox does not advertise SSML support; tags will either be read aloud or stripped silently. Use punctuation, line breaks, and pacing words to control delivery.
2. **One call per logical thought.** Keep a sentence intact rather than splitting mid-clause across MCP calls. Voicebox prosody works best on whole sentences.
3. **Strip markdown markers** before sending. Remove `#`, `*`, `_`, backticks, list bullets, and table characters. They have no audible meaning and add awkward pauses.
4. **Strip emojis.** Voicebox usually reads them as the literal Unicode name, which sounds wrong. Replace with the intended word when relevant ("success" instead of a check mark).
5. **Mixed-language text stays in a single call.** Pick a multilingual engine. Splitting Korean and English into separate calls breaks intonation and timing.

## Punctuation and pacing

- Use a comma where you want a short breath. Use a period for a longer stop.
- A blank line between paragraphs produces a longer pause than punctuation alone.
- Avoid stacked exclamation marks. Voicebox treats them as a single emphatic stop. Use one when emphasis matters.
- Use ellipsis sparingly for hesitation. Three dots `...` works; one dot does not.

## Numbers, units, and acronyms

- Spell out numbers when natural reading matters: "삼 킬로그램" rather than "3kg", or "three kilograms" rather than "3 kg".
- Decimals: keep digit form when context is data-heavy ("0.42 percent"), spell out when conversational ("nearly half").
- Currency: write the symbol where readers expect it ("$12.50") and let Voicebox handle the read-out. If Voicebox misreads, switch to spelled form ("twelve dollars fifty").
- Acronyms: spell out the first occurrence when the meaning matters ("Application Programming Interface, A P I"). For short notifications, leave well-known acronyms as is.

## Notifications

- One sentence, fewer than 240 characters.
- Lead with the outcome: "Build succeeded, 4 minor warnings" rather than "I have completed the build process and there are 4 minor warnings".
- Do not stack adjectives. "Tests passed" beats "All the tests have now successfully passed".
- Skip filler ("just to let you know that"). Time is the entire value of a voice notification.

## Asset narration

- Read the text aloud yourself before sending. Awkward phrasing becomes obvious.
- Prefer active voice. Voicebox handles passive voice but it stretches the pacing.
- Where ambiguity exists ("read" past vs present), reword. Voicebox cannot disambiguate from context alone.

## Caveats

- These are recommendations, not hard validators. The skill does not strip markdown or emojis automatically in v1; the calling agent should normalize text before invoking the MCP tool.
- If a future Voicebox release supports SSML or prosody hints, revisit rule 1 and add a section on supported tags.
