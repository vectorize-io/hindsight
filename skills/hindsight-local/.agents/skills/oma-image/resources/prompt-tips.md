# Prompt Tips

Good prompts are specific about scene, subject, lighting, and style. Both `gpt-image-2` (codex) and `gemini-2.5-flash-image` (antigravity / nano-banana) respond well to the same structural cues.

## Structure

```
Scene/backdrop → Subject → Details → Constraints
```

Example: `minimalist product photography: single white ceramic coffee cup on dark marble surface, steam rising softly, dramatic side lighting`

## Do's

- **Lighting**: "warm golden hour side light", "overcast diffused", "backlit with rim light"
- **Camera**: "shallow depth of field", "aerial view", "close-up macro", "35mm film grain"
- **Style**: "photorealistic", "oil painting", "3D render", "concept art", "isometric vector"
- **Mood**: "serene", "dramatic", "moody", "vibrant", "washed-out"
- **Resolution cue**: "ultra detailed", "8K", "high fidelity"

## Don'ts

- Avoid vague prompts like "a nice picture"; both models produce generic output.
- Don't stack contradicting styles ("photorealistic cel-shaded 3D").
- Skip negative prompts; neither `gpt-image-2` nor `gemini-2.5-flash-image` treats them as first-class.

## Examples

| Category | Prompt |
|----------|--------|
| Product | `Elegant perfume bottle on reflective black surface, studio lighting, luxury brand catalog style` |
| Landscape | `Aerial drone shot of Jeju coastline, turquoise water meeting volcanic rock, golden hour` |
| Food | `Overhead flat-lay of Korean bibimbap in stone pot, steam rising, vibrant vegetables, dark wood table` |
| Architecture | `Modern minimalist house with floor-to-ceiling windows overlooking misty mountain valley` |
| Portrait | `Professional headshot, soft natural window light, shallow depth of field, neutral background` |
| UI Mockup | `iPhone 15 Pro mockup showing a fitness app dashboard, clean UI, dark mode, floating on gradient background` |
| Concept | `Lone astronaut on crater edge of Mars, looking at Earth rising on the horizon, cinematic, volumetric dust` |

## Vendor Nuances

- **Codex (`gpt-image-2`)**: prefers slightly longer, descriptive prompts. Quality flag `high` noticeably sharpens fine detail but doubles generation time.
- **Antigravity (`gemini-2.5-flash-image` aka nano-banana)**: robust on CJK-composed scenes ("hanok", "cherry blossom"); slightly stronger on illustrated/painterly styles. Driven via the `agy` CLI's agentic loop — output is JPEG by default even if `.png` is requested (the provider sniffs magic bytes and renames the extension accordingly).

## Comparing Output

`--vendor all` generates the same prompt on both providers and writes both PNGs into a `…-compare/` folder with a single `manifest.json`. Use it for A/B picking when starting a new visual style.

## External Prompt Libraries (agent reference only)

When the **Clarification Protocol** (see `SKILL.md`) reaches the *amplify* step, these galleries provide structural references.

| Source | Vendor match | Prompts | Categories |
|--------|--------------|---------|-----------|
| [awesome-gpt-image-2](https://github.com/YouMind-OpenLab/awesome-gpt-image-2) | `codex` (gpt-image-2) | ~100 | Profile/Avatar · Social Post · Infographic · YouTube Thumbnail · Comic/Storyboard · Poster/Flyer · App/Web Design |
| [awesome-nano-banana-pro-prompts](https://github.com/YouMind-OpenLab/awesome-nano-banana-pro-prompts) | `antigravity` (nano-banana-pro) | 10,000+ | Same taxonomy; 16 localized READMEs (en, ko-KR, ja-JP, zh-TW, de-DE, …) |

For `pollinations` (flux / zimage), patterns from either library transfer; use the subject framing, lighting cues, and style-keyword structure, not the literal wording.

**How to use (agent-side):**

1. Classify the user's intent into one of the 7 categories. If no match, fall back to the *Scene/backdrop → Subject → Details → Constraints* template at the top of this file.
2. Fetch the relevant README section via `gh api`, e.g. Korean user + nano-banana:
   ```
   gh api repos/YouMind-OpenLab/awesome-nano-banana-pro-prompts/contents/README_ko-KR.md \
     --jq .content | base64 -d | less
   ```
   Scan headings like `### No. N: <Category> - <Title>` to locate 1–2 analogous entries.
3. **Internalize the pattern, do not reproduce the text.** Extract lighting / framing / camera / style-keyword choices. The structural approach is not copyrightable; the expression is.
4. Compose your amplified prompt by applying that pattern to the user's subject in your own wording. Show the result to the user for approval before invoking `oma image generate`.

**Fallback:** If no category matches, use the structural template at the top of this file.
