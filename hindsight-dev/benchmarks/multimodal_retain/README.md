# Multimodal retain

Is the information in the pictures reaching memory?

Knowledge-base articles routinely put load-bearing detail in images: a screenshot
of the button an instruction refers to, a diagram holding the escalation path.
Retained as text alone, those articles produce memories that point confidently at
things nobody saw — "click the button shown below", with no idea what the button
says. This benchmark measures how much of that is recovered by retaining images
inline, and how the same corpus behaves without them.

## The two arms

Both arms retain the **same prose**. They differ only in whether the images go
with it:

- **multimodal** — each image inline, in the position it occupies in the article.
- **text-only** — the images simply absent. This is the honest pre-feature
  baseline. It is *not* a caption-stripped variant: pretending the caller wrote
  alt text would measure a system nobody has.

Every question in the corpus is **unanswerable from the prose alone**. So the
text-only arm is expected to score near zero, and any points it does score are
worth reading — a lucky guess, or a detail that leaked into the prose by mistake.

## What is measured

Two things, kept apart because they fail differently:

| | |
|---|---|
| **Image facts recalled** | Did detail visible only in the picture reach the bank as facts? The *mechanism*. |
| **Correct / Wrong / Abstained** | Did reflect answer the question? The *outcome*. |

Wrong and Abstained are never collapsed into one "incorrect" bucket. The failure
inline images exist to remove is not "I don't know" — it is a fluent, confident
answer assembled from prose that pointed at a picture. An arm that abstains is
being unhelpful; an arm that invents is being harmful, and the report says which.

## Running it

The server must have a **vision-capable retain LLM**. Without one, the multimodal
arm is refused with `422` and the report says so rather than quietly reporting
zeros.

```bash
# a server with a vision model
HINDSIGHT_API_LLM_PROVIDER=gemini \
HINDSIGHT_API_LLM_MODEL=gemini-2.5-flash \
HINDSIGHT_API_LLM_API_KEY=$GEMINI_API_KEY \
HINDSIGHT_API_PORT=8917 uv run hindsight-api

# then, from hindsight-dev/
uv run python -m benchmarks.multimodal_retain run --api-url http://localhost:8917 --build my-branch

# re-render a saved artifact without spending LLM calls again
uv run python -m benchmarks.multimodal_retain report benchmarks/results/multimodal_retain/<artifact>.json
```

`--article <name>` narrows the run; `--keep-banks` leaves the benchmark banks
behind for inspection.

## The corpus

Three hand-written articles (`corpus.py`), each with images drawn from a typed
spec (`images.py`) rather than committed as fixtures. The spec *is* the ground
truth — what the picture says and what the corpus expects to be recalled come
from the same object, so they cannot drift — and the bytes are deterministic, so
a re-run re-ingests the same content-addressed images.

It is small on purpose. The question is whether picture content reaches memory at
all, which does not need scale to answer; a large generated corpus would spend
LLM calls without sharpening it.

## Baseline

Three runs per build, gemini-2.5-flash for both retain and judging:

| Build | Image facts recalled | Correct |
|---|---|---|
| before the attachment instruction | 50%, 88%, 50% | 67%, 83%, 50% |
| **with it** | **88%, 100%, 75%** | **83%, 100%, 83%** |
| text-only arm (either build) | 0%, 0%, 0% | 0%, 0%, 0% |

The instruction is one paragraph in the retain user message naming the
attachments and putting their content in scope. Until it existed the prompt said
"text chunk" and "Text:" throughout, so a model handed a screenshot was being
told, in effect, to extract from the prose — and routinely did exactly that.
Naming them moved the *floor* from 50% to 75%, which matters more than the
ceiling: the bad runs were the ones that made the feature look unreliable.

`baseline_report.json` holds one run. **Read the gap, not the multimodal
number.** The text-only arm scored exactly zero on every run of both builds — no
picture-borne detail survives without inline attachments, and several answers are
confidently wrong rather than absent.

The multimodal arm still moves between runs of identical code, so do not treat a
single run as a regression signal — reproduce first. When one looks wrong, check
the pipeline directly before blaming it: resolve the chunk's placeholders through
`RetainAttachmentLoader` and assert `build_prompt_parts` returns
`['text', 'image_url', 'text']`. That check has twice shown a "bad" run to be the
model re-reading the same image differently, not a broken path.

Where the variance lives:

- **Primary labels are reliable.** Button captions and diagram nodes ("Reset VPN
  Tunnel", "Tier 3 Platform") survive nearly every run.
- **Small supporting lines are not.** Muted subtitle text (`Profile:
  corp-eu-west`, `Rows: 18,420`) is what gets dropped.
- **PDFs are less reliable than images.** A one-page PDF whose text is a raster
  render is read in some runs and ignored in others, even while the model
  acknowledges "the attached policy" in its output. The bytes demonstrably reach
  the provider — verify with the loader check above before suspecting otherwise.

Both of those last two eased considerably once extraction was told to transcribe
structured attachments rather than summarize them, and to carry each value's
colour and position (see `benchmarks/imageqa`, where that change was measured):

| build | image facts recalled | correct | wrong | abstained |
|---|---|---|---|---|
| `attach-prompt-3` | 6/8 (75%) | 5/6 (83%) | 0 | 1 |
| `visual-key` | 8/8 (100%) | 6/6 (100%) | 0 | 0 |

The muted subtitle lines are exactly the "every other value" a summary discards,
and the PDF is the sync-escalation article. Worth reading as corroboration rather
than proof — six questions is far too few to call a five-point move — but it is
the evidence that the ChartQA work did not simply over-fit to charts.

### On judging

Negatives are re-asked and upheld only on a majority, mirroring
`tests/llm_judge.py`. This is not ceremony: the first run of this benchmark scored
the claim *"the export control is labelled 'Download CSV'"* as absent from a fact
reading *"...an export control that includes a 'Download CSV' button"*, which
understated the feature by a third. A single temperature-0 judge call flips, and
it flips towards "no".
