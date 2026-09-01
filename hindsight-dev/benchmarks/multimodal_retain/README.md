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

`baseline_report.json`, from the branch that introduced inline images
(gemini-2.5-flash for both retain and judging):

| Arm | Image facts recalled | Correct | Wrong | Abstained |
|---|---|---|---|---|
| multimodal | 7/8 (88%) | 5/6 (83%) | 0 | 1 |
| text-only | 0/8 (0%) | 0/6 (0%) | 4 | 2 |

Read it as: without inline images, none of the picture-borne detail survives, and
the majority of answers are confidently wrong rather than absent. With them, most
of it survives and nothing is confidently wrong.

Two caveats worth keeping in view:

- **The remaining multimodal miss is a small supporting label**, not a primary
  one. Button captions and diagram nodes are extracted reliably; the smaller
  subtitle lines (`Profile: corp-eu-west`) are the ones that get dropped. That is
  model behaviour on a rendered image, not a pipeline gap — the image reaches the
  model in position either way.
- **Run-to-run variance is real.** The Wrong/Abstained split in the text-only arm
  moved between runs of the same build; the 0%-vs-88% gap did not. Treat the gap
  as the signal and the split as directional.

### On judging

Negatives are re-asked and upheld only on a majority, mirroring
`tests/llm_judge.py`. This is not ceremony: the first run of this benchmark scored
the claim *"the export control is labelled 'Download CSV'"* as absent from a fact
reading *"...an export control that includes a 'Download CSV' button"*, which
understated the feature by a third. A single temperature-0 judge call flips, and
it flips towards "no".
