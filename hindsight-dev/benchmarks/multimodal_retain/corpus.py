"""Knowledge-base articles whose load-bearing detail lives in the pictures.

This is the shape of the problem inline images exist to solve: an article that
says "click the button shown below" and then shows it. The sentence is useless on
its own, and a memory system that only reads the prose will answer confidently
and wrongly.

So every question here is **unanswerable from the text alone**. That is what makes
the A/B meaningful — the text-only arm is not expected to score badly in a vague
way, it is expected to score near zero, and any points it does score are worth
looking at (a lucky guess, or a detail that leaked into the prose by mistake).

The corpus is small and hand-written on purpose. The measurement is whether
picture content reaches memory at all, which does not need scale to see; a large
generated corpus would cost LLM calls without sharpening the answer.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .images import ImageSpec


class Question(BaseModel):
    """One question, and what a correct answer has to contain."""

    question: str
    expected: str = Field(description="What a right answer must convey. Judged semantically, not by string match.")
    depends_on_image: str = Field(description="Name of the ImageSpec holding the answer.")


class Article(BaseModel):
    """A KB article: prose with images interleaved where they belong."""

    name: str
    title: str
    #: Ordered body. A string is prose; an ImageSpec is an image in that position.
    body: list[str | ImageSpec]
    questions: list[Question]

    @property
    def images(self) -> list[ImageSpec]:
        return [block for block in self.body if isinstance(block, ImageSpec)]

    def text_only_body(self) -> str:
        """The article as it reaches memory without inline images.

        This is the baseline arm, and it models the honest pre-feature behaviour:
        the prose is retained and the image is simply absent. It is NOT a
        caption-stripped variant — pretending the caller wrote alt text would
        measure a different system than the one people actually have.
        """
        return "\n\n".join(block for block in self.body if isinstance(block, str))


VPN_RESET = Article(
    name="vpn-reset",
    title="Resetting the corporate VPN",
    body=[
        "If the VPN client hangs on 'Connecting', a reset usually clears it. "
        "Open Network Settings and click the button shown below:",
        ImageSpec(
            name="vpn-button",
            kind="screenshot",
            title="Network Settings",
            primary="Reset VPN Tunnel",
            lines=["Status: Connecting...", "Profile: corp-eu-west"],
            facts=[
                "The button that resets the VPN is labelled 'Reset VPN Tunnel'.",
                "The VPN profile shown in Network Settings is 'corp-eu-west'.",
            ],
        ),
        "Wait ten seconds after clicking, then reconnect. If it hangs a second time, the tunnel is not the problem "
        "and you should escalate.",
    ],
    questions=[
        Question(
            question="What exactly is the button called that resets the VPN?",
            expected="The button is labelled 'Reset VPN Tunnel'.",
            depends_on_image="vpn-button",
        ),
        Question(
            question="Which VPN profile does the Network Settings screen show?",
            expected="The profile is 'corp-eu-west'.",
            depends_on_image="vpn-button",
        ),
    ],
)

SYNC_ESCALATION = Article(
    name="sync-escalation",
    title="Escalating a stuck data sync",
    body=[
        "A sync that has not advanced in thirty minutes is stuck. Do not page an engineer directly — "
        "follow the escalation path in the diagram below.",
        ImageSpec(
            name="sync-path",
            kind="diagram",
            title="Stuck sync escalation",
            primary="Tier 3 Platform",
            lines=["Owner: Data Platform Team", "Response target: 15 minutes"],
            facts=[
                "A stuck sync escalates to Tier 3 Platform.",
                "The Data Platform Team owns the escalation.",
                "The response target is 15 minutes.",
            ],
        ),
        "Record the sync id in the incident channel before escalating so the on-call has something to work from.",
    ],
    questions=[
        Question(
            question="Where does a stuck sync escalate to?",
            expected="It escalates to Tier 3 Platform.",
            depends_on_image="sync-path",
        ),
        Question(
            question="Which team owns stuck-sync escalations, and what is the response target?",
            expected="The Data Platform Team owns it, with a 15 minute response target.",
            depends_on_image="sync-path",
        ),
    ],
)

BILLING_EXPORT = Article(
    name="billing-export",
    title="Exporting a billing report",
    body=[
        "Billing reports are exported from the Reports tab. The export control is shown here:",
        ImageSpec(
            name="billing-button",
            kind="screenshot",
            title="Reports",
            primary="Download CSV",
            lines=["Period: Q3 2025", "Rows: 18,420"],
            facts=[
                "The billing export control is labelled 'Download CSV'.",
                "The report covers the Q3 2025 period.",
                "The report contains 18,420 rows.",
            ],
        ),
        "Exports are generated asynchronously; large periods can take a few minutes to appear in your downloads.",
    ],
    questions=[
        Question(
            question="What is the billing export button labelled, and what format does it produce?",
            expected="It is labelled 'Download CSV' and produces a CSV.",
            depends_on_image="billing-button",
        ),
        Question(
            question="What period does the billing report on the Reports screen cover, and how many rows does it have?",
            expected="Q3 2025, with 18,420 rows.",
            depends_on_image="billing-button",
        ),
    ],
)

ARTICLES: list[Article] = [VPN_RESET, SYNC_ESCALATION, BILLING_EXPORT]


def article_by_name(name: str) -> Article:
    for article in ARTICLES:
        if article.name == name:
            return article
    raise SystemExit(f"Unknown article {name!r}. Available: {', '.join(a.name for a in ARTICLES)}")
