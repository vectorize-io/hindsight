"""The repo the agent works in, and the turns it is asked to take.

Both halves are chosen to make the measurement honest rather than flattering:

* **The repo is small but real** — a package with a config layer, a couple of
  modules that reference each other, tests, and a git history whose messages
  carry the *why*. That history is what the plugin seeds memory from, so a repo
  with one commit would measure a plugin with nothing to retrieve.
* **The prompts are shaped like real ones.** They were rewritten from a sample
  of 1168 actual user turns from this developer's last week of Claude Code:
  terse imperatives, lowercase, typos, each one continuing the work rather than
  opening a topic (median length 114 characters). The rare question is about
  what just happened, not about repo history.

  The first draft was not like that — it asked "why does pricing use Decimal
  instead of float here?" and "what's our convention for reporting errors?" —
  and it measured 27.5% searches/turn against a field rate of 0.5%, with 10 of
  11 searches landing on exactly those two prompts. An interview question is a
  search request in disguise: it measures whether the model can follow an
  instruction it was practically handed, not whether the plugin gets it to
  reach for memory during ordinary work.
"""

from __future__ import annotations

import subprocess
from collections.abc import Awaitable, Callable
from pathlib import Path

#: Wait for a bank's background work to finish — the suite's settle helper,
#: passed in rather than imported so this module stays free of fixtures.
SettleFn = Callable[[str], Awaitable[None]]

#: One scripted session. Sessions in a run are independent: same prompts, fresh
#: Claude session each time, so the per-turn rate averages over model sampling.
BASELINE_PROMPTS: list[str] = [
    "add a discount_pct to the pricing config and apply it in quote",
    "now wire it into post_quote too, should come from the payload",
    "hmm the discount and the free shipping threshold interact weirdly, sort it out",
    "add a test for it",
    "getting a 409 on reserve with qty 0, that's wrong - look into it",
    "nice. btw why is everything Decimal here",
    "ok commit this",
    "what's left?",
]

_FILES: dict[str, str] = {
    "README.md": """# orderbook

A small order-pricing service. Three pieces:

- `orderbook/config.py` — pricing knobs, loaded once at startup
- `orderbook/pricing.py` — turns a cart into a quote
- `orderbook/inventory.py` — reserves stock for a quote
- `orderbook/api.py` — the HTTP surface
""",
    "orderbook/__init__.py": "",
    "orderbook/config.py": '''"""Pricing configuration, loaded once at startup."""

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class PricingConfig:
    currency: str = "EUR"
    tax_rate: Decimal = Decimal("0.22")
    #: Orders at or above this total ship for free.
    free_shipping_threshold: Decimal = Decimal("50.00")
    shipping_flat: Decimal = Decimal("4.90")


DEFAULT = PricingConfig()
''',
    "orderbook/pricing.py": '''"""Cart -> quote."""

from decimal import ROUND_HALF_UP, Decimal

from orderbook.config import DEFAULT, PricingConfig


class PricingError(Exception):
    """A cart that cannot be priced."""


def _money(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def quote(lines: list[tuple[str, int, Decimal]], config: PricingConfig = DEFAULT) -> dict:
    """Price a cart. Each line is (sku, quantity, unit_price)."""
    if not lines:
        raise PricingError("an empty cart has no quote")

    subtotal = Decimal("0.00")
    for sku, quantity, unit_price in lines:
        if quantity <= 0:
            raise PricingError(f"{sku}: quantity must be positive")
        subtotal += unit_price * quantity

    tax = _money(subtotal * config.tax_rate)
    shipping = (
        Decimal("0.00") if subtotal >= config.free_shipping_threshold else config.shipping_flat
    )
    return {
        "currency": config.currency,
        "subtotal": _money(subtotal),
        "tax": tax,
        "shipping": _money(shipping),
        "total": _money(subtotal + tax + shipping),
    }
''',
    "orderbook/inventory.py": '''"""Stock reservation.

Reservations are held in memory: this service is single-process today.
"""

from dataclasses import dataclass, field


class OutOfStock(Exception):
    pass


@dataclass
class Inventory:
    on_hand: dict[str, int] = field(default_factory=dict)
    reserved: dict[str, int] = field(default_factory=dict)

    def available(self, sku: str) -> int:
        return self.on_hand.get(sku, 0) - self.reserved.get(sku, 0)

    def reserve(self, sku: str, quantity: int) -> None:
        if self.available(sku) < quantity:
            raise OutOfStock(f"{sku}: only {self.available(sku)} available")
        self.reserved[sku] = self.reserved.get(sku, 0) + quantity

    def release(self, sku: str, quantity: int) -> None:
        self.reserved[sku] = max(0, self.reserved.get(sku, 0) - quantity)
''',
    "orderbook/catalog.py": '''"""Catalog lookups.

Deliberately uncached: prices change during flash sales and a stale price is a
customer-visible error, while the lookup itself is a single indexed read.
"""

from decimal import Decimal

_ROWS: dict[str, Decimal] = {
    "SKU-1": Decimal("19.99"),
    "SKU-2": Decimal("45.00"),
    "SKU-3": Decimal("7.50"),
}


def unit_price(sku: str) -> Decimal:
    try:
        return _ROWS[sku]
    except KeyError as exc:
        raise LookupError(f"unknown sku {sku}") from exc
''',
    "orderbook/api.py": '''"""HTTP surface.

Every handler returns a plain dict and raises a domain exception on failure;
translation to a status code happens here and nowhere else, so the domain layer
never imports the web framework.
"""

from decimal import Decimal

from orderbook.catalog import unit_price
from orderbook.inventory import Inventory, OutOfStock
from orderbook.pricing import PricingError, quote

INVENTORY = Inventory(on_hand={"SKU-1": 10, "SKU-2": 4, "SKU-3": 0})


def post_quote(payload: dict) -> tuple[int, dict]:
    lines = []
    for item in payload.get("items", []):
        try:
            price = unit_price(item["sku"])
        except LookupError as exc:
            return 404, {"error": str(exc)}
        lines.append((item["sku"], int(item["quantity"]), Decimal(price)))
    try:
        return 200, quote(lines)
    except PricingError as exc:
        return 400, {"error": str(exc)}


def post_reserve(payload: dict) -> tuple[int, dict]:
    try:
        for item in payload.get("items", []):
            INVENTORY.reserve(item["sku"], int(item["quantity"]))
    except OutOfStock as exc:
        return 409, {"error": str(exc)}
    return 200, {"reserved": True}
''',
    "tests/test_pricing.py": """from decimal import Decimal

from orderbook.pricing import PricingError, quote

import pytest


def test_small_order_pays_shipping():
    result = quote([("SKU-1", 1, Decimal("19.99"))])
    assert result["shipping"] == Decimal("4.90")


def test_large_order_ships_free():
    result = quote([("SKU-2", 2, Decimal("45.00"))])
    assert result["shipping"] == Decimal("0.00")


def test_empty_cart_is_an_error():
    with pytest.raises(PricingError):
        quote([])
""",
}

#: (message, {path: content}) — the history the plugin seeds memory from. The
#: messages carry the rationale on purpose: that is the material a later "why do
#: we ..." turn should be able to retrieve instead of re-deriving.
_COMMITS: list[tuple[str, list[str]]] = [
    ("initial commit: package skeleton and README", ["README.md", "orderbook/__init__.py"]),
    ("feat(config): pricing knobs in one frozen dataclass", ["orderbook/config.py"]),
    (
        "feat(pricing): quote a cart\n\n"
        "Money is Decimal throughout, never float: a float subtotal drifted by a cent on carts "
        "with three or more lines and the totals stopped matching the invoices.",
        ["orderbook/pricing.py"],
    ),
    (
        "feat(catalog): sku lookup\n\n"
        "No cache here on purpose. Flash sales change prices mid-session and a stale price is a "
        "customer-visible error; the lookup is one indexed read, so caching buys nothing worth "
        "that risk.",
        ["orderbook/catalog.py"],
    ),
    ("feat(inventory): in-memory reservations", ["orderbook/inventory.py"]),
    (
        "feat(api): quote and reserve endpoints\n\n"
        "Convention: handlers raise domain exceptions and this layer alone maps them to status "
        "codes, so nothing under orderbook/ imports the web framework.",
        ["orderbook/api.py"],
    ),
    ("test(pricing): shipping threshold cases", ["tests/test_pricing.py"]),
]


def write_fixture_repo(path: Path) -> Path:
    """Materialise the fixture repo with its git history. Returns the path."""
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init", "-q", "-b", "main")
    _git(path, "config", "user.email", "evals@hindsight.local")
    _git(path, "config", "user.name", "Hindsight Evals")
    for message, files in _COMMITS:
        for name in files:
            target = path / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(_FILES[name], encoding="utf-8")
        _git(path, "add", *files)
        _git(path, "commit", "-q", "-m", message)
    return path


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        # A developer's global hooks/templates must not run inside the fixture.
        env={"PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": str(cwd), "GIT_CONFIG_NOSYSTEM": "1"},
    )
