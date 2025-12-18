"""Benchmark Visualizer - FastHTML App

A fast web interface for visualizing benchmark results.
Supports LoComo and LongMemEval benchmark visualization.

Usage:
    python main.py
"""

import json
from pathlib import Path
from typing import Any

from fasthtml.common import *

# Get the benchmarks directory
BENCHMARKS_DIR = Path(__file__).resolve().parent.parent


# Tailwind + shadcn/ui theme
def get_head():
    return (
        Script(src="https://cdn.tailwindcss.com"),
        Script("""
            tailwind.config = {
                theme: {
                    extend: {
                        colors: {
                            border: "hsl(214.3 31.8% 91.4%)",
                            input: "hsl(214.3 31.8% 91.4%)",
                            ring: "hsl(222.2 84% 4.9%)",
                            background: "hsl(0 0% 100%)",
                            foreground: "hsl(222.2 84% 4.9%)",
                            primary: {
                                DEFAULT: "hsl(222.2 47.4% 11.2%)",
                                foreground: "hsl(210 40% 98%)",
                            },
                            secondary: {
                                DEFAULT: "hsl(210 40% 96.1%)",
                                foreground: "hsl(222.2 47.4% 11.2%)",
                            },
                            destructive: {
                                DEFAULT: "hsl(0 84.2% 60.2%)",
                                foreground: "hsl(210 40% 98%)",
                            },
                            muted: {
                                DEFAULT: "hsl(210 40% 96.1%)",
                                foreground: "hsl(215.4 16.3% 46.9%)",
                            },
                            accent: {
                                DEFAULT: "hsl(210 40% 96.1%)",
                                foreground: "hsl(222.2 47.4% 11.2%)",
                            },
                            success: {
                                DEFAULT: "hsl(142.1 76.2% 36.3%)",
                                foreground: "hsl(355.7 100% 97.3%)",
                            },
                        },
                        borderRadius: {
                            lg: "0.5rem",
                            md: "calc(0.5rem - 2px)",
                            sm: "calc(0.5rem - 4px)",
                        }
                    }
                }
            }
        """),
        Style("""
            @layer base {
                * { border-color: hsl(var(--border)); }
                body {
                    background-color: hsl(210 40% 96.1%);
                    color: hsl(222.2 84% 4.9%);
                }
            }
        """),
    )


# Create FastHTML app
app, rt = fast_app()


def load_locomo_results(mode: str = "search") -> dict[str, Any] | None:
    """Load LoComo benchmark results."""
    filename = "benchmark_results_think.json" if mode == "think" else "benchmark_results.json"
    results_path = BENCHMARKS_DIR / "locomo" / "results" / filename

    if not results_path.exists():
        return None

    try:
        with open(results_path) as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None


def load_longmemeval_results() -> dict[str, Any] | None:
    """Load LongMemEval benchmark results."""
    results_path = BENCHMARKS_DIR / "longmemeval" / "results" / "benchmark_results.json"

    if not results_path.exists():
        return None

    try:
        with open(results_path) as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None


def get_category_name(category: int | str) -> str:
    """Map category ID to name for LoComo."""
    if isinstance(category, str):
        return category

    categories = {1: "Multi-hop", 2: "Single-hop", 3: "Temporal", 4: "Open-domain"}
    return categories.get(category, "Unknown")


@rt("/")
def get():
    """Main page."""
    return (
        Title("Benchmark Visualizer"),
        get_head(),
        Main(
            Div(
                H1("📊 Benchmark Visualizer", cls="text-4xl font-bold text-foreground"),
                P("Analyze and visualize benchmark results", cls="text-muted-foreground mt-2"),
                cls="text-center py-12",
            ),
            Div(
                Label("Select a benchmark to view:", cls="block text-sm font-medium text-foreground mb-2"),
                Select(
                    Option("-- Choose a benchmark --", value="", selected=True),
                    Option("LoComo (search mode)", value="/locomo/search"),
                    Option("LoComo (think mode)", value="/locomo/think"),
                    Option("LongMemEval", value="/longmemeval"),
                    onchange="if(this.value) window.location.href = this.value;",
                    cls="w-full px-3 py-2 border border-input rounded-md bg-background text-foreground focus:outline-none focus:ring-2 focus:ring-ring",
                ),
                cls="max-w-md mx-auto bg-white border border-border rounded-lg p-6 shadow-sm",
            ),
            cls="container mx-auto max-w-7xl px-4 py-8",
        ),
    )


@rt("/locomo/{mode}")
def get_locomo(mode: str, filter_type: str = "all", category_filter: str = "all"):
    """Render LoComo results."""
    data = load_locomo_results(mode)

    if not data:
        mode_label = "think" if mode == "think" else "search"
        extra_args = " --use-think" if mode == "think" else ""
        return (
            Title(f"LoComo ({mode_label}) - Not Found"),
            get_head(),
            Main(
                H1("⚠️ Benchmark Results Not Found", cls="text-3xl font-bold text-foreground mb-4"),
                P(f"The {mode_label} mode results are not available.", cls="text-muted-foreground mb-6"),
                H4("To generate results:", cls="text-lg font-semibold text-foreground mb-2"),
                Pre(
                    f"./scripts/benchmarks/run-locomo.sh --env local{extra_args}",
                    cls="bg-slate-900 text-slate-100 p-4 rounded-md overflow-x-auto text-sm",
                ),
                A(
                    "← Back",
                    href="/",
                    cls="inline-flex items-center mt-6 px-4 py-2 bg-primary text-primary-foreground rounded-md hover:bg-primary/90 text-sm font-medium",
                ),
                cls="container mx-auto max-w-7xl px-4 py-8",
            ),
        )

    all_results = data.get("item_results", data.get("conversation_results", []))

    # Filter items based on their questions
    results = []
    for item in all_results:
        detailed_results = item.get("metrics", {}).get("detailed_results", [])

        # Apply correctness filter
        passes_correctness_filter = False
        if filter_type == "all":
            passes_correctness_filter = True
        elif filter_type == "correct":
            # Show items where all questions are correct (and not invalid)
            passes_correctness_filter = detailed_results and all(
                r.get("is_correct") and not r.get("is_invalid") for r in detailed_results
            )
        elif filter_type == "incorrect":
            # Show items that have at least one incorrect question
            passes_correctness_filter = any(
                not r.get("is_correct") and not r.get("is_invalid") for r in detailed_results
            )
        elif filter_type == "invalid":
            # Show items that have at least one invalid question
            passes_correctness_filter = any(r.get("is_invalid") for r in detailed_results)

        # Apply category filter
        passes_category_filter = False
        if category_filter == "all":
            passes_category_filter = True
        else:
            # Show items that have at least one question of the specified category
            category_id = int(category_filter)
            passes_category_filter = any(r.get("category") == category_id for r in detailed_results)

        if passes_correctness_filter and passes_category_filter:
            results.append(item)

    # Calculate stats (use all_results for overall stats, not filtered results)
    category_stats = {
        1: {"name": "Multi-hop", "correct": 0, "total": 0, "invalid": 0},
        2: {"name": "Single-hop", "correct": 0, "total": 0, "invalid": 0},
        3: {"name": "Temporal", "correct": 0, "total": 0, "invalid": 0},
        4: {"name": "Open-domain", "correct": 0, "total": 0, "invalid": 0},
    }

    total_invalid = 0
    for item in all_results:
        if item.get("metrics", {}).get("detailed_results"):
            for result in item["metrics"]["detailed_results"]:
                category = result.get("category")
                if category in category_stats:
                    category_stats[category]["total"] += 1
                    if result.get("is_invalid"):
                        category_stats[category]["invalid"] += 1
                        total_invalid += 1
                    elif result.get("is_correct"):
                        category_stats[category]["correct"] += 1

    mode_label = " (Think Mode)" if mode == "think" else " (Search Mode)"

    # Overall stats
    stats_html = Div(
        H3(f"LoComo Benchmark{mode_label} - Overall Performance", cls="text-2xl font-bold text-foreground mb-6"),
        Div(
            Div(
                P("Overall Accuracy", cls="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-2"),
                P(f"{data['overall_accuracy']:.2f}%", cls="text-3xl font-bold text-foreground"),
                cls="bg-white border border-border rounded-lg p-6 text-center shadow-sm",
            ),
            Div(
                P("Correct Answers", cls="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-2"),
                P(f"{data['total_correct']} / {data['total_questions']}", cls="text-3xl font-bold text-foreground"),
                cls="bg-white border border-border rounded-lg p-6 text-center shadow-sm",
            ),
            Div(
                P("Invalid Questions", cls="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-2"),
                P(str(total_invalid), cls="text-3xl font-bold text-foreground"),
                cls="bg-white border border-border rounded-lg p-6 text-center shadow-sm",
            )
            if total_invalid > 0
            else None,
            Div(
                P("Items", cls="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-2"),
                P(str(len(all_results)), cls="text-3xl font-bold text-foreground"),
                cls="bg-white border border-border rounded-lg p-6 text-center shadow-sm",
            ),
            cls="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8",
        ),
        H4("Accuracy by Category", cls="text-xl font-semibold text-foreground mb-4"),
        Div(
            *[
                Div(
                    P(cat["name"], cls="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-2"),
                    P(
                        f"{(cat['correct'] / (cat['total'] - cat['invalid']) * 100) if (cat['total'] - cat['invalid']) > 0 else 0:.1f}%",
                        cls="text-2xl font-bold text-foreground",
                    ),
                    P(f"{cat['correct']} / {cat['total']}", cls="text-sm text-muted-foreground mt-1"),
                    cls="bg-white border border-border rounded-lg p-6 text-center shadow-sm",
                )
                for cat in category_stats.values()
            ],
            cls="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4",
        ),
    )

    # Filter controls
    filters = Div(
        # Correctness filter
        Div(
            P("Filter by correctness:", cls="text-sm font-medium text-foreground mb-2"),
            Div(
                A(
                    "All",
                    href=f"/locomo/{mode}?filter_type=all&category_filter={category_filter}",
                    cls="px-3 py-1.5 rounded-md text-sm font-medium "
                    + (
                        "bg-primary text-primary-foreground"
                        if filter_type == "all"
                        else "bg-white text-foreground border border-border hover:bg-accent"
                    ),
                ),
                A(
                    "✅ All Correct",
                    href=f"/locomo/{mode}?filter_type=correct&category_filter={category_filter}",
                    cls="px-3 py-1.5 rounded-md text-sm font-medium "
                    + (
                        "bg-primary text-primary-foreground"
                        if filter_type == "correct"
                        else "bg-white text-foreground border border-border hover:bg-accent"
                    ),
                ),
                A(
                    "❌ Has Incorrect",
                    href=f"/locomo/{mode}?filter_type=incorrect&category_filter={category_filter}",
                    cls="px-3 py-1.5 rounded-md text-sm font-medium "
                    + (
                        "bg-primary text-primary-foreground"
                        if filter_type == "incorrect"
                        else "bg-white text-foreground border border-border hover:bg-accent"
                    ),
                ),
                A(
                    "⚠️ Has Invalid",
                    href=f"/locomo/{mode}?filter_type=invalid&category_filter={category_filter}",
                    cls="px-3 py-1.5 rounded-md text-sm font-medium "
                    + (
                        "bg-primary text-primary-foreground"
                        if filter_type == "invalid"
                        else "bg-white text-foreground border border-border hover:bg-accent"
                    ),
                )
                if total_invalid > 0
                else None,
                cls="flex flex-wrap gap-2",
            ),
            cls="mb-4",
        ),
        # Category filter
        Div(
            P("Filter by question category:", cls="text-sm font-medium text-foreground mb-2"),
            Div(
                A(
                    "All Categories",
                    href=f"/locomo/{mode}?filter_type={filter_type}&category_filter=all",
                    cls="px-3 py-1.5 rounded-md text-sm font-medium "
                    + (
                        "bg-primary text-primary-foreground"
                        if category_filter == "all"
                        else "bg-white text-foreground border border-border hover:bg-accent"
                    ),
                ),
                A(
                    "Multi-hop",
                    href=f"/locomo/{mode}?filter_type={filter_type}&category_filter=1",
                    cls="px-3 py-1.5 rounded-md text-sm font-medium "
                    + (
                        "bg-primary text-primary-foreground"
                        if category_filter == "1"
                        else "bg-white text-foreground border border-border hover:bg-accent"
                    ),
                ),
                A(
                    "Single-hop",
                    href=f"/locomo/{mode}?filter_type={filter_type}&category_filter=2",
                    cls="px-3 py-1.5 rounded-md text-sm font-medium "
                    + (
                        "bg-primary text-primary-foreground"
                        if category_filter == "2"
                        else "bg-white text-foreground border border-border hover:bg-accent"
                    ),
                ),
                A(
                    "Temporal",
                    href=f"/locomo/{mode}?filter_type={filter_type}&category_filter=3",
                    cls="px-3 py-1.5 rounded-md text-sm font-medium "
                    + (
                        "bg-primary text-primary-foreground"
                        if category_filter == "3"
                        else "bg-white text-foreground border border-border hover:bg-accent"
                    ),
                ),
                A(
                    "Open-domain",
                    href=f"/locomo/{mode}?filter_type={filter_type}&category_filter=4",
                    cls="px-3 py-1.5 rounded-md text-sm font-medium "
                    + (
                        "bg-primary text-primary-foreground"
                        if category_filter == "4"
                        else "bg-white text-foreground border border-border hover:bg-accent"
                    ),
                ),
                cls="flex flex-wrap gap-2",
            ),
            cls="mb-4",
        ),
        cls="mb-6",
    )

    # Render items
    items_html = []
    for item in results:
        # Find the original index in all_results
        original_idx = all_results.index(item)
        item_id = item.get("item_id", item.get("sample_id", f"item-{original_idx}"))
        metrics = item.get("metrics", {})

        # Calculate accuracy for filtered category
        if category_filter != "all":
            detailed_results = metrics.get("detailed_results", [])
            category_id = int(category_filter)
            filtered_correct = sum(
                1
                for r in detailed_results
                if r.get("category") == category_id and r.get("is_correct") and not r.get("is_invalid")
            )
            filtered_total = sum(
                1 for r in detailed_results if r.get("category") == category_id and not r.get("is_invalid")
            )
            accuracy = (filtered_correct / filtered_total * 100) if filtered_total > 0 else 0
            correct = filtered_correct
            total = filtered_total
        else:
            accuracy = metrics.get("accuracy", 0)
            correct = metrics.get("correct", 0)
            total = metrics.get("total", 0)

        color = "🟢" if accuracy >= 70 else ("🟡" if accuracy >= 50 else "🔴")

        # Determine border color based on accuracy
        if accuracy >= 70:
            border_class = "border-l-4 border-green-600"
            bg_class = "hover:bg-green-50"
        elif accuracy >= 50:
            border_class = "border-l-4 border-yellow-500"
            bg_class = "hover:bg-yellow-50"
        else:
            border_class = "border-l-4 border-red-600"
            bg_class = "hover:bg-red-50"

        # Show preview with link to detail page
        items_html.append(
            A(
                Div(
                    Div(
                        P(f"{color} {item_id}", cls="text-lg font-semibold text-foreground mb-2"),
                        Div(
                            Div(
                                P("Accuracy", cls="text-xs font-medium text-muted-foreground uppercase tracking-wide"),
                                P(f"{accuracy:.1f}%", cls="text-2xl font-bold text-foreground"),
                                cls="text-center",
                            ),
                            Div(
                                P("Correct", cls="text-xs font-medium text-muted-foreground uppercase tracking-wide"),
                                P(f"{correct}/{total}", cls="text-xl font-semibold text-foreground"),
                                cls="text-center",
                            ),
                            cls="flex gap-6 items-center",
                        ),
                        cls="p-6",
                    ),
                    cls=f"bg-white border border-border rounded-lg shadow-sm transition-all {border_class} {bg_class}",
                ),
                href=f"/locomo/{mode}/item/{original_idx}?filter_type={filter_type}&category_filter={category_filter}",
                cls="block no-underline",
            )
        )

    return (
        Title(f"LoComo ({mode_label})"),
        get_head(),
        Main(
            Div(
                A(
                    "← Back to benchmarks",
                    href="/",
                    cls="inline-flex items-center px-4 py-2 bg-white border border-border rounded-md text-sm font-medium text-foreground hover:bg-accent mb-6",
                ),
                stats_html,
                Hr(cls="my-6 border-border"),
                filters,
                P(f"Showing {len(results)} of {len(all_results)} items", cls="text-sm text-muted-foreground mb-6"),
                Div(*items_html, cls="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4"),
                cls="container mx-auto max-w-7xl px-4 py-8",
            )
        ),
    )


@rt("/locomo/{mode}/item/{item_idx}")
def get_locomo_item(mode: str, item_idx: int, filter_type: str = "all", category_filter: str = "all"):
    """Render a single LoComo item with questions."""
    data = load_locomo_results(mode)
    if not data:
        return Redirect("/")

    results = data.get("item_results", data.get("conversation_results", []))
    if item_idx >= len(results):
        return Redirect(f"/locomo/{mode}")

    item = results[item_idx]
    item_id = item.get("item_id", item.get("sample_id", f"item-{item_idx}"))
    metrics = item.get("metrics", {})
    accuracy = metrics.get("accuracy", 0)
    correct = metrics.get("correct", 0)
    total = metrics.get("total", 0)
    invalid = metrics.get("invalid", 0)
    detailed_results = metrics.get("detailed_results", [])
    category_stats_raw = metrics.get("category_stats", {})

    # Filter questions
    filtered_questions = []
    for q_idx, result in enumerate(detailed_results):
        is_invalid = result.get("is_invalid", False)
        is_correct = result.get("is_correct", False)
        question_category = result.get("category")

        # Apply correctness filter
        passes_correctness = False
        if filter_type == "all":
            passes_correctness = True
        elif filter_type == "correct" and is_correct and not is_invalid:
            passes_correctness = True
        elif filter_type == "incorrect" and not is_correct and not is_invalid:
            passes_correctness = True
        elif filter_type == "invalid" and is_invalid:
            passes_correctness = True

        # Apply category filter
        passes_category = False
        if category_filter == "all":
            passes_category = True
        elif category_filter.isdigit() and question_category == int(category_filter):
            passes_category = True

        if passes_correctness and passes_category:
            filtered_questions.append((q_idx, result))

    # Build category stats for this item
    category_stats = {
        1: {"name": "Multi-hop", "correct": 0, "total": 0, "invalid": 0},
        2: {"name": "Single-hop", "correct": 0, "total": 0, "invalid": 0},
        3: {"name": "Temporal", "correct": 0, "total": 0, "invalid": 0},
        4: {"name": "Open-domain", "correct": 0, "total": 0, "invalid": 0},
    }

    for cat_id_str, stats in category_stats_raw.items():
        cat_id = int(cat_id_str)
        if cat_id in category_stats:
            category_stats[cat_id]["correct"] = stats.get("correct", 0)
            category_stats[cat_id]["total"] = stats.get("total", 0)
            category_stats[cat_id]["invalid"] = stats.get("invalid", 0)

    # Overall stats for this item
    mode_label = " (Think Mode)" if mode == "think" else " (Search Mode)"
    stats_html = Div(
        H3(f"{item_id}{mode_label} - Performance", cls="text-2xl font-bold text-foreground mb-6"),
        Div(
            Div(
                P("Overall Accuracy", cls="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-2"),
                P(f"{accuracy:.2f}%", cls="text-3xl font-bold text-foreground"),
                cls="bg-white border border-border rounded-lg p-6 text-center shadow-sm",
            ),
            Div(
                P("Correct Answers", cls="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-2"),
                P(f"{correct} / {total}", cls="text-3xl font-bold text-foreground"),
                cls="bg-white border border-border rounded-lg p-6 text-center shadow-sm",
            ),
            Div(
                P("Invalid Questions", cls="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-2"),
                P(str(invalid), cls="text-3xl font-bold text-foreground"),
                cls="bg-white border border-border rounded-lg p-6 text-center shadow-sm",
            )
            if invalid > 0
            else None,
            Div(
                P("Total Questions", cls="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-2"),
                P(str(total), cls="text-3xl font-bold text-foreground"),
                cls="bg-white border border-border rounded-lg p-6 text-center shadow-sm",
            ),
            cls="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8",
        ),
        H4("Accuracy by Category", cls="text-xl font-semibold text-foreground mb-4"),
        Div(
            *[
                Div(
                    P(cat["name"], cls="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-2"),
                    P(
                        f"{(cat['correct'] / (cat['total'] - cat['invalid']) * 100) if (cat['total'] - cat['invalid']) > 0 else 0:.1f}%",
                        cls="text-2xl font-bold text-foreground",
                    ),
                    P(f"{cat['correct']} / {cat['total']}", cls="text-sm text-muted-foreground mt-1"),
                    cls="bg-white border border-border rounded-lg p-6 text-center shadow-sm",
                )
                for cat in category_stats.values()
                if cat["total"] > 0
            ],
            cls="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4",
        )
        if any(cat["total"] > 0 for cat in category_stats.values())
        else None,
        cls="mb-6",
    )

    # Generate markdown table for copying
    markdown_rows = [f"| {item_id} | {accuracy:.1f}% |"]
    for cat in category_stats.values():
        if cat["total"] > 0:
            cat_accuracy = (
                (cat["correct"] / (cat["total"] - cat["invalid"]) * 100) if (cat["total"] - cat["invalid"]) > 0 else 0
            )
            markdown_rows.append(f" {cat_accuracy:.1f}% |")

    markdown_table = f"""| Conversation | Overall |{" | ".join([cat["name"] for cat in category_stats.values() if cat["total"] > 0])} |
|---|---|{" | ".join(["---" for cat in category_stats.values() if cat["total"] > 0])} |
{"".join(markdown_rows)}"""

    # Copy button
    copy_button = Div(
        Button(
            "📋 Copy Stats Table",
            onclick=f"""
                navigator.clipboard.writeText(`{markdown_table}`).then(() => {{
                    this.textContent = '✓ Copied!';
                    setTimeout(() => {{ this.textContent = '📋 Copy Stats Table'; }}, 2000);
                }});
            """,
            cls="px-4 py-2 bg-primary text-primary-foreground rounded-md hover:bg-primary/90 text-sm font-medium cursor-pointer",
        ),
        cls="mb-6",
    )

    # Filters for questions
    has_invalid = any(r.get("is_invalid", False) for r in detailed_results)
    q_filters = Div(
        # Correctness filter
        Div(
            P("Filter by correctness:", cls="text-sm font-medium text-foreground mb-2"),
            Div(
                A(
                    "All",
                    href=f"/locomo/{mode}/item/{item_idx}?filter_type=all&category_filter={category_filter}",
                    cls="px-3 py-1.5 rounded-md text-sm font-medium "
                    + (
                        "bg-primary text-primary-foreground"
                        if filter_type == "all"
                        else "bg-white text-foreground border border-border hover:bg-accent"
                    ),
                ),
                A(
                    "✅ Correct",
                    href=f"/locomo/{mode}/item/{item_idx}?filter_type=correct&category_filter={category_filter}",
                    cls="px-3 py-1.5 rounded-md text-sm font-medium "
                    + (
                        "bg-primary text-primary-foreground"
                        if filter_type == "correct"
                        else "bg-white text-foreground border border-border hover:bg-accent"
                    ),
                ),
                A(
                    "❌ Incorrect",
                    href=f"/locomo/{mode}/item/{item_idx}?filter_type=incorrect&category_filter={category_filter}",
                    cls="px-3 py-1.5 rounded-md text-sm font-medium "
                    + (
                        "bg-primary text-primary-foreground"
                        if filter_type == "incorrect"
                        else "bg-white text-foreground border border-border hover:bg-accent"
                    ),
                ),
                A(
                    "⚠️ Invalid",
                    href=f"/locomo/{mode}/item/{item_idx}?filter_type=invalid&category_filter={category_filter}",
                    cls="px-3 py-1.5 rounded-md text-sm font-medium "
                    + (
                        "bg-primary text-primary-foreground"
                        if filter_type == "invalid"
                        else "bg-white text-foreground border border-border hover:bg-accent"
                    ),
                )
                if has_invalid
                else None,
                cls="flex flex-wrap gap-2",
            ),
            cls="mb-4",
        ),
        # Category filter
        Div(
            P("Filter by category:", cls="text-sm font-medium text-foreground mb-2"),
            Div(
                A(
                    "All Categories",
                    href=f"/locomo/{mode}/item/{item_idx}?filter_type={filter_type}&category_filter=all",
                    cls="px-3 py-1.5 rounded-md text-sm font-medium "
                    + (
                        "bg-primary text-primary-foreground"
                        if category_filter == "all"
                        else "bg-white text-foreground border border-border hover:bg-accent"
                    ),
                ),
                A(
                    "Multi-hop",
                    href=f"/locomo/{mode}/item/{item_idx}?filter_type={filter_type}&category_filter=1",
                    cls="px-3 py-1.5 rounded-md text-sm font-medium "
                    + (
                        "bg-primary text-primary-foreground"
                        if category_filter == "1"
                        else "bg-white text-foreground border border-border hover:bg-accent"
                    ),
                )
                if any(cat_id == 1 for cat_id in category_stats.keys() if category_stats[cat_id]["total"] > 0)
                else None,
                A(
                    "Single-hop",
                    href=f"/locomo/{mode}/item/{item_idx}?filter_type={filter_type}&category_filter=2",
                    cls="px-3 py-1.5 rounded-md text-sm font-medium "
                    + (
                        "bg-primary text-primary-foreground"
                        if category_filter == "2"
                        else "bg-white text-foreground border border-border hover:bg-accent"
                    ),
                )
                if any(cat_id == 2 for cat_id in category_stats.keys() if category_stats[cat_id]["total"] > 0)
                else None,
                A(
                    "Temporal",
                    href=f"/locomo/{mode}/item/{item_idx}?filter_type={filter_type}&category_filter=3",
                    cls="px-3 py-1.5 rounded-md text-sm font-medium "
                    + (
                        "bg-primary text-primary-foreground"
                        if category_filter == "3"
                        else "bg-white text-foreground border border-border hover:bg-accent"
                    ),
                )
                if any(cat_id == 3 for cat_id in category_stats.keys() if category_stats[cat_id]["total"] > 0)
                else None,
                A(
                    "Open-domain",
                    href=f"/locomo/{mode}/item/{item_idx}?filter_type={filter_type}&category_filter=4",
                    cls="px-3 py-1.5 rounded-md text-sm font-medium "
                    + (
                        "bg-primary text-primary-foreground"
                        if category_filter == "4"
                        else "bg-white text-foreground border border-border hover:bg-accent"
                    ),
                )
                if any(cat_id == 4 for cat_id in category_stats.keys() if category_stats[cat_id]["total"] > 0)
                else None,
                cls="flex flex-wrap gap-2",
            ),
            cls="mb-4",
        ),
        cls="mb-6",
    )

    # Render questions
    questions_html = []
    for q_idx, result in filtered_questions:
        is_invalid = result.get("is_invalid", False)
        is_correct = result.get("is_correct", False)
        question = result.get("question", "")
        correct_answer = result.get("correct_answer", "")
        predicted_answer = result.get("predicted_answer", "")
        category = get_category_name(result.get("category", "Unknown"))

        icon = "⚠️" if is_invalid else ("✅" if is_correct else "❌")
        border_class = (
            "border-l-4 border-yellow-500"
            if is_invalid
            else ("border-l-4 border-green-600" if is_correct else "border-l-4 border-red-600")
        )

        questions_html.append(
            Div(
                # Header
                Div(
                    P(f"{icon} Question {q_idx + 1}", cls="text-lg font-semibold text-foreground"),
                    P(f"Category: {category}", cls="text-sm text-muted-foreground"),
                    cls="mb-4",
                ),
                # Question
                Div(
                    P("Question:", cls="text-sm font-medium text-foreground mb-1"),
                    P(question, cls="text-foreground"),
                    cls="mb-4",
                ),
                # Answers side by side
                Div(
                    Div(
                        P("✓ Correct Answer", cls="text-sm font-medium text-foreground mb-2"),
                        Div(correct_answer, cls="bg-green-50 border border-green-200 rounded-md p-3 text-foreground"),
                        cls="flex-1",
                    ),
                    Div(
                        P(
                            f"{'✓' if is_correct else '✗'} Predicted Answer",
                            cls="text-sm font-medium text-foreground mb-2",
                        ),
                        Div(
                            predicted_answer,
                            cls="border rounded-md p-3 text-foreground "
                            + ("bg-green-50 border-green-200" if is_correct else "bg-red-50 border-red-200"),
                        ),
                        cls="flex-1",
                    ),
                    cls="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4",
                ),
                # Details
                Details(
                    Summary(
                        "📝 Show Reasoning & Retrieved Memories",
                        cls="cursor-pointer font-medium text-foreground hover:text-primary py-2",
                    ),
                    Div(
                        # System Reasoning
                        Div(
                            P("System Reasoning:", cls="text-sm font-medium text-foreground mb-2"),
                            Pre(
                                result.get("reasoning", "N/A"),
                                cls="bg-slate-900 text-slate-100 p-3 rounded-md overflow-x-auto text-sm",
                            ),
                            cls="mb-4",
                        ),
                        # Judge Reasoning
                        Div(
                            P("Judge Reasoning:", cls="text-sm font-medium text-foreground mb-2"),
                            Pre(
                                result.get("correctness_reasoning", "N/A"),
                                cls="bg-slate-900 text-slate-100 p-3 rounded-md overflow-x-auto text-sm",
                            ),
                            cls="mb-4",
                        ),
                        # Retrieved Memories
                        Div(
                            P(
                                f"Retrieved Memories ({len(result.get('retrieved_memories', []))}):",
                                cls="text-sm font-medium text-foreground mb-2",
                            ),
                            *[
                                Div(
                                    P(
                                        f"#{i + 1} • "
                                        + " • ".join(
                                            filter(
                                                None,
                                                [
                                                    f"Occurred: {mem.get('occurred_start', '')[:10]}"
                                                    + (
                                                        f" to {mem.get('occurred_end', '')[:10]}"
                                                        if mem.get("occurred_end")
                                                        and mem.get("occurred_start", "")[:10]
                                                        != mem.get("occurred_end", "")[:10]
                                                        else ""
                                                    )
                                                    if mem.get("occurred_start")
                                                    else None,
                                                    f"Mentioned: {mem.get('mentioned_at', '')[:10]}"
                                                    if mem.get("mentioned_at")
                                                    else None,
                                                    f"Type: {mem.get('fact_type', 'unknown').upper()}",
                                                ],
                                            )
                                        ),
                                        cls="text-xs text-muted-foreground mb-1",
                                    ),
                                    P(mem.get("text", ""), cls="text-sm text-foreground"),
                                    cls="bg-muted/50 border border-border rounded-md p-3 mb-2",
                                )
                                for i, mem in enumerate(result.get("retrieved_memories", []))
                            ]
                            if result.get("retrieved_memories")
                            else [P("No memories retrieved", cls="text-sm text-muted-foreground")],
                        ),
                        cls="mt-3 space-y-2",
                    ),
                    cls="border border-border rounded-md p-4 bg-muted/30",
                ),
                cls=f"bg-white border border-border rounded-lg p-6 mb-4 shadow-sm {border_class}",
            )
        )

    return (
        Title(f"{item_id}"),
        get_head(),
        Main(
            Div(
                A(
                    f"← Back to LoComo ({mode})",
                    href=f"/locomo/{mode}",
                    cls="inline-flex items-center px-4 py-2 bg-white border border-border rounded-md text-sm font-medium text-foreground hover:bg-accent mb-6",
                ),
                stats_html,
                copy_button,
                Hr(cls="my-6 border-border"),
                q_filters,
                P(f"Showing {len(filtered_questions)} questions", cls="text-sm text-muted-foreground mb-6"),
                Div(*questions_html, cls="space-y-4"),
                cls="container mx-auto max-w-7xl px-4 py-8",
            )
        ),
    )


@rt("/longmemeval")
def get_longmemeval(filter_type: str = "all", category_filter: str = "all"):
    """Render LongMemEval results."""
    data = load_longmemeval_results()

    if not data:
        return (
            Title("LongMemEval - Not Found"),
            get_head(),
            Main(
                H1("⚠️ Benchmark Results Not Found"),
                P("The benchmark results are not available."),
                H4("To generate results:"),
                Pre("./scripts/benchmarks/run-longmemeval.sh --env local"),
                A("← Back", href="/", cls="btn mt-3"),
                cls="container",
            ),
        )

    all_results = data.get("item_results", [])

    # Filter items based on their questions
    results = []
    for item in all_results:
        detailed_results = item.get("metrics", {}).get("detailed_results", [])

        # Apply correctness filter
        passes_correctness_filter = False
        if filter_type == "all":
            passes_correctness_filter = True
        elif filter_type == "correct":
            # Show items where all questions are correct (and not invalid)
            passes_correctness_filter = detailed_results and all(
                r.get("is_correct") and not r.get("is_invalid") for r in detailed_results
            )
        elif filter_type == "incorrect":
            # Show items that have at least one incorrect question
            passes_correctness_filter = any(
                not r.get("is_correct") and not r.get("is_invalid") for r in detailed_results
            )
        elif filter_type == "invalid":
            # Show items that have at least one invalid question
            passes_correctness_filter = any(r.get("is_invalid") for r in detailed_results)

        # Apply category filter
        passes_category_filter = False
        if category_filter == "all":
            passes_category_filter = True
        else:
            # Show items that have at least one question of the specified category
            passes_category_filter = any(r.get("category") == category_filter for r in detailed_results)

        if passes_correctness_filter and passes_category_filter:
            results.append(item)

    # Calculate stats (use all_results for overall stats, not filtered results)
    category_stats = {}
    total_invalid = 0

    for item in all_results:
        if item.get("metrics", {}).get("category_stats"):
            for category, stats in item["metrics"]["category_stats"].items():
                if category not in category_stats:
                    category_stats[category] = {"name": category, "correct": 0, "total": 0, "invalid": 0}
                category_stats[category]["correct"] += stats.get("correct", 0)
                category_stats[category]["total"] += stats.get("total", 0)
                category_stats[category]["invalid"] += stats.get("invalid", 0)

        if item.get("metrics", {}).get("detailed_results"):
            for result in item["metrics"]["detailed_results"]:
                if result.get("is_invalid"):
                    total_invalid += 1

    # Overall stats
    stats_html = Div(
        H3("LongMemEval Benchmark - Overall Performance", cls="text-2xl font-bold text-foreground mb-6"),
        Div(
            Div(
                P("Overall Accuracy", cls="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-2"),
                P(f"{data['overall_accuracy']:.2f}%", cls="text-3xl font-bold text-foreground"),
                cls="bg-white border border-border rounded-lg p-6 text-center shadow-sm",
            ),
            Div(
                P("Correct Answers", cls="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-2"),
                P(f"{data['total_correct']} / {data['total_questions']}", cls="text-3xl font-bold text-foreground"),
                cls="bg-white border border-border rounded-lg p-6 text-center shadow-sm",
            ),
            Div(
                P("Invalid Questions", cls="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-2"),
                P(str(total_invalid), cls="text-3xl font-bold text-foreground"),
                cls="bg-white border border-border rounded-lg p-6 text-center shadow-sm",
            )
            if total_invalid > 0
            else None,
            Div(
                P("Items", cls="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-2"),
                P(str(len(all_results)), cls="text-3xl font-bold text-foreground"),
                cls="bg-white border border-border rounded-lg p-6 text-center shadow-sm",
            ),
            cls="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8",
        ),
        H4("Accuracy by Category", cls="text-xl font-semibold text-foreground mb-4"),
        Div(
            *[
                Div(
                    P(cat["name"], cls="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-2"),
                    P(
                        f"{(cat['correct'] / (cat['total'] - cat['invalid']) * 100) if (cat['total'] - cat['invalid']) > 0 else 0:.1f}%",
                        cls="text-2xl font-bold text-foreground",
                    ),
                    P(f"{cat['correct']} / {cat['total']}", cls="text-sm text-muted-foreground mt-1"),
                    cls="bg-white border border-border rounded-lg p-6 text-center shadow-sm",
                )
                for cat in category_stats.values()
            ],
            cls="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4",
        )
        if category_stats
        else None,
    )

    # Filter controls
    filters = Div(
        # Correctness filter
        Div(
            P("Filter by correctness:", cls="text-sm font-medium text-foreground mb-2"),
            Div(
                A(
                    "All",
                    href=f"/longmemeval?filter_type=all&category_filter={category_filter}",
                    cls="px-3 py-1.5 rounded-md text-sm font-medium "
                    + (
                        "bg-primary text-primary-foreground"
                        if filter_type == "all"
                        else "bg-white text-foreground border border-border hover:bg-accent"
                    ),
                ),
                A(
                    "✅ All Correct",
                    href=f"/longmemeval?filter_type=correct&category_filter={category_filter}",
                    cls="px-3 py-1.5 rounded-md text-sm font-medium "
                    + (
                        "bg-primary text-primary-foreground"
                        if filter_type == "correct"
                        else "bg-white text-foreground border border-border hover:bg-accent"
                    ),
                ),
                A(
                    "❌ Has Incorrect",
                    href=f"/longmemeval?filter_type=incorrect&category_filter={category_filter}",
                    cls="px-3 py-1.5 rounded-md text-sm font-medium "
                    + (
                        "bg-primary text-primary-foreground"
                        if filter_type == "incorrect"
                        else "bg-white text-foreground border border-border hover:bg-accent"
                    ),
                ),
                A(
                    "⚠️ Has Invalid",
                    href=f"/longmemeval?filter_type=invalid&category_filter={category_filter}",
                    cls="px-3 py-1.5 rounded-md text-sm font-medium "
                    + (
                        "bg-primary text-primary-foreground"
                        if filter_type == "invalid"
                        else "bg-white text-foreground border border-border hover:bg-accent"
                    ),
                )
                if total_invalid > 0
                else None,
                cls="flex flex-wrap gap-2",
            ),
            cls="mb-4",
        ),
        # Category filter
        Div(
            P("Filter by question category:", cls="text-sm font-medium text-foreground mb-2"),
            Div(
                A(
                    "All Categories",
                    href=f"/longmemeval?filter_type={filter_type}&category_filter=all",
                    cls="px-3 py-1.5 rounded-md text-sm font-medium "
                    + (
                        "bg-primary text-primary-foreground"
                        if category_filter == "all"
                        else "bg-white text-foreground border border-border hover:bg-accent"
                    ),
                ),
                A(
                    "Multi-session",
                    href=f"/longmemeval?filter_type={filter_type}&category_filter=multi-session",
                    cls="px-3 py-1.5 rounded-md text-sm font-medium "
                    + (
                        "bg-primary text-primary-foreground"
                        if category_filter == "multi-session"
                        else "bg-white text-foreground border border-border hover:bg-accent"
                    ),
                ),
                A(
                    "Single-session User",
                    href=f"/longmemeval?filter_type={filter_type}&category_filter=single-session-user",
                    cls="px-3 py-1.5 rounded-md text-sm font-medium "
                    + (
                        "bg-primary text-primary-foreground"
                        if category_filter == "single-session-user"
                        else "bg-white text-foreground border border-border hover:bg-accent"
                    ),
                ),
                A(
                    "Single-session Assistant",
                    href=f"/longmemeval?filter_type={filter_type}&category_filter=single-session-assistant",
                    cls="px-3 py-1.5 rounded-md text-sm font-medium "
                    + (
                        "bg-primary text-primary-foreground"
                        if category_filter == "single-session-assistant"
                        else "bg-white text-foreground border border-border hover:bg-accent"
                    ),
                ),
                A(
                    "Single-session Preference",
                    href=f"/longmemeval?filter_type={filter_type}&category_filter=single-session-preference",
                    cls="px-3 py-1.5 rounded-md text-sm font-medium "
                    + (
                        "bg-primary text-primary-foreground"
                        if category_filter == "single-session-preference"
                        else "bg-white text-foreground border border-border hover:bg-accent"
                    ),
                ),
                A(
                    "Temporal Reasoning",
                    href=f"/longmemeval?filter_type={filter_type}&category_filter=temporal-reasoning",
                    cls="px-3 py-1.5 rounded-md text-sm font-medium "
                    + (
                        "bg-primary text-primary-foreground"
                        if category_filter == "temporal-reasoning"
                        else "bg-white text-foreground border border-border hover:bg-accent"
                    ),
                ),
                A(
                    "Knowledge Update",
                    href=f"/longmemeval?filter_type={filter_type}&category_filter=knowledge-update",
                    cls="px-3 py-1.5 rounded-md text-sm font-medium "
                    + (
                        "bg-primary text-primary-foreground"
                        if category_filter == "knowledge-update"
                        else "bg-white text-foreground border border-border hover:bg-accent"
                    ),
                ),
                cls="flex flex-wrap gap-2",
            ),
            cls="mb-4",
        ),
        cls="mb-6",
    )

    # Render items
    items_html = []
    for item in results:
        # Find the original index in all_results
        original_idx = all_results.index(item)
        item_id = item.get("item_id", f"item-{original_idx}")
        metrics = item.get("metrics", {})
        accuracy = metrics.get("accuracy", 0)
        correct = metrics.get("correct", 0)
        total = metrics.get("total", 0)

        color = "🟢" if accuracy >= 70 else ("🟡" if accuracy >= 50 else "🔴")

        # Determine border color based on accuracy
        if accuracy >= 70:
            border_class = "border-l-4 border-green-600"
            bg_class = "hover:bg-green-50"
        elif accuracy >= 50:
            border_class = "border-l-4 border-yellow-500"
            bg_class = "hover:bg-yellow-50"
        else:
            border_class = "border-l-4 border-red-600"
            bg_class = "hover:bg-red-50"

        items_html.append(
            A(
                Div(
                    Div(
                        P(f"{color} {item_id}", cls="text-lg font-semibold text-foreground mb-2"),
                        Div(
                            Div(
                                P("Accuracy", cls="text-xs font-medium text-muted-foreground uppercase tracking-wide"),
                                P(f"{accuracy:.1f}%", cls="text-2xl font-bold text-foreground"),
                                cls="text-center",
                            ),
                            Div(
                                P("Correct", cls="text-xs font-medium text-muted-foreground uppercase tracking-wide"),
                                P(f"{correct}/{total}", cls="text-xl font-semibold text-foreground"),
                                cls="text-center",
                            ),
                            cls="flex gap-6 items-center",
                        ),
                        cls="p-6",
                    ),
                    cls=f"bg-white border border-border rounded-lg shadow-sm transition-all {border_class} {bg_class}",
                ),
                href=f"/longmemeval/item/{original_idx}?filter_type={filter_type}&category_filter={category_filter}",
                cls="block no-underline",
            )
        )

    return (
        Title("LongMemEval"),
        get_head(),
        Main(
            Div(
                A(
                    "← Back to benchmarks",
                    href="/",
                    cls="inline-flex items-center px-4 py-2 bg-white border border-border rounded-md text-sm font-medium text-foreground hover:bg-accent mb-6",
                ),
                stats_html,
                Hr(cls="my-6 border-border"),
                filters,
                P(f"Showing {len(results)} of {len(all_results)} items", cls="text-sm text-muted-foreground mb-6"),
                Div(*items_html, cls="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4"),
                cls="container mx-auto max-w-7xl px-4 py-8",
            )
        ),
    )


@rt("/longmemeval/item/{item_idx}")
def get_longmemeval_item(item_idx: int, filter_type: str = "all"):
    """Render a single LongMemEval item with questions."""
    data = load_longmemeval_results()
    if not data:
        return Redirect("/")

    results = data.get("item_results", [])
    if item_idx >= len(results):
        return Redirect("/longmemeval")

    item = results[item_idx]
    item_id = item.get("item_id", f"item-{item_idx}")
    metrics = item.get("metrics", {})
    accuracy = metrics.get("accuracy", 0)
    detailed_results = metrics.get("detailed_results", [])

    # Filter questions
    filtered_questions = []
    for q_idx, result in enumerate(detailed_results):
        is_invalid = result.get("is_invalid", False)
        is_correct = result.get("is_correct", False)

        if filter_type == "all":
            filtered_questions.append((q_idx, result))
        elif filter_type == "correct" and is_correct and not is_invalid:
            filtered_questions.append((q_idx, result))
        elif filter_type == "incorrect" and not is_correct and not is_invalid:
            filtered_questions.append((q_idx, result))
        elif filter_type == "invalid" and is_invalid:
            filtered_questions.append((q_idx, result))

    # Filters for questions
    has_invalid = any(r.get("is_invalid", False) for r in detailed_results)
    q_filters = Div(
        P("Filter:", cls="text-sm font-medium text-foreground mb-2"),
        Div(
            A(
                "All",
                href=f"/longmemeval/item/{item_idx}?filter_type=all",
                cls="px-3 py-1.5 rounded-md text-sm font-medium "
                + (
                    "bg-primary text-primary-foreground"
                    if filter_type == "all"
                    else "bg-white text-foreground border border-border hover:bg-accent"
                ),
            ),
            A(
                "✅ Correct",
                href=f"/longmemeval/item/{item_idx}?filter_type=correct",
                cls="px-3 py-1.5 rounded-md text-sm font-medium "
                + (
                    "bg-primary text-primary-foreground"
                    if filter_type == "correct"
                    else "bg-white text-foreground border border-border hover:bg-accent"
                ),
            ),
            A(
                "❌ Incorrect",
                href=f"/longmemeval/item/{item_idx}?filter_type=incorrect",
                cls="px-3 py-1.5 rounded-md text-sm font-medium "
                + (
                    "bg-primary text-primary-foreground"
                    if filter_type == "incorrect"
                    else "bg-white text-foreground border border-border hover:bg-accent"
                ),
            ),
            A(
                "⚠️ Invalid",
                href=f"/longmemeval/item/{item_idx}?filter_type=invalid",
                cls="px-3 py-1.5 rounded-md text-sm font-medium "
                + (
                    "bg-primary text-primary-foreground"
                    if filter_type == "invalid"
                    else "bg-white text-foreground border border-border hover:bg-accent"
                ),
            )
            if has_invalid
            else None,
            cls="flex flex-wrap gap-2",
        ),
        cls="mb-6",
    )

    # Render questions
    questions_html = []
    for q_idx, result in filtered_questions:
        is_invalid = result.get("is_invalid", False)
        is_correct = result.get("is_correct", False)
        question = result.get("question", "")
        correct_answer = result.get("correct_answer", "")
        predicted_answer = result.get("predicted_answer", "")
        category = result.get("category", "Unknown")

        icon = "⚠️" if is_invalid else ("✅" if is_correct else "❌")
        border_class = (
            "border-l-4 border-yellow-500"
            if is_invalid
            else ("border-l-4 border-green-600" if is_correct else "border-l-4 border-red-600")
        )

        questions_html.append(
            Div(
                # Header
                Div(
                    P(f"{icon} Question {q_idx + 1}", cls="text-lg font-semibold text-foreground"),
                    P(f"Category: {category}", cls="text-sm text-muted-foreground"),
                    cls="mb-4",
                ),
                # Question
                Div(
                    P("Question:", cls="text-sm font-medium text-foreground mb-1"),
                    P(question, cls="text-foreground"),
                    cls="mb-4",
                ),
                # Answers side by side
                Div(
                    Div(
                        P("✓ Correct Answer", cls="text-sm font-medium text-foreground mb-2"),
                        Div(correct_answer, cls="bg-green-50 border border-green-200 rounded-md p-3 text-foreground"),
                        cls="flex-1",
                    ),
                    Div(
                        P(
                            f"{'✓' if is_correct else '✗'} Predicted Answer",
                            cls="text-sm font-medium text-foreground mb-2",
                        ),
                        Div(
                            predicted_answer,
                            cls="border rounded-md p-3 text-foreground "
                            + ("bg-green-50 border-green-200" if is_correct else "bg-red-50 border-red-200"),
                        ),
                        cls="flex-1",
                    ),
                    cls="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4",
                ),
                # Details
                Details(
                    Summary(
                        "📝 Show Reasoning & Retrieved Memories",
                        cls="cursor-pointer font-medium text-foreground hover:text-primary py-2",
                    ),
                    Div(
                        # System Reasoning
                        Div(
                            P("System Reasoning:", cls="text-sm font-medium text-foreground mb-2"),
                            Pre(
                                result.get("reasoning", "N/A"),
                                cls="bg-slate-900 text-slate-100 p-3 rounded-md overflow-x-auto text-sm",
                            ),
                            cls="mb-4",
                        ),
                        # Judge Reasoning
                        Div(
                            P("Judge Reasoning:", cls="text-sm font-medium text-foreground mb-2"),
                            Pre(
                                result.get("correctness_reasoning", "N/A"),
                                cls="bg-slate-900 text-slate-100 p-3 rounded-md overflow-x-auto text-sm",
                            ),
                            cls="mb-4",
                        ),
                        # Retrieved Memories
                        Div(
                            P(
                                f"Retrieved Memories ({len(result.get('retrieved_memories', []))}):",
                                cls="text-sm font-medium text-foreground mb-2",
                            ),
                            *[
                                Div(
                                    P(
                                        f"#{i + 1} • "
                                        + " • ".join(
                                            filter(
                                                None,
                                                [
                                                    f"Occurred: {mem.get('occurred_start', '')[:10]}"
                                                    + (
                                                        f" to {mem.get('occurred_end', '')[:10]}"
                                                        if mem.get("occurred_end")
                                                        and mem.get("occurred_start", "")[:10]
                                                        != mem.get("occurred_end", "")[:10]
                                                        else ""
                                                    )
                                                    if mem.get("occurred_start")
                                                    else None,
                                                    f"Mentioned: {mem.get('mentioned_at', '')[:10]}"
                                                    if mem.get("mentioned_at")
                                                    else None,
                                                    f"Type: {mem.get('fact_type', 'unknown').upper()}",
                                                ],
                                            )
                                        ),
                                        cls="text-xs text-muted-foreground mb-1",
                                    ),
                                    P(mem.get("text", ""), cls="text-sm text-foreground"),
                                    cls="bg-muted/50 border border-border rounded-md p-3 mb-2",
                                )
                                for i, mem in enumerate(result.get("retrieved_memories", []))
                            ]
                            if result.get("retrieved_memories")
                            else [P("No memories retrieved", cls="text-sm text-muted-foreground")],
                        ),
                        cls="mt-3 space-y-2",
                    ),
                    cls="border border-border rounded-md p-4 bg-muted/30",
                ),
                cls=f"bg-white border border-border rounded-lg p-6 mb-4 shadow-sm {border_class}",
            )
        )

    return (
        Title(f"{item_id}"),
        get_head(),
        Main(
            Div(
                A(
                    "← Back to LongMemEval",
                    href="/longmemeval",
                    cls="inline-flex items-center px-4 py-2 bg-white border border-border rounded-md text-sm font-medium text-foreground hover:bg-accent mb-6",
                ),
                H3(f"📊 {item_id} - {accuracy:.2f}%", cls="text-2xl font-bold text-foreground mb-4"),
                Hr(cls="my-6 border-border"),
                q_filters,
                P(f"Showing {len(filtered_questions)} questions", cls="text-sm text-muted-foreground mb-6"),
                Div(*questions_html, cls="space-y-4"),
                cls="container mx-auto max-w-7xl px-4 py-8",
            )
        ),
    )


if __name__ == "__main__":
    import uvicorn

    print("🚀 Starting Benchmark Visualizer...")
    print("📊 Server running at: http://localhost:8001")
    uvicorn.run("main:app", host="127.0.0.1", port=8001, reload=True)
