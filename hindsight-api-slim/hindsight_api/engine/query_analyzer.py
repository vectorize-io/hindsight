"""
Query analysis abstraction for the memory system.

Provides an interface for analyzing natural language queries to extract
structured information like temporal constraints.
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta

from pydantic import BaseModel, Field

from hindsight_api.engine.temporal_periods import (
    NO_TEMPORAL_CONSTRAINT,
    extract_period,
    is_embedded_cjk_dateparser_match,
)

logger = logging.getLogger(__name__)


class TemporalConstraint(BaseModel):
    """
    Temporal constraint extracted from a query.

    Represents a time range with start and end dates.
    """

    start_date: datetime = Field(description="Start of the time range (inclusive)")
    end_date: datetime = Field(description="End of the time range (inclusive)")

    def __str__(self) -> str:
        return f"{self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}"


class QueryAnalysis(BaseModel):
    """
    Result of analyzing a natural language query.

    Contains extracted structured information like temporal constraints.
    """

    temporal_constraint: TemporalConstraint | None = Field(
        default=None, description="Extracted temporal constraint, if any"
    )


class QueryAnalyzer(ABC):
    """
    Abstract base class for query analysis.

    Implementations analyze natural language queries to extract structured
    information like temporal constraints, entities, etc.
    """

    @abstractmethod
    def load(self) -> None:
        """
        Load the query analyzer model.

        This should be called during initialization to load the model
        and avoid cold start latency on first analyze() call.
        """
        pass

    @abstractmethod
    def analyze(self, query: str, reference_date: datetime | None = None) -> QueryAnalysis:
        """
        Analyze a natural language query.

        Args:
            query: Natural language query to analyze
            reference_date: Reference date for relative terms (defaults to now)

        Returns:
            QueryAnalysis containing extracted information
        """
        pass


class DateparserQueryAnalyzer(QueryAnalyzer):
    """
    Query analyzer using dateparser library.

    Uses dateparser to extract temporal expressions from natural language
    queries. Supports 200+ languages including English, Spanish, Italian,
    French, German, etc.

    Performance:
    - ~10-50ms per query
    - No model loading required (lazy import on first use)
    """

    def __init__(self):
        """Initialize dateparser query analyzer."""
        self._search_dates = None

    def load(self) -> None:
        """Load dateparser and warm up internal data structures.

        Triggers the real initialization cost (regex tables, timezone data) at
        load time so the first actual recall doesn't pay the cold-start penalty.
        """
        if self._search_dates is None:
            from dateparser.search import search_dates

            self._search_dates = search_dates
            # Warm up: fire a dummy call to trigger lazy-loaded internal tables.
            self._search_dates("today")

    def analyze(self, query: str, reference_date: datetime | None = None) -> QueryAnalysis:
        """
        Analyze query using dateparser.

        Extracts temporal expressions from the query text. Supports multiple
        languages automatically.

        Args:
            query: Natural language query (any language)
            reference_date: Reference date for relative terms (defaults to now)

        Returns:
            QueryAnalysis with temporal_constraint if found
        """
        if reference_date is None:
            reference_date = datetime.now()

        # Check for period expressions first (these need special handling)
        query_lower = query.lower()
        period_result = extract_period(query_lower, reference_date)
        if period_result is NO_TEMPORAL_CONSTRAINT:
            return QueryAnalysis(temporal_constraint=None)
        if isinstance(period_result, tuple):
            start_date, end_date = period_result
            return QueryAnalysis(temporal_constraint=TemporalConstraint(start_date=start_date, end_date=end_date))

        # Lazy load dateparser (only imports on first call, then cached)
        self.load()

        # Use dateparser's search_dates to find temporal expressions
        settings = {
            "RELATIVE_BASE": reference_date,
            "PREFER_DATES_FROM": "past",
            "RETURN_AS_TIMEZONE_AWARE": False,
        }

        # Wrap dateparser in a defensive try/except. dateparser has been
        # observed to crash with internal errors (e.g., IndexError from
        # locale.translate_search) on certain query inputs. A parser bug
        # should not bring down the whole search/consolidation pipeline —
        # treat any failure as "no temporal constraint found" so the caller
        # can fall back to non-temporal retrieval.
        try:
            results = self._search_dates(query, settings=settings)
        except Exception as e:
            logger.warning(
                "dateparser raised %s on query (treating as no temporal constraint): %s",
                type(e).__name__,
                e,
            )
            return QueryAnalysis(temporal_constraint=None)

        if not results:
            return QueryAnalysis(temporal_constraint=None)

        # Filter out false positives (common words parsed as dates)
        false_positives = {"do", "may", "march", "will", "can", "sat", "sun", "mon", "tue", "wed", "thu", "fri"}
        valid_results = [
            (text, date)
            for text, date in results
            if (text.lower() not in false_positives or len(text) > 3)
            and not is_embedded_cjk_dateparser_match(query, text)
        ]

        if not valid_results:
            return QueryAnalysis(temporal_constraint=None)

        # Use the first valid date found
        _, parsed_date = valid_results[0]

        # Create constraint for single day
        start_date = parsed_date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = parsed_date.replace(hour=23, minute=59, second=59, microsecond=999999)

        return QueryAnalysis(temporal_constraint=TemporalConstraint(start_date=start_date, end_date=end_date))


class TransformerQueryAnalyzer(QueryAnalyzer):
    """
    Query analyzer using T5-based generative models.

    Uses T5 to convert natural language temporal expressions into structured
    date ranges without pattern matching or regex.

    Performance:
    - ~30-80ms on CPU, ~5-15ms on GPU
    - Model size: ~80M params (~300MB download)
    """

    def __init__(self, model_name: str = "google/flan-t5-small", device: str = "cpu"):
        """
        Initialize T5 query analyzer.

        Args:
            model_name: Name of the HuggingFace T5 model to use.
                       Default: google/flan-t5-small (~80M params, ~300MB download)
                       Alternative: google/flan-t5-base (~1GB, more accurate)
            device: Device to run model on ("cpu" or "cuda")
        """
        self.model_name = model_name
        self.device = device
        self._model = None
        self._tokenizer = None

    def load(self) -> None:
        """Load the T5 model for temporal extraction."""
        if self._model is not None:
            return

        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers is required for TransformerQueryAnalyzer. Install it with: pip install transformers"
            )

        logger.info(f"Loading query analyzer model: {self.model_name}...")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self._model.to(self.device)
        self._model.eval()
        logger.info("Query analyzer model loaded")

    def _load_model(self):
        """Lazy load the T5 model for temporal extraction (calls load())."""
        self.load()

    def _extract_with_rules(self, query: str, reference_date: datetime) -> TemporalConstraint | None:
        """
        Extract temporal expressions using rule-based patterns.

        Handles common patterns reliably and fast. Returns None for
        patterns that need model-based extraction.
        """
        import re

        query_lower = query.lower()

        def get_last_weekday(weekday: int) -> datetime:
            days_ago = (reference_date.weekday() - weekday) % 7
            if days_ago == 0:
                days_ago = 7
            return reference_date - timedelta(days=days_ago)

        def constraint(start: datetime, end: datetime) -> TemporalConstraint:
            return TemporalConstraint(
                start_date=start.replace(hour=0, minute=0, second=0, microsecond=0),
                end_date=end.replace(hour=23, minute=59, second=59, microsecond=999999),
            )

        # Yesterday
        if re.search(r"\byesterday\b", query_lower):
            d = reference_date - timedelta(days=1)
            return constraint(d, d)

        # Last week
        if re.search(r"\blast\s+week\b", query_lower):
            start = reference_date - timedelta(days=reference_date.weekday() + 7)
            return constraint(start, start + timedelta(days=6))

        # Last month
        if re.search(r"\blast\s+month\b", query_lower):
            first = reference_date.replace(day=1)
            end = first - timedelta(days=1)
            start = end.replace(day=1)
            return constraint(start, end)

        # Last year
        if re.search(r"\blast\s+year\b", query_lower):
            y = reference_date.year - 1
            return constraint(datetime(y, 1, 1), datetime(y, 12, 31))

        # Last weekend
        if re.search(r"\blast\s+weekend\b", query_lower):
            sat = get_last_weekday(5)
            return constraint(sat, sat + timedelta(days=1))

        # Last <weekday>
        weekdays = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6}
        for name, num in weekdays.items():
            if re.search(rf"\blast\s+{name}\b", query_lower):
                d = get_last_weekday(num)
                return constraint(d, d)

        # Month + Year: "June 2024", "in March 2023"
        months = {
            "january": 1,
            "february": 2,
            "march": 3,
            "april": 4,
            "may": 5,
            "june": 6,
            "july": 7,
            "august": 8,
            "september": 9,
            "october": 10,
            "november": 11,
            "december": 12,
        }
        for name, num in months.items():
            match = re.search(rf"\b{name}\s+(\d{{4}})\b", query_lower)
            if match:
                year = int(match.group(1))
                if num == 12:
                    last_day = 31
                else:
                    last_day = (datetime(year, num + 1, 1) - timedelta(days=1)).day
                return constraint(datetime(year, num, 1), datetime(year, num, last_day))

        return None

    def analyze(self, query: str, reference_date: datetime | None = None) -> QueryAnalysis:
        """
        Analyze query for temporal expressions.

        Uses rule-based extraction for common patterns (fast & reliable),
        falls back to T5 model for complex/unusual patterns.

        Args:
            query: Natural language query
            reference_date: Reference date for relative terms (defaults to now)

        Returns:
            QueryAnalysis with temporal_constraint if found
        """
        if reference_date is None:
            reference_date = datetime.now()

        # Try rule-based extraction first (handles 90%+ of cases)
        result = self._extract_with_rules(query, reference_date)
        if result is not None:
            return QueryAnalysis(temporal_constraint=result)

        # Fall back to T5 model for unusual patterns
        self._load_model()

        # Helper to calculate example dates
        def get_last_weekday(weekday: int) -> datetime:
            days_ago = (reference_date.weekday() - weekday) % 7
            if days_ago == 0:
                days_ago = 7
            return reference_date - timedelta(days=days_ago)

        yesterday = reference_date - timedelta(days=1)
        last_saturday = get_last_weekday(5)

        # Build prompt for T5
        prompt = f"""Today is {reference_date.strftime("%Y-%m-%d")}. Extract date range or "none".

June 2024 = 2024-06-01 to 2024-06-30
yesterday = {yesterday.strftime("%Y-%m-%d")} to {yesterday.strftime("%Y-%m-%d")}
last Saturday = {last_saturday.strftime("%Y-%m-%d")} to {last_saturday.strftime("%Y-%m-%d")}
what is the weather = none
{query} ="""

        # Tokenize and generate
        inputs = self._tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with self._no_grad():
            outputs = self._model.generate(**inputs, max_new_tokens=30, num_beams=3, do_sample=False, temperature=1.0)

        result = self._tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        # Parse the generated output
        temporal = self._parse_generated_output(result, reference_date)
        return QueryAnalysis(temporal_constraint=temporal)

    def _no_grad(self):
        """Get torch.no_grad context manager."""
        try:
            import torch

            return torch.no_grad()
        except ImportError:
            from contextlib import nullcontext

            return nullcontext()

    def _parse_generated_output(self, result: str, reference_date: datetime) -> TemporalConstraint | None:
        """
        Parse T5 generated output into TemporalConstraint.

        Expected format: "YYYY-MM-DD to YYYY-MM-DD"

        Args:
            result: Generated text from T5
            reference_date: Reference date for validation

        Returns:
            TemporalConstraint if valid output, else None
        """
        if not result or result.lower().strip() in ("none", "null", "no"):
            return None

        try:
            # Parse "YYYY-MM-DD to YYYY-MM-DD"
            import re

            pattern = r"(\d{4}-\d{2}-\d{2})\s+to\s+(\d{4}-\d{2}-\d{2})"
            match = re.search(pattern, result, re.IGNORECASE)

            if match:
                start_str = match.group(1)
                end_str = match.group(2)

                start_date = datetime.strptime(start_str, "%Y-%m-%d")
                end_date = datetime.strptime(end_str, "%Y-%m-%d")

                # Set time boundaries
                start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
                end_date = end_date.replace(hour=23, minute=59, second=59, microsecond=999999)

                # Validation
                if end_date < start_date:
                    logger.warning(f"Invalid date range: {start_date} to {end_date}")
                    return None

                return TemporalConstraint(start_date=start_date, end_date=end_date)

        except (ValueError, AttributeError):
            return None

        return None
