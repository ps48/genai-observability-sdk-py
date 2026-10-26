# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
#
# Modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

"""Benchmark tracking for GenAI evaluation results.

Upload evaluation results from any harness (RAGAS, DeepEval, pytest, custom)
as OTel ``test_suite_run`` / ``test_case`` spans with ``test.*`` attributes,
plus one ``gen_ai.evaluation.result`` log event per score. Results flow via
OTLP to OpenSearch for analysis and comparison (spans to
``otel-v1-apm-span-*``, evaluation events to ``otel-v1-genai-evaluation``).

Two modes of use:

**Direct upload** — you already have results::

    with Benchmark(name="rag-v2", metadata={"model": "gpt-4.1"}) as b:
        b.log(input="What is Python?", output="A language",
              scores={"accuracy": 1.0})

**Full eval runner** — SDK runs your task and scorers::

    result = evaluate(
        name="rag-v2",
        task=my_agent,
        data=[{"input": "What is Python?", "expected": "programming language"}],
        scores=[accuracy_scorer],
    )

Telemetry shape::

    span "test_suite_run <suite>"  test.suite.name, test.suite.run.id, test.suite.run.status
      span "test_case <case>"      test.case.id, test.case.name, test.case.result.status,
                                   opensearch.test_case.input / .expected / .output (record_io)
        ... agent spans (evaluate() runs the task inside the case span)
        LogRecord gen_ai.evaluation.result (parented to the test_case span, or to the
                  linked agent span), with test.suite.name, test.suite.run.id,
                  test.case.id and test.case.name copied as attributes

- ``test.*`` follows the proposed test conventions
  (semantic-conventions-genai#79, semantic-conventions#1262).
- ``gen_ai.evaluation.result`` follows the GenAI events spec
  (https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-events/).
- Test case content lives under ``opensearch.test_case.*``; ``test.*`` is
  owned by the OTel CICD SIG and has no names for it.

Set ``OPENSEARCH_GENAI_EVAL_CARRIER=span`` (deprecated) to put scores on the
case span as span events instead, or ``both`` while migrating.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import time
import uuid
from collections import defaultdict
from collections.abc import Callable, Iterable
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import Link, SpanContext, SpanKind, TraceFlags

from opensearch_genai_observability_sdk_py._internal import (
    _RESERVED_KEYS,
    parse_hex,
    validate_metadata_keys,
)
from opensearch_genai_observability_sdk_py.score import (
    EVALUATION_EVENT_NAME,
    EXPLANATION_MAX_CHARS,
    build_evaluation_attributes,
    emit_evaluation_event,
    resolve_carrier,
    span_context_for,
    warn_span_carrier_deprecated,
)

logger = logging.getLogger(__name__)

_TRACER_NAME = "opensearch-genai-observability-sdk-py-benchmarks"

_IO_TRUNCATION_LIMIT = 10_000
_TRUNCATION_MARKER = "...[truncated]"


# ---------------------------------------------------------------------------
# EvalScore dataclass — return type for scorer functions used with evaluate()
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class EvalScore:
    """Result from a scorer function.

    Returned by scorer callables passed to ``evaluate()``.
    """

    name: str
    value: float
    label: str | None = None
    explanation: str | None = None
    metadata: dict[str, Any] | None = None


@dataclasses.dataclass
class _FailedScore:
    """Internal stand-in for a scorer that raised (no score value)."""

    name: str
    value: float | None = None
    label: str | None = None
    explanation: str | None = None
    metadata: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ScoreSummary:
    name: str
    avg: float
    min: float
    max: float
    count: int


@dataclasses.dataclass
class BenchmarkSummary:
    benchmark_name: str
    run_id: str
    total_cases: int
    error_count: int
    duration_ms: float
    scores: dict[str, ScoreSummary]

    def __str__(self) -> str:
        width = 60
        lines = [
            "=" * width,
            f"  BENCHMARK: {self.benchmark_name}",
            f"  Run: {self.run_id} | Cases: {self.total_cases}"
            f" | Errors: {self.error_count}"
            f" | Duration: {self.duration_ms / 1000:.1f}s",
            "",
        ]
        for s in self.scores.values():
            lines.append(
                f"  {s.name:<20s} avg={s.avg:.2f}  min={s.min:.2f}  max={s.max:.2f}  (n={s.count})"
            )
        lines.append("=" * width)
        return "\n".join(lines)


@dataclasses.dataclass
class TestCaseResult:
    case_id: str
    case_name: str | None
    input: Any
    output: Any
    expected: Any
    scores: dict[str, float]
    error: str | None
    status: str
    scorer_errors: list[str] | None = None


@dataclasses.dataclass
class BenchmarkResult:
    summary: BenchmarkSummary
    cases: list[TestCaseResult]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_case_id(input_val: Any) -> str:
    """Derive a stable case ID from the input value."""
    content = json.dumps(input_val, sort_keys=True, default=str)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _make_run_id() -> str:
    """Generate a unique run ID with timestamp prefix."""
    ts = time.strftime("%Y%m%d_%H%M%S")
    short_uuid = uuid.uuid4().hex[:8]
    return f"run_{ts}_{short_uuid}"


def _truncate_io(value: str) -> str:
    """Truncate a serialized IO string, appending a marker if truncated."""
    if len(value) <= _IO_TRUNCATION_LIMIT:
        return value
    return value[: _IO_TRUNCATION_LIMIT - len(_TRUNCATION_MARKER)] + _TRUNCATION_MARKER


def _build_span_link(trace_id_hex: str, span_id_hex: str) -> Link | None:
    """Build an OTel span link from hex trace/span IDs.

    Returns ``None`` if either value fails to parse.
    """
    trace_id_int = parse_hex(trace_id_hex)
    if trace_id_int is None:
        logger.warning("Invalid trace_id '%s', skipping span link", trace_id_hex)
        return None

    span_id_int = parse_hex(span_id_hex)
    if span_id_int is None:
        logger.warning("Invalid span_id '%s', skipping span link", span_id_hex)
        return None

    ctx = SpanContext(
        trace_id=trace_id_int,
        span_id=span_id_int,
        is_remote=True,
        trace_flags=TraceFlags(TraceFlags.SAMPLED),
    )
    return Link(ctx)


_GROUPING_KEYS = ("test.suite.name", "test.suite.run.id", "test.case.id", "test.case.name")


def _case_span_name(case_name: str | None, case_id: str) -> str:
    return f"test_case {case_name or case_id}"


def _emit_scores(
    case_span: trace.Span,
    case_attrs: dict[str, Any],
    scores: Iterable[EvalScore | _FailedScore],
    *,
    carrier: str,
    event_context: Any = None,
    evaluator: str | None = None,
    error_type: str | None = None,
) -> None:
    """Record scores as ``gen_ai.evaluation.result`` events.

    ``carrier="event"`` (default) emits one log-based event per score, parented
    to ``event_context`` (the evaluated span) or the case span, with the
    ``test.*`` grouping attributes copied. ``"span"`` keeps the deprecated
    span events on the case span; ``"both"`` does both.
    """
    grouping = {k: case_attrs[k] for k in _GROUPING_KEYS if k in case_attrs}
    ctx = event_context or trace.set_span_in_context(case_span)

    for s in scores:
        extra = dict(grouping)
        if s.metadata:
            extra.update(_safe_metadata(s.metadata, context=f"EvalScore({s.name!r})"))
        attrs = build_evaluation_attributes(
            name=s.name,
            value=s.value,
            label=s.label,
            explanation=s.explanation,
            explanation_limit=EXPLANATION_MAX_CHARS,
            level="span",
            evaluator=evaluator,
            error_type=error_type,
            extra=extra,
        )
        if carrier in ("event", "both"):
            emit_evaluation_event(attrs, context=ctx, evaluator=evaluator)
        if carrier in ("span", "both"):
            warn_span_carrier_deprecated()
            case_span.add_event(EVALUATION_EVENT_NAME, attributes=attrs)


def _safe_metadata(metadata: dict[str, Any], *, context: str = "") -> dict[str, Any]:
    """Filter out SDK-reserved keys from user metadata and warn."""
    validate_metadata_keys(metadata, context=context)
    return {k: v for k, v in metadata.items() if k not in _RESERVED_KEYS}


def _compute_summary(
    benchmark_name: str,
    run_id: str,
    cases: list[TestCaseResult],
    start_time: float,
) -> BenchmarkSummary:
    """Compute aggregate score statistics from case records."""
    score_values: dict[str, list[float]] = defaultdict(list)
    error_count = 0

    for case in cases:
        if case.error:
            error_count += 1
        for name, value in case.scores.items():
            score_values[name].append(value)

    score_summaries: dict[str, ScoreSummary] = {}
    for name, values in score_values.items():
        score_summaries[name] = ScoreSummary(
            name=name,
            avg=sum(values) / len(values),
            min=min(values),
            max=max(values),
            count=len(values),
        )

    return BenchmarkSummary(
        benchmark_name=benchmark_name,
        run_id=run_id,
        total_cases=len(cases),
        error_count=error_count,
        duration_ms=(time.monotonic() - start_time) * 1000,
        scores=score_summaries,
    )


# ---------------------------------------------------------------------------
# Benchmark class — direct upload (Mode B)
# ---------------------------------------------------------------------------


class Benchmark:
    """Upload evaluation results as OTel benchmark spans.

    Creates a root ``test_suite_run`` span and child ``test_case`` spans
    for each ``log()`` call. Results are exported via the configured OTel
    pipeline to OpenSearch.

    Use as a context manager::

        with Benchmark(name="rag-v2") as b:
            b.log(input="q1", output="a1", scores={"accuracy": 1.0})

    Or manually::

        b = Benchmark(name="rag-v2")
        b.log(input="q1", output="a1", scores={"accuracy": 1.0})
        summary = b.close()

    Args:
        name: Benchmark name (``test.suite.name``). Stable across runs.
        metadata: Arbitrary metadata attached to the root span.
            Keys starting with ``test.`` or ``gen_ai.`` are reserved
            and will be filtered out with a warning.
        record_io: If ``True``, record input/output/expected on the
            ``test_case`` span as ``opensearch.test_case.input`` /
            ``.output`` / ``.expected``. Default ``False`` (PII-safe).
    """

    def __init__(
        self,
        name: str,
        *,
        metadata: dict[str, Any] | None = None,
        record_io: bool = False,
    ) -> None:
        self._name = name
        self._metadata = _safe_metadata(metadata or {}, context=f"Benchmark({name!r})")
        self._record_io = record_io
        self._run_id = _make_run_id()
        self._start_time = time.monotonic()
        self._cases: list[TestCaseResult] = []
        self._closed = False
        self._carrier = resolve_carrier(None)

        tracer = trace.get_tracer(_TRACER_NAME)

        root_attrs: dict[str, Any] = {
            "test.suite.name": name,
            "test.suite.run.id": self._run_id,
        }
        for key, val in self._metadata.items():
            root_attrs[key] = val

        self._root_span = tracer.start_span(
            f"test_suite_run {name}",
            kind=SpanKind.INTERNAL,
            attributes=root_attrs,
        )
        self._root_context = trace.set_span_in_context(self._root_span)

        logger.info("Benchmark started: name=%s run_id=%s", name, self._run_id)

    @property
    def name(self) -> str:
        return self._name

    @property
    def run_id(self) -> str:
        return self._run_id

    def log(
        self,
        *,
        input: Any = None,
        output: Any = None,
        expected: Any = None,
        scores: dict[str, float] | None = None,
        metadata: dict[str, Any] | None = None,
        error: str | None = None,
        case_id: str | None = None,
        case_name: str | None = None,
        trace_id: str | None = None,
        span_id: str | None = None,
    ) -> None:
        """Log a single test case result.

        Creates a ``test_case <case>`` span as a child of the benchmark
        root, with ``test.case.*`` attributes, and one
        ``gen_ai.evaluation.result`` log event per score.

        Args:
            input: Test case input data.
            output: Actual output from the agent/system.
            expected: Expected/ground-truth output.
            scores: Pre-computed scores (name -> float).
            metadata: Arbitrary case-level metadata. Keys starting with
                ``test.`` or ``gen_ai.`` are reserved and will be filtered.
            error: Error message. Sets status to ``"fail"``.
            case_id: Explicit ``test.case.id``. If omitted, derived
                from a hash of ``input``.
            case_name: Human-readable ``test.case.name``.
            trace_id: Hex trace ID to link to an existing agent trace.
            span_id: Hex span ID to link to a specific agent span.
                Required when ``trace_id`` is provided. The case span links
                to it and the score events are parented to it.
        """
        if self._closed:
            raise RuntimeError("Benchmark is already closed")

        scores = scores or {}
        resolved_case_id = case_id or _make_case_id(input)

        # Determine pass/fail
        if error:
            status = "fail"
        else:
            status = "pass"

        # Build span attributes
        case_attrs: dict[str, Any] = {
            "test.suite.name": self._name,
            "test.suite.run.id": self._run_id,
            "test.case.id": resolved_case_id,
            "test.case.result.status": status,
        }
        if case_name:
            case_attrs["test.case.name"] = case_name
        if error:
            case_attrs["opensearch.test_case.error"] = error
        if metadata:
            safe = _safe_metadata(metadata, context=f"Benchmark.log(case_id={resolved_case_id!r})")
            for key, val in safe.items():
                case_attrs[key] = val

        # Record IO if opted in
        if self._record_io:
            if input is not None:
                case_attrs["opensearch.test_case.input"] = _truncate_io(
                    json.dumps(input, default=str)
                )
            if output is not None:
                case_attrs["opensearch.test_case.output"] = _truncate_io(
                    json.dumps(output, default=str)
                )
            if expected is not None:
                case_attrs["opensearch.test_case.expected"] = _truncate_io(
                    json.dumps(expected, default=str)
                )

        # Build span link if trace context provided
        if bool(trace_id) != bool(span_id):
            raise ValueError("Both trace_id and span_id must be provided together, or neither.")
        links: list[Link] = []
        evaluated_ctx = None
        if trace_id and span_id:
            link = _build_span_link(trace_id, span_id)
            if link:
                links.append(link)
                evaluated_ctx = span_context_for(trace_id, span_id)

        # Create the case span as a child of the root
        tracer = trace.get_tracer(_TRACER_NAME)
        case_span = tracer.start_span(
            _case_span_name(case_name, resolved_case_id),
            context=self._root_context,
            kind=SpanKind.INTERNAL,
            attributes=case_attrs,
            links=links or None,
        )

        # One gen_ai.evaluation.result event per score
        _emit_scores(
            case_span,
            case_attrs,
            [EvalScore(name=n, value=v) for n, v in scores.items()],
            carrier=self._carrier,
            event_context=evaluated_ctx,
        )

        # Set error status if applicable
        if error:
            case_span.set_status(trace.StatusCode.ERROR, error)

        case_span.end()

        # Track for summary
        self._cases.append(
            TestCaseResult(
                case_id=resolved_case_id,
                case_name=case_name,
                input=input,
                output=output,
                expected=expected,
                scores=scores,
                error=error,
                status=status,
            )
        )

    def close(self) -> BenchmarkSummary:
        """Finalize the benchmark and return a summary.

        Ends the root span, computes aggregate statistics, and logs
        a formatted summary.

        Returns:
            ``BenchmarkSummary`` with aggregate score statistics.
        """
        if self._closed:
            raise RuntimeError("Benchmark is already closed")

        self._closed = True

        summary = _compute_summary(self._name, self._run_id, self._cases, self._start_time)

        # Set final status on root span
        has_errors = summary.error_count > 0
        run_status = "failure" if has_errors else "success"
        self._root_span.set_attribute("test.suite.run.status", run_status)
        if has_errors:
            self._root_span.set_status(
                trace.StatusCode.ERROR,
                f"{summary.error_count} case(s) failed",
            )

        self._root_span.end()

        print(str(summary))
        logger.info("Benchmark completed: %s", self._run_id)

        return summary

    def __enter__(self) -> Benchmark:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if not self._closed:
            self.close()


# ---------------------------------------------------------------------------
# evaluate() — full eval runner (Mode A)
# ---------------------------------------------------------------------------


def evaluate(
    name: str,
    *,
    task: Callable[[Any], Any],
    data: list[dict[str, Any]] | Iterable[dict[str, Any]],
    scores: list[Callable[..., EvalScore | list[EvalScore] | float]],
    metadata: dict[str, Any] | None = None,
    record_io: bool = False,
) -> BenchmarkResult:
    """Run a task against a dataset, score outputs, and record results.

    Creates benchmark spans where agent execution spans are children
    of each case span, giving full trace waterfall per case.

    Args:
        name: Benchmark name (``test.suite.name``).
        task: Callable that takes the input and returns output. Should
            be decorated with ``@observe()`` for full tracing.
        data: Iterable of dicts with ``"input"`` and optionally
            ``"expected"`` and ``"case_id"`` keys. Consumed once
            (single-pass); generators are materialized into a list.
        scores: List of scorer callables. Each receives
            ``(input, output, expected)`` and returns an ``EvalScore``,
            list of ``EvalScore``, or a plain float.
        metadata: Attached to the root benchmark span. Keys starting
            with ``test.`` or ``gen_ai.`` are reserved and will be
            filtered out with a warning.
        record_io: Record input/output/expected on the ``test_case`` span
            as ``opensearch.test_case.input`` / ``.output`` / ``.expected``.

    Returns:
        ``BenchmarkResult`` with summary and per-case results.
    """
    run_id = _make_run_id()
    start_time = time.monotonic()
    tracer = trace.get_tracer(_TRACER_NAME)
    carrier = resolve_carrier(None)

    safe_meta = _safe_metadata(metadata or {}, context=f"evaluate({name!r})")

    root_attrs: dict[str, Any] = {
        "test.suite.name": name,
        "test.suite.run.id": run_id,
    }
    for key, val in safe_meta.items():
        root_attrs[key] = val

    root_span = tracer.start_span(
        f"test_suite_run {name}",
        kind=SpanKind.INTERNAL,
        attributes=root_attrs,
    )
    root_context = trace.set_span_in_context(root_span)

    case_results: list[TestCaseResult] = []

    # Materialize iterable upfront so generators work
    data_list = list(data)

    for item in data_list:
        input_val = item.get("input")
        expected_val = item.get("expected")
        case_id = item.get("case_id") or _make_case_id(input_val)
        case_name_val = item.get("case_name")

        # Case span attributes
        case_attrs: dict[str, Any] = {
            "test.suite.name": name,
            "test.suite.run.id": run_id,
            "test.case.id": case_id,
        }
        if case_name_val:
            case_attrs["test.case.name"] = case_name_val

        if record_io:
            if input_val is not None:
                case_attrs["opensearch.test_case.input"] = _truncate_io(
                    json.dumps(input_val, default=str)
                )
            if expected_val is not None:
                case_attrs["opensearch.test_case.expected"] = _truncate_io(
                    json.dumps(expected_val, default=str)
                )

        case_span = tracer.start_span(
            _case_span_name(case_name_val, case_id),
            context=root_context,
            kind=SpanKind.INTERNAL,
            attributes=case_attrs,
        )
        case_context = trace.set_span_in_context(case_span)

        # Run the task inside the case span context
        output_val = None
        error_msg = None
        try:
            token = trace.context_api.attach(case_context)
            try:
                output_val = task(input_val)
            finally:
                trace.context_api.detach(token)
        except Exception as exc:
            error_msg = str(exc)
            case_span.set_status(trace.StatusCode.ERROR, error_msg)
            case_span.record_exception(exc)

        if record_io and output_val is not None:
            case_span.set_attribute(
                "opensearch.test_case.output",
                _truncate_io(json.dumps(output_val, default=str)),
            )

        # Run scorers; each scorer's results become gen_ai.evaluation.result
        # events (evaluator = scorer function name).
        case_scores: dict[str, float] = {}
        scorer_errors: list[str] = []
        if error_msg is None:
            for scorer_fn in scores:
                scorer_name = getattr(scorer_fn, "__name__", str(scorer_fn))
                # Named scorers identify the evaluator; lambdas etc. do not.
                evaluator = scorer_name if scorer_name.isidentifier() else None
                try:
                    result = scorer_fn(input_val, output_val, expected_val)
                except Exception as exc:
                    msg = f"Scorer {scorer_name} failed: {exc}"
                    scorer_errors.append(msg)
                    logger.warning(
                        "Scorer %s failed for case %s: %s",
                        scorer_name,
                        case_id,
                        exc,
                    )
                    # Record the evaluator failure (error.type, no score value).
                    _emit_scores(
                        case_span,
                        case_attrs,
                        [_FailedScore(name=scorer_name, explanation=str(exc))],
                        carrier=carrier,
                        evaluator=evaluator,
                        error_type=type(exc).__qualname__,
                    )
                    continue

                produced: list[EvalScore] = []
                if isinstance(result, EvalScore):
                    produced.append(result)
                elif isinstance(result, list):
                    produced.extend(s for s in result if isinstance(s, EvalScore))
                elif isinstance(result, (int, float)):
                    produced.append(EvalScore(name=scorer_name, value=float(result)))
                for s in produced:
                    case_scores[s.name] = s.value
                _emit_scores(
                    case_span,
                    case_attrs,
                    produced,
                    carrier=carrier,
                    evaluator=evaluator,
                )

        # Determine status — fail if task errored or any scorer errored
        if error_msg:
            status = "fail"
        elif scorer_errors:
            status = "fail"
        else:
            status = "pass"
        case_span.set_attribute("test.case.result.status", status)
        case_span.end()

        case_results.append(
            TestCaseResult(
                case_id=case_id,
                case_name=case_name_val,
                input=input_val,
                output=output_val,
                expected=expected_val,
                scores=case_scores,
                error=error_msg,
                status=status,
                scorer_errors=scorer_errors or None,
            )
        )

    # Finalize root span
    summary = _compute_summary(name, run_id, case_results, start_time)
    has_errors = summary.error_count > 0
    root_span.set_attribute("test.suite.run.status", "failure" if has_errors else "success")
    if has_errors:
        root_span.set_status(
            trace.StatusCode.ERROR,
            f"{summary.error_count} case(s) failed",
        )
    root_span.end()

    print(str(summary))
    logger.info("evaluate() completed: %s", run_id)

    return BenchmarkResult(summary=summary, cases=case_results)
