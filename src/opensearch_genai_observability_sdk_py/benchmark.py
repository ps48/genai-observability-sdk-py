# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
#
# Modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

"""Benchmark tracking for GenAI evaluation results.

Run offline evaluations, or upload results from any harness (RAGAS, DeepEval,
pytest, custom), as OTel **log events**. Results flow via OTLP logs to
OpenSearch, where Data Prepper routes them to ``otel-v1-genai-evaluation-*``.

Two modes of use:

**Direct upload**: you already have results::

    with Benchmark(name="rag-v2", metadata={"model": "gpt-4.1"}) as b:
        b.log(input="What is Python?", output="A language",
              scores={"accuracy": 1.0})

**Full eval runner**: the SDK runs your task and scorers::

    result = evaluate(
        name="rag-v2",
        task=my_agent,
        data=[{"input": "What is Python?", "expected": "programming language"}],
        scores=[accuracy_scorer],
    )

Telemetry shape::

    trace: the agent run of each case, exactly like production. No harness span.
      root span (first span the task starts)
          test.suite.name, test.suite.run.id, test.case.id, test.case.name
      ... normal GenAI spans

    LogRecord gen_ai.evaluation.result       one per score, parented to the
                                             agent run's root span, test.* copied
    LogRecord opensearch.test_case.result    one per case, parented to the
                                             agent run's root span:
        test.suite.name, test.suite.run.id, test.case.id, test.case.name,
        test.case.result.status (pass | fail | error), error.type,
        opensearch.test_case.input / .expected / .output (record_io),
        opensearch.test_case.duration_ms
    LogRecord opensearch.test_suite_run.result   one per run, no trace context:
        test.suite.name, test.suite.run.id, test.suite.run.status,
        opensearch.test_suite_run.case_count / .pass_count / .fail_count /
        .error_count / .duration_ms

If the task starts no span, the case event and the score events carry no
trace context (the ``test.*`` attributes are still set).

- ``test.*`` follows the proposed test conventions
  (semantic-conventions-genai#79, semantic-conventions#1262).
- ``gen_ai.evaluation.result`` follows the GenAI events spec
  (https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-events/).
- Test case content lives under ``opensearch.test_case.*``; ``test.*`` is
  owned by the OTel CICD SIG and has no names for it.

Scope: score events use the scorer (evaluator) name when it is a named
function, else the SDK scope; case and run events always use the SDK scope
``opensearch-genai-observability-sdk-py`` with the SDK version. Every scope
carries ``opensearch.genai.sdk.name`` / ``opensearch.genai.sdk.version``
scope attributes.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import logging
import threading
import time
import uuid
import weakref
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from contextvars import ContextVar
from typing import Any

from opentelemetry import trace
from opentelemetry._logs import SeverityNumber
from opentelemetry.context import Context
from opentelemetry.sdk.trace import Span as SdkSpan
from opentelemetry.sdk.trace import SpanProcessor
from opentelemetry.trace import NonRecordingSpan, SpanContext

from opensearch_genai_observability_sdk_py._internal import (
    _RESERVED_KEYS,
    get_event_logger,
    validate_metadata_keys,
)
from opensearch_genai_observability_sdk_py.score import (
    _no_span_context,
    build_evaluation_attributes,
    emit_evaluation_event,
    span_context_for,
)

logger = logging.getLogger(__name__)

TEST_CASE_EVENT_NAME = "opensearch.test_case.result"
TEST_SUITE_RUN_EVENT_NAME = "opensearch.test_suite_run.result"

_IO_TRUNCATION_LIMIT = 10_000
_TRUNCATION_MARKER = "...[truncated]"

_CASE_STATUSES = ("pass", "fail", "error")


# ---------------------------------------------------------------------------
# EvalScore dataclass: return type for scorer functions used with evaluate()
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
    """Per-case result.

    ``status`` keeps its historical values (``"pass"`` / ``"fail"``; a task or
    scorer error is ``"fail"``). The ``opensearch.test_case.result`` event
    reports ``pass`` / ``fail`` / ``error`` and ``error_type`` holds the
    exception class name when the task or a scorer raised. ``trace_id`` /
    ``span_id`` (hex) identify the span the case and score events are
    parented to (the agent run's root span), or are ``None`` when the task
    started no span.
    """

    case_id: str
    case_name: str | None
    input: Any
    output: Any
    expected: Any
    scores: dict[str, float]
    error: str | None
    status: str
    scorer_errors: list[str] | None = None
    error_type: str | None = None
    duration_ms: float | None = None
    trace_id: str | None = None
    span_id: str | None = None


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


def _io_attr(value: Any) -> str:
    return _truncate_io(json.dumps(value, default=str))


_GROUPING_KEYS = ("test.suite.name", "test.suite.run.id", "test.case.id", "test.case.name")


def _grouping(suite: str, run_id: str, case_id: str, case_name: str | None) -> dict[str, Any]:
    """The ``test.*`` ids stamped on the agent root span and copied onto events."""
    attrs: dict[str, Any] = {
        "test.suite.name": suite,
        "test.suite.run.id": run_id,
        "test.case.id": case_id,
    }
    if case_name:
        attrs["test.case.name"] = case_name
    return attrs


def _span_ids(ctx: Context) -> tuple[str | None, str | None]:
    sc = trace.get_current_span(ctx).get_span_context()
    if not sc.is_valid:
        return None, None
    return format(sc.trace_id, "032x"), format(sc.span_id, "016x")


# ---------------------------------------------------------------------------
# Agent run scope: stamp test.* on the root span(s) a case's task starts
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _CaseScope:
    """State for the test case whose task is currently running."""

    attributes: dict[str, Any]
    root: SpanContext | None = None
    lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)


_current_case: ContextVar[_CaseScope | None] = ContextVar(
    "opensearch_genai_current_test_case", default=None
)


class _CaseRootStamper(SpanProcessor):
    """Stamps ``test.*`` ids on root spans started while a test case runs.

    A root span here is a span without a local parent (no parent, or a remote
    one). The first such span is captured as the agent run's root: the case
    and score events are parented to it. Spans outside ``evaluate()`` are not
    touched.
    """

    def on_start(self, span: SdkSpan, parent_context: Context | None = None) -> None:
        scope = _current_case.get()
        if scope is None:
            return
        parent = span.parent
        if parent is not None and parent.is_valid and not parent.is_remote:
            return
        for key, val in scope.attributes.items():
            span.set_attribute(key, val)
        with scope.lock:
            if scope.root is None:
                scope.root = span.get_span_context()

    def on_end(self, span: Any) -> None:
        return None

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


_stamped_providers: weakref.WeakSet[Any] = weakref.WeakSet()
_stamper_lock = threading.Lock()


def _ensure_root_stamper() -> None:
    """Install ``_CaseRootStamper`` once on the global SDK TracerProvider."""
    provider = trace.get_tracer_provider()
    add = getattr(provider, "add_span_processor", None)
    if add is None:
        logger.debug(
            "evaluate(): global TracerProvider %r has no add_span_processor; "
            "agent root spans will not be stamped with test.* attributes",
            provider,
        )
        return
    with _stamper_lock:
        if provider in _stamped_providers:
            return
        add(_CaseRootStamper())
        _stamped_providers.add(provider)


@contextlib.contextmanager
def _agent_run_scope(grouping: dict[str, Any]) -> Iterator[_CaseScope]:
    """Run a case's task as its own trace and capture that trace's root span.

    The active span is cleared (so the agent run starts a new trace even if
    ``evaluate()`` is called inside a span); other context values such as
    baggage are kept.
    """
    scope = _CaseScope(attributes=dict(grouping))
    var_token = _current_case.set(scope)
    ctx_token = trace.context_api.attach(trace.set_span_in_context(trace.INVALID_SPAN))
    try:
        yield scope
    finally:
        trace.context_api.detach(ctx_token)
        _current_case.reset(var_token)


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


def _emit_sdk_event(event_name: str, attributes: dict[str, Any], context: Context) -> None:
    """Emit one LogRecord event with the SDK instrumentation scope."""
    otel_logger = get_event_logger()
    otel_logger.emit(
        event_name=event_name,
        timestamp=time.time_ns(),
        severity_number=SeverityNumber.INFO,
        attributes=attributes,
        context=context,
    )


def _emit_case_event(
    grouping: dict[str, Any],
    *,
    status: str,
    context: Context,
    error_type: str | None = None,
    io: dict[str, Any] | None = None,
    duration_ms: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Emit the ``opensearch.test_case.result`` event for one case."""
    attrs: dict[str, Any] = dict(metadata or {})
    attrs.update(grouping)
    attrs["test.case.result.status"] = status
    if error_type:
        attrs["error.type"] = error_type
    if io:
        attrs.update(io)
    if duration_ms is not None:
        attrs["opensearch.test_case.duration_ms"] = float(duration_ms)
    _emit_sdk_event(TEST_CASE_EVENT_NAME, attrs, context)


def _emit_run_event(
    suite: str,
    run_id: str,
    statuses: list[str],
    *,
    duration_ms: float,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Emit the ``opensearch.test_suite_run.result`` event; return the run status."""
    error_count = statuses.count("error")
    run_status = "failure" if error_count else "success"
    attrs: dict[str, Any] = dict(metadata or {})
    attrs.update(
        {
            "test.suite.name": suite,
            "test.suite.run.id": run_id,
            "test.suite.run.status": run_status,
            "opensearch.test_suite_run.case_count": len(statuses),
            "opensearch.test_suite_run.pass_count": statuses.count("pass"),
            "opensearch.test_suite_run.fail_count": statuses.count("fail"),
            "opensearch.test_suite_run.error_count": error_count,
            "opensearch.test_suite_run.duration_ms": float(duration_ms),
        }
    )
    _emit_sdk_event(TEST_SUITE_RUN_EVENT_NAME, attrs, _no_span_context())
    return run_status


def _emit_scores(
    grouping: dict[str, Any],
    scores: Iterable[EvalScore | _FailedScore],
    *,
    context: Context,
    evaluator: str | None = None,
    error_type: str | None = None,
) -> None:
    """Emit one ``gen_ai.evaluation.result`` event per score.

    Each event is parented to ``context`` (the evaluated span, or no span)
    and carries the ``test.*`` grouping attributes.
    """
    has_span = trace.get_current_span(context).get_span_context().is_valid
    for s in scores:
        extra = dict(grouping)
        if s.metadata:
            extra.update(_safe_metadata(s.metadata, context=f"EvalScore({s.name!r})"))
        attrs = build_evaluation_attributes(
            name=s.name,
            value=s.value,
            label=s.label,
            explanation=s.explanation,
            level="span" if has_span else None,
            evaluator=evaluator,
            error_type=error_type,
            extra=extra,
        )
        emit_evaluation_event(attrs, context=context, evaluator=evaluator)


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
# Benchmark class: direct upload (Mode B)
# ---------------------------------------------------------------------------


class Benchmark:
    """Upload evaluation results as OTel log events.

    Each ``log()`` call emits one ``opensearch.test_case.result`` event and
    one ``gen_ai.evaluation.result`` event per score; ``close()`` emits one
    ``opensearch.test_suite_run.result`` event. When ``trace_id`` /
    ``span_id`` of the agent run are given, the case and score events are
    parented to that span; otherwise they carry no trace context. No spans
    are created.

    Use as a context manager::

        with Benchmark(name="rag-v2") as b:
            b.log(input="q1", output="a1", scores={"accuracy": 1.0})

    Or manually::

        b = Benchmark(name="rag-v2")
        b.log(input="q1", output="a1", scores={"accuracy": 1.0})
        summary = b.close()

    Args:
        name: Benchmark name (``test.suite.name``). Stable across runs.
        metadata: Arbitrary run metadata, added to the case and run events.
            Keys reserved
            by the SDK (``test.suite.name`` etc.) are filtered out with a
            warning.
        record_io: If ``True``, record input/output/expected as
            ``opensearch.test_case.input`` / ``.output`` / ``.expected`` on
            the case event. Default ``False`` (PII-safe).
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
        self._statuses: list[str] = []
        self._closed = False

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
        status: str | None = None,
        error_type: str | None = None,
        duration_ms: float | None = None,
    ) -> None:
        """Log a single test case result.

        Emits one ``opensearch.test_case.result`` event and one
        ``gen_ai.evaluation.result`` event per score, with the ``test.*``
        ids set.

        Args:
            input: Test case input data.
            output: Actual output from the agent/system.
            expected: Expected/ground-truth output.
            scores: Pre-computed scores (name -> float).
            metadata: Arbitrary case-level metadata. Keys reserved by the
                SDK are filtered out with a warning.
            error: Error message. The case event status becomes ``"error"``
                (``TestCaseResult.status`` stays ``"fail"`` as before).
            case_id: Explicit ``test.case.id``. If omitted, derived
                from a hash of ``input``.
            case_name: Human-readable ``test.case.name``.
            trace_id: Hex trace ID of the agent run that produced ``output``.
            span_id: Hex span ID of the agent run's root span. Required when
                ``trace_id`` is provided. The case and score events are
                parented to it.
            status: Explicit case status, ``"pass"``, ``"fail"`` or
                ``"error"``. Defaults to ``"error"`` when ``error`` is given,
                else ``"pass"``.
            error_type: ``error.type`` for an errored case (exception class
                name or other low-cardinality error identifier).
            duration_ms: Case duration, recorded as
                ``opensearch.test_case.duration_ms``.
        """
        if self._closed:
            raise RuntimeError("Benchmark is already closed")
        if bool(trace_id) != bool(span_id):
            raise ValueError("Both trace_id and span_id must be provided together, or neither.")
        if status is not None and status not in _CASE_STATUSES:
            raise ValueError(f"status must be one of {', '.join(_CASE_STATUSES)}; got {status!r}")

        scores = scores or {}
        resolved_case_id = case_id or _make_case_id(input)

        event_status = status or ("error" if error or error_type else "pass")
        result_status = status or ("fail" if error else "pass")

        grouping = _grouping(self._name, self._run_id, resolved_case_id, case_name)

        safe_case_meta = (
            _safe_metadata(metadata, context=f"Benchmark.log(case_id={resolved_case_id!r})")
            if metadata
            else {}
        )
        case_meta = {**self._metadata, **safe_case_meta}

        io: dict[str, Any] = {}
        if self._record_io:
            if input is not None:
                io["opensearch.test_case.input"] = _io_attr(input)
            if output is not None:
                io["opensearch.test_case.output"] = _io_attr(output)
            if expected is not None:
                io["opensearch.test_case.expected"] = _io_attr(expected)

        event_ctx: Context = _no_span_context()
        if trace_id and span_id:
            evaluated_ctx = span_context_for(trace_id, span_id)
            if evaluated_ctx is not None:
                event_ctx = evaluated_ctx
            else:
                logger.warning(
                    "Benchmark.log(): invalid trace_id/span_id %r/%r; "
                    "emitting events without trace context",
                    trace_id,
                    span_id,
                )

        _emit_scores(
            grouping,
            [EvalScore(name=n, value=v) for n, v in scores.items()],
            context=event_ctx,
        )
        _emit_case_event(
            grouping,
            status=event_status,
            context=event_ctx,
            error_type=error_type,
            io=io,
            duration_ms=duration_ms,
            metadata=case_meta,
        )

        ev_trace_id, ev_span_id = _span_ids(event_ctx)
        self._statuses.append(event_status)
        self._cases.append(
            TestCaseResult(
                case_id=resolved_case_id,
                case_name=case_name,
                input=input,
                output=output,
                expected=expected,
                scores=scores,
                error=error,
                status=result_status,
                error_type=error_type,
                duration_ms=duration_ms,
                trace_id=ev_trace_id,
                span_id=ev_span_id,
            )
        )

    def close(self) -> BenchmarkSummary:
        """Finalize the benchmark and return a summary.

        Emits the ``opensearch.test_suite_run.result`` event, computes aggregate
        statistics, and prints a formatted summary.

        Returns:
            ``BenchmarkSummary`` with aggregate score statistics.
        """
        if self._closed:
            raise RuntimeError("Benchmark is already closed")

        self._closed = True

        summary = _compute_summary(self._name, self._run_id, self._cases, self._start_time)

        _emit_run_event(
            self._name,
            self._run_id,
            self._statuses,
            duration_ms=summary.duration_ms,
            metadata=self._metadata,
        )

        print(str(summary))
        logger.info("Benchmark completed: %s", self._run_id)

        return summary

    def __enter__(self) -> Benchmark:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if not self._closed:
            self.close()


# ---------------------------------------------------------------------------
# evaluate(): full eval runner (Mode A)
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

    Each case's task call is its own trace, exactly like production: the
    SDK adds no span around it. The root span(s) the task starts get
    ``test.suite.name``, ``test.suite.run.id``, ``test.case.id`` and
    ``test.case.name`` (stamped by a span processor installed on the global
    SDK ``TracerProvider``). Per case the SDK then emits one
    ``gen_ai.evaluation.result`` event per score and one
    ``opensearch.test_case.result`` event, both parented to that root span
    (or without trace context when the task started no span). At the end it
    emits one ``opensearch.test_suite_run.result`` event.

    A task that raises does not stop the run: its case event has status
    ``error`` and ``error.type`` = the exception class, and its scorers are
    skipped. A scorer that raises emits an evaluation event with
    ``error.type`` and marks the case ``error``.

    Args:
        name: Benchmark name (``test.suite.name``).
        task: Callable that takes the input and returns output. Should
            be decorated with ``@observe()`` (or otherwise instrumented) so
            each case produces a trace.
        data: Iterable of dicts with ``"input"`` and optionally
            ``"expected"``, ``"case_id"`` and ``"case_name"`` keys. Consumed
            once (single-pass); generators are materialized into a list.
        scores: List of scorer callables. Each receives
            ``(input, output, expected)`` and returns an ``EvalScore``,
            list of ``EvalScore``, or a plain float. Scorers run outside the
            agent trace, so a traced LLM judge gets its own trace.
        metadata: Run metadata added to the case and run events. Keys
            reserved by the SDK are filtered out with a warning.
        record_io: Record input/output/expected as
            ``opensearch.test_case.input`` / ``.output`` / ``.expected`` on
            the case event.

    Returns:
        ``BenchmarkResult`` with summary and per-case results.
    """
    run_id = _make_run_id()
    start_time = time.monotonic()

    safe_meta = _safe_metadata(metadata or {}, context=f"evaluate({name!r})")
    _ensure_root_stamper()

    case_results: list[TestCaseResult] = []
    statuses: list[str] = []

    # Materialize iterable upfront so generators work
    data_list = list(data)

    for item in data_list:
        case_start = time.monotonic()
        input_val = item.get("input")
        expected_val = item.get("expected")
        case_id = item.get("case_id") or _make_case_id(input_val)
        case_name_val = item.get("case_name")

        grouping = _grouping(name, run_id, case_id, case_name_val)

        io: dict[str, Any] = {}
        if record_io:
            if input_val is not None:
                io["opensearch.test_case.input"] = _io_attr(input_val)
            if expected_val is not None:
                io["opensearch.test_case.expected"] = _io_attr(expected_val)

        # Run the task as its own trace and capture its root span.
        output_val = None
        error_msg: str | None = None
        error_type: str | None = None
        scope: _CaseScope | None = None
        try:
            with _agent_run_scope(grouping) as scope:
                output_val = task(input_val)
        except Exception as exc:
            error_msg = str(exc)
            error_type = type(exc).__qualname__
            logger.warning("Task failed for case %s: %s", case_id, exc)

        if record_io and output_val is not None:
            io["opensearch.test_case.output"] = _io_attr(output_val)

        # Case and score events are parented to the agent run's root span.
        event_ctx: Context
        if scope is not None and scope.root is not None:
            event_ctx = trace.set_span_in_context(NonRecordingSpan(scope.root))
        else:
            event_ctx = _no_span_context()

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
                    error_type = error_type or type(exc).__qualname__
                    logger.warning(
                        "Scorer %s failed for case %s: %s",
                        scorer_name,
                        case_id,
                        exc,
                    )
                    # Record the evaluator failure (error.type, no score value).
                    _emit_scores(
                        grouping,
                        [_FailedScore(name=scorer_name, explanation=str(exc))],
                        context=event_ctx,
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
                    grouping,
                    produced,
                    context=event_ctx,
                    evaluator=evaluator,
                )

        # TestCaseResult.status: fail on any task or scorer error.
        # Event status: error on any task or scorer error.
        errored = error_msg is not None or bool(scorer_errors)
        result_status = "fail" if errored else "pass"
        event_status = "error" if errored else "pass"
        duration_ms = (time.monotonic() - case_start) * 1000

        _emit_case_event(
            grouping,
            status=event_status,
            context=event_ctx,
            error_type=error_type,
            io=io,
            duration_ms=duration_ms,
            metadata=safe_meta,
        )

        ev_trace_id, ev_span_id = _span_ids(event_ctx)
        statuses.append(event_status)
        case_results.append(
            TestCaseResult(
                case_id=case_id,
                case_name=case_name_val,
                input=input_val,
                output=output_val,
                expected=expected_val,
                scores=case_scores,
                error=error_msg,
                status=result_status,
                scorer_errors=scorer_errors or None,
                error_type=error_type,
                duration_ms=duration_ms,
                trace_id=ev_trace_id,
                span_id=ev_span_id,
            )
        )

    summary = _compute_summary(name, run_id, case_results, start_time)
    _emit_run_event(name, run_id, statuses, duration_ms=summary.duration_ms, metadata=safe_meta)

    print(str(summary))
    logger.info("evaluate() completed: %s", run_id)

    return BenchmarkResult(summary=summary, cases=case_results)
