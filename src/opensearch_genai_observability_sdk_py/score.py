# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
#
# Modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

"""Evaluation results for OpenSearch AI observability.

``score()`` records one evaluation result as an OTel **log-based event**
named ``gen_ai.evaluation.result``, following the `OTel GenAI semantic
conventions <https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-events/>`_.
The event is a LogRecord exported over OTLP logs (``/v1/logs``) and carries
the trace context of the **evaluated** span, so it joins the trace without
creating a new span. Data Prepper routes these events to the
``otel-v1-genai-evaluation`` index.

Scoring levels:

- **Span-level:** ``trace_id`` + ``span_id``, or call ``score()`` while the
  evaluated span is active. The event carries that span's ids.
- **Trace-level:** ``trace_id`` only. The event has no span context and
  carries ``opensearch.evaluation.trace_id`` instead. No span id is derived.
- **Session-level:** ``conversation_id`` (``gen_ai.conversation.id``) with
  ``level="session"``.

The previous carrier (a child ``evaluation <name>`` span with mirrored span
attributes and a span event) is deprecated. It is still available for one
release with ``carrier="span"`` or ``OPENSEARCH_GENAI_EVAL_CARRIER=span``;
``carrier="both"`` emits both for migration.
"""

from __future__ import annotations

import logging
import os
import time
import warnings
from typing import Any, Literal

from opentelemetry import _logs, trace
from opentelemetry._logs import SeverityNumber
from opentelemetry.context import Context
from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags

from opensearch_genai_observability_sdk_py._internal import (
    SDK_SCOPE_NAME,
    SDK_VERSION,
    parse_hex,
)

logger = logging.getLogger(__name__)

_TRACER_NAME = "opensearch-genai-observability-sdk-py-scores"

EVALUATION_EVENT_NAME = "gen_ai.evaluation.result"
CARRIER_ENV_VAR = "OPENSEARCH_GENAI_EVAL_CARRIER"

# Safety cap for ``gen_ai.evaluation.explanation`` on the event carrier.
# Judge rationales are kept whole up to this many characters; longer values
# are cut and end with ``_TRUNCATION_MARKER``.
EXPLANATION_MAX_CHARS = 8192
# The deprecated span carrier keeps its historical 500 character limit.
_SPAN_CARRIER_EXPLANATION_MAX_CHARS = 500
_TRUNCATION_MARKER = "...[truncated]"

Carrier = Literal["event", "span", "both"]
_VALID_CARRIERS = ("event", "span", "both")
_VALID_LEVELS = ("span", "trace", "session", "tool_call")
_VALID_EVALUATOR_TYPES = ("llm_judge", "code", "human", "external")
_VALID_DIRECTIONS = ("maximize", "minimize")

_span_carrier_warned = False


def _no_span_context() -> Context:
    """A context that explicitly holds no valid span.

    An empty ``Context()`` is falsy, and the Logs SDK then falls back to the
    current context (the active span), so set ``INVALID_SPAN`` explicitly.
    """
    return trace.set_span_in_context(trace.INVALID_SPAN, Context())


def score(
    name: str,
    value: float | None = None,
    *,
    trace_id: str | None = None,
    span_id: str | None = None,
    label: str | None = None,
    explanation: str | None = None,
    response_id: str | None = None,
    attributes: dict[str, Any] | None = None,
    level: str | None = None,
    evaluator: str | None = None,
    evaluator_version: str | None = None,
    evaluator_type: str | None = None,
    direction: str | None = None,
    score_min: float | None = None,
    score_max: float | None = None,
    conversation_id: str | None = None,
    agent_name: str | None = None,
    error_type: str | None = None,
    judge_trace_id: str | None = None,
    carrier: Carrier | None = None,
) -> None:
    """Record an evaluation result as a ``gen_ai.evaluation.result`` event.

    Emits one OTel LogRecord through the Logs API with ``event_name`` set to
    ``gen_ai.evaluation.result``, ``timestamp`` set to now, severity INFO and
    all data in attributes. The record is parented to the evaluated span:

    - ``trace_id`` + ``span_id`` given: that span (works for spans that have
      already ended, for example offline or async evaluation).
    - nothing given: the currently active span, if any (inline evaluation).
    - ``trace_id`` only: no span context; ``opensearch.evaluation.trace_id``
      holds the target trace and the level defaults to ``trace``.

    The instrumentation scope is the evaluator identity: ``evaluator`` /
    ``evaluator_version`` when given, otherwise the SDK name and version.

    ``register()`` configures the OTLP logs pipeline. With your own OTel
    setup, configure a global ``LoggerProvider`` or the events are dropped.

    Attributes written (only when the value is given):

    - Standard: ``gen_ai.evaluation.name``, ``gen_ai.evaluation.score.value``,
      ``gen_ai.evaluation.score.label``, ``gen_ai.evaluation.explanation``,
      ``gen_ai.response.id``, ``error.type``, ``gen_ai.conversation.id``,
      ``gen_ai.agent.name``.
    - Extensions (until the upstream proposals merge):
      ``opensearch.evaluation.level``, ``.evaluator.name``,
      ``.evaluator.version``, ``.evaluator.type``, ``.score.direction``,
      ``.score.min``, ``.score.max``, ``.judge.trace_id``, ``.trace_id``.

    Args:
        name: Evaluation metric name (e.g., ``"relevance"``, ``"factuality"``).
        value: Numeric score value.
        trace_id: Hex trace ID of the evaluated trace.
        span_id: Hex span ID of the evaluated span. Without it the result is
            trace-level (no span id is ever derived from the trace id).
        label: Low-cardinality label (e.g., ``"pass"``, ``"relevant"``).
        explanation: Evaluator rationale. Kept whole up to
            ``EXPLANATION_MAX_CHARS`` (8192) characters on the event; the
            deprecated span carrier still truncates at 500.
        response_id: Completion ID of the evaluated LLM response.
        attributes: Extra attributes added as-is (e.g., ``test.*`` grouping
            keys from semantic-conventions-genai#79).
        level: ``"span"``, ``"trace"``, ``"session"`` or ``"tool_call"``.
            Defaults to ``"span"`` with a span context, ``"trace"`` with only
            ``trace_id``, ``"session"`` with only ``conversation_id``.
        evaluator: Evaluator name; also used as the instrumentation scope.
        evaluator_version: Evaluator version (instrumentation scope version).
        evaluator_type: ``"llm_judge"``, ``"code"``, ``"human"`` or ``"external"``.
        direction: ``"maximize"`` or ``"minimize"``: which way is better.
        score_min: Lower bound of the score scale.
        score_max: Upper bound of the score scale.
        conversation_id: ``gen_ai.conversation.id`` of the evaluated session.
        agent_name: ``gen_ai.agent.name`` of the evaluated agent.
        error_type: ``error.type`` when the evaluator itself failed.
        judge_trace_id: Trace id of the judge's own run, if traced.
        carrier: ``"event"`` (default), ``"span"`` (deprecated) or ``"both"``.
            When omitted, ``OPENSEARCH_GENAI_EVAL_CARRIER`` is honored.

    Examples:
        # Inline: score the active span
        with tracer.start_as_current_span("chat gpt-4o"):
            ...
            score("relevance", 0.92, label="relevant", evaluator="relevance_judge")

        # Offline: score a stored span
        score(
            name="accuracy",
            value=0.95,
            trace_id="6ebb9835f43af1552f2cebb9f5165e39",
            span_id="89829115c2128845",
            explanation="Weather data is correct",
            evaluator="accuracy_judge",
            evaluator_type="llm_judge",
            score_min=0.0,
            score_max=1.0,
            direction="maximize",
        )

        # Trace-level with benchmark grouping
        score(
            name="helpfulness",
            value=0.83,
            trace_id="6ebb9835f43af1552f2cebb9f5165e39",
            label="helpful",
            attributes={"test.suite.run.id": "run_001", "test.case.id": "case_001"},
        )
    """
    resolved_carrier = resolve_carrier(carrier)

    ctx, has_span, target_trace_id = _resolve_target(trace_id, span_id)

    resolved_level = level
    if resolved_level is None:
        if has_span:
            resolved_level = "span"
        elif target_trace_id:
            resolved_level = "trace"
        elif conversation_id:
            resolved_level = "session"

    common: dict[str, Any] = dict(
        name=name,
        value=value,
        label=label,
        response_id=response_id,
        level=resolved_level,
        evaluator=evaluator,
        evaluator_version=evaluator_version,
        evaluator_type=evaluator_type,
        direction=direction,
        score_min=score_min,
        score_max=score_max,
        conversation_id=conversation_id,
        agent_name=agent_name,
        error_type=error_type,
        judge_trace_id=judge_trace_id,
        target_trace_id=target_trace_id,
        extra=attributes,
    )

    if resolved_carrier in ("event", "both"):
        event_attrs = build_evaluation_attributes(
            explanation=explanation, explanation_limit=EXPLANATION_MAX_CHARS, **common
        )
        emit_evaluation_event(
            event_attrs,
            context=ctx,
            evaluator=evaluator,
            evaluator_version=evaluator_version,
        )

    if resolved_carrier in ("span", "both"):
        warn_span_carrier_deprecated()
        span_attrs = build_evaluation_attributes(
            explanation=explanation,
            explanation_limit=_SPAN_CARRIER_EXPLANATION_MAX_CHARS,
            **common,
        )
        _emit_score_span(name, span_attrs, ctx)

    logger.debug(
        "Score emitted: %s=%s (trace=%s, carrier=%s)", name, value, trace_id, resolved_carrier
    )


# ---------------------------------------------------------------------------
# Shared helpers (also used by benchmark.py)
# ---------------------------------------------------------------------------


def resolve_carrier(carrier: str | None) -> str:
    """Resolve the carrier from the argument or ``OPENSEARCH_GENAI_EVAL_CARRIER``."""
    if carrier is not None:
        if carrier not in _VALID_CARRIERS:
            raise ValueError(
                f"Unknown carrier {carrier!r}. Use one of: {', '.join(_VALID_CARRIERS)}."
            )
        return carrier

    env = os.environ.get(CARRIER_ENV_VAR, "").strip().lower()
    if not env:
        return "event"
    if env not in _VALID_CARRIERS:
        logger.warning(
            "%s=%r is not one of %s; using 'event'", CARRIER_ENV_VAR, env, _VALID_CARRIERS
        )
        return "event"
    return env


def warn_span_carrier_deprecated() -> None:
    """Emit the span-carrier ``DeprecationWarning`` once per process."""
    global _span_carrier_warned
    if _span_carrier_warned:
        return
    _span_carrier_warned = True
    warnings.warn(
        "The span carrier for evaluation results (carrier='span' or "
        f"{CARRIER_ENV_VAR}=span) is deprecated and will be removed in a future "
        "release. Evaluation results are now emitted as gen_ai.evaluation.result "
        "log events (carrier='event').",
        DeprecationWarning,
        stacklevel=3,
    )


def build_evaluation_attributes(
    *,
    name: str,
    value: float | None = None,
    label: str | None = None,
    explanation: str | None = None,
    explanation_limit: int = EXPLANATION_MAX_CHARS,
    response_id: str | None = None,
    level: str | None = None,
    evaluator: str | None = None,
    evaluator_version: str | None = None,
    evaluator_type: str | None = None,
    direction: str | None = None,
    score_min: float | None = None,
    score_max: float | None = None,
    conversation_id: str | None = None,
    agent_name: str | None = None,
    error_type: str | None = None,
    judge_trace_id: str | None = None,
    target_trace_id: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the ``gen_ai.evaluation.result`` attribute map (only set keys)."""
    _check_enum("level", level, _VALID_LEVELS)
    _check_enum("evaluator_type", evaluator_type, _VALID_EVALUATOR_TYPES)
    _check_enum("direction", direction, _VALID_DIRECTIONS)

    attrs: dict[str, Any] = {"gen_ai.evaluation.name": name}
    if value is not None:
        attrs["gen_ai.evaluation.score.value"] = float(value)
    if label:
        attrs["gen_ai.evaluation.score.label"] = label
    if explanation:
        attrs["gen_ai.evaluation.explanation"] = _truncate(explanation, explanation_limit)
    if response_id:
        attrs["gen_ai.response.id"] = response_id
    if error_type:
        attrs["error.type"] = error_type
    if conversation_id:
        attrs["gen_ai.conversation.id"] = conversation_id
    if agent_name:
        attrs["gen_ai.agent.name"] = agent_name
    if level:
        attrs["opensearch.evaluation.level"] = level
    if evaluator:
        attrs["opensearch.evaluation.evaluator.name"] = evaluator
    if evaluator_version:
        attrs["opensearch.evaluation.evaluator.version"] = evaluator_version
    if evaluator_type:
        attrs["opensearch.evaluation.evaluator.type"] = evaluator_type
    if direction:
        attrs["opensearch.evaluation.score.direction"] = direction
    if score_min is not None:
        attrs["opensearch.evaluation.score.min"] = float(score_min)
    if score_max is not None:
        attrs["opensearch.evaluation.score.max"] = float(score_max)
    if judge_trace_id:
        attrs["opensearch.evaluation.judge.trace_id"] = judge_trace_id
    if target_trace_id:
        attrs["opensearch.evaluation.trace_id"] = target_trace_id
    if extra:
        attrs.update(extra)
    return attrs


def emit_evaluation_event(
    attributes: dict[str, Any],
    *,
    context: Context | None = None,
    evaluator: str | None = None,
    evaluator_version: str | None = None,
) -> None:
    """Emit one ``gen_ai.evaluation.result`` LogRecord via the global LoggerProvider.

    ``context`` carries the evaluated span (``None`` means the current
    context, i.e. the active span). The scope is the evaluator identity, or
    the SDK scope when no evaluator is named.
    """
    if evaluator:
        scope_name, scope_version = evaluator, evaluator_version or ""
    else:
        scope_name, scope_version = SDK_SCOPE_NAME, SDK_VERSION or ""
    otel_logger = _logs.get_logger(scope_name, scope_version)
    otel_logger.emit(
        event_name=EVALUATION_EVENT_NAME,
        timestamp=time.time_ns(),
        severity_number=SeverityNumber.INFO,
        attributes=attributes,
        context=context,
    )


def span_context_for(trace_id: str, span_id: str) -> Context | None:
    """Return a Context whose current span is the (remote) span ``trace_id/span_id``.

    Returns ``None`` if either id fails to parse.
    """
    trace_id_int = parse_hex(trace_id)
    span_id_int = parse_hex(span_id)
    if not trace_id_int or not span_id_int:
        return None
    sc = SpanContext(
        trace_id=trace_id_int,
        span_id=span_id_int,
        is_remote=True,
        trace_flags=TraceFlags(TraceFlags.SAMPLED),
    )
    return trace.set_span_in_context(NonRecordingSpan(sc))


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _resolve_target(
    trace_id: str | None,
    span_id: str | None,
) -> tuple[Context | None, bool, str | None]:
    """Work out which span/trace is evaluated.

    Returns ``(context, has_span, target_trace_id)``:

    - ``context`` is passed to ``Logger.emit`` / used as the span parent.
      ``None`` means "use the current context" (the active span).
      ``_no_span_context()`` means "no span context at all".
    - ``has_span`` is ``True`` when the context carries a valid span.
    - ``target_trace_id`` is the normalized trace id for trace-level results
      (no span id known), else ``None``.
    """
    if not trace_id:
        if span_id:
            logger.warning("score(): span_id given without trace_id; ignoring both ids")
            return _no_span_context(), False, None
        active = trace.get_current_span().get_span_context()
        return None, active.is_valid, None

    trace_id_int = parse_hex(trace_id)
    if not trace_id_int:
        logger.warning("score(): invalid trace_id '%s', emitting without trace context", trace_id)
        return _no_span_context(), False, None
    normalized_trace_id = format(trace_id_int, "032x")

    if span_id:
        ctx = span_context_for(trace_id, span_id)
        if ctx is not None:
            return ctx, True, None
        logger.warning(
            "score(): invalid span_id '%s', recording a trace-level result instead", span_id
        )

    # Trace-level: never invent a span id. Emit with a span-less context so the
    # active span (if any) is not picked up, and carry the target trace id.
    return _no_span_context(), False, normalized_trace_id


def _emit_score_span(name: str, attrs: dict[str, Any], ctx: Context | None) -> None:
    """Deprecated span carrier: child span ``evaluation <name>`` plus a span event."""
    tracer = trace.get_tracer(_TRACER_NAME)
    span_attrs: dict[str, Any] = {"gen_ai.operation.name": "evaluation", **attrs}
    with tracer.start_as_current_span(
        f"evaluation {name}",
        context=ctx,
        attributes=span_attrs,
    ) as span:
        span.add_event(EVALUATION_EVENT_NAME, attributes=attrs)


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    if limit <= len(_TRUNCATION_MARKER):
        return value[:limit]
    return value[: limit - len(_TRUNCATION_MARKER)] + _TRUNCATION_MARKER


def _check_enum(arg: str, value: str | None, allowed: tuple[str, ...]) -> None:
    if value is not None and value not in allowed:
        logger.warning("score(): %s=%r is not one of %s; recording as-is", arg, value, allowed)
