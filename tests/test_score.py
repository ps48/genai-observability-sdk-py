# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
#
# Modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

"""Tests for opensearch_genai_observability_sdk_py.score."""

import importlib
import warnings

import pytest
from opentelemetry import trace
from opentelemetry._logs import SeverityNumber

from opensearch_genai_observability_sdk_py._internal import SDK_SCOPE_NAME, SDK_VERSION
from opensearch_genai_observability_sdk_py._internal import parse_hex as _parse_hex
from opensearch_genai_observability_sdk_py.score import EXPLANATION_MAX_CHARS, score

_score_mod = importlib.import_module("opensearch_genai_observability_sdk_py.score")

TRACE_ID = "6ebb9835f43af1552f2cebb9f5165e39"
SPAN_ID = "89829115c2128845"


@pytest.fixture(autouse=True)
def _reset_carrier(monkeypatch):
    """Isolate tests from the carrier env var and the one-time warning flag."""
    monkeypatch.delenv("OPENSEARCH_GENAI_EVAL_CARRIER", raising=False)
    monkeypatch.setattr(_score_mod, "_span_carrier_warned", False)
    # Span-carrier tests trigger the DeprecationWarning; keep output quiet.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        yield


def _one_event(eval_events):
    events = eval_events()
    assert len(events) == 1, events
    return events[0]


def _ids(record):
    lr = record.log_record
    return format(lr.trace_id, "032x"), format(lr.span_id, "016x")


# ---------------------------------------------------------------------------
# Event carrier (default): gen_ai.evaluation.result log record
# ---------------------------------------------------------------------------


class TestEventEmission:
    def test_event_name_is_top_level_field(self, eval_events):
        score(name="relevance", value=0.9)
        lr = _one_event(eval_events).log_record
        assert lr.event_name == "gen_ai.evaluation.result"
        assert "event.name" not in lr.attributes

    def test_severity_info(self, eval_events):
        score(name="relevance", value=0.9)
        lr = _one_event(eval_events).log_record
        assert lr.severity_number == SeverityNumber.INFO

    def test_timestamp_set(self, eval_events):
        import time

        before = time.time_ns()
        score(name="relevance", value=0.9)
        after = time.time_ns()
        lr = _one_event(eval_events).log_record
        assert lr.timestamp is not None
        assert before <= lr.timestamp <= after

    def test_body_empty(self, eval_events):
        score(name="relevance", value=0.9)
        assert _one_event(eval_events).log_record.body is None

    def test_no_span_created_by_default(self, exporter, eval_events):
        score(name="relevance", value=0.9, trace_id=TRACE_ID, span_id=SPAN_ID)
        assert exporter.get_finished_spans() == ()
        assert len(eval_events()) == 1

    def test_default_scope_is_sdk(self, eval_events):
        score(name="relevance", value=0.9)
        scope = _one_event(eval_events).instrumentation_scope
        assert scope.name == SDK_SCOPE_NAME == "opensearch-genai-observability-sdk-py"
        assert scope.version == (SDK_VERSION or "")

    def test_evaluator_is_scope(self, eval_events):
        score(name="relevance", value=0.9, evaluator="relevance_judge", evaluator_version="3.1")
        rec = _one_event(eval_events)
        assert rec.instrumentation_scope.name == "relevance_judge"
        assert rec.instrumentation_scope.version == "3.1"
        assert (
            rec.log_record.attributes["opensearch.evaluation.evaluator.name"] == "relevance_judge"
        )
        assert rec.log_record.attributes["opensearch.evaluation.evaluator.version"] == "3.1"

    def test_multiple_scores_are_independent(self, eval_events):
        score(name="a", value=0.1)
        score(name="b", value=0.2)
        score(name="c", value=0.3)
        values = {
            r.log_record.attributes["gen_ai.evaluation.name"]: r.log_record.attributes[
                "gen_ai.evaluation.score.value"
            ]
            for r in eval_events()
        }
        assert values == {"a": 0.1, "b": 0.2, "c": 0.3}


class TestEventParenting:
    def test_explicit_span(self, eval_events):
        score(name="accuracy", value=0.95, trace_id=TRACE_ID, span_id=SPAN_ID)
        rec = _one_event(eval_events)
        assert _ids(rec) == (TRACE_ID, SPAN_ID)
        assert rec.log_record.attributes["opensearch.evaluation.level"] == "span"
        assert "opensearch.evaluation.trace_id" not in rec.log_record.attributes

    def test_explicit_ids_accept_0x_and_uppercase(self, eval_events):
        score(name="accuracy", trace_id="0x" + TRACE_ID.upper(), span_id=SPAN_ID.upper())
        assert _ids(_one_event(eval_events)) == (TRACE_ID, SPAN_ID)

    def test_active_span(self, eval_events):
        tracer = trace.get_tracer("test")
        with tracer.start_as_current_span("chat gpt-4o") as span:
            score(name="relevance", value=0.92)
            sc = span.get_span_context()
        rec = _one_event(eval_events)
        assert rec.log_record.trace_id == sc.trace_id
        assert rec.log_record.span_id == sc.span_id
        assert rec.log_record.attributes["opensearch.evaluation.level"] == "span"

    def test_explicit_ids_win_over_active_span(self, eval_events):
        tracer = trace.get_tracer("test")
        with tracer.start_as_current_span("other"):
            score(name="relevance", trace_id=TRACE_ID, span_id=SPAN_ID)
        assert _ids(_one_event(eval_events)) == (TRACE_ID, SPAN_ID)

    def test_trace_only_has_no_span_context(self, eval_events):
        score(name="relevance", value=0.92, trace_id=TRACE_ID)
        lr = _one_event(eval_events).log_record
        assert lr.trace_id == 0
        assert lr.span_id == 0
        assert lr.attributes["opensearch.evaluation.trace_id"] == TRACE_ID
        assert lr.attributes["opensearch.evaluation.level"] == "trace"

    def test_trace_only_never_derives_root_span_id(self, eval_events, exporter):
        """Regression: no span id from the low 64 bits of the trace id."""
        score(name="relevance", value=0.92, trace_id=TRACE_ID)
        lr = _one_event(eval_events).log_record
        assert lr.span_id != int(TRACE_ID, 16) & 0xFFFFFFFFFFFFFFFF

    def test_trace_only_ignores_active_span(self, eval_events):
        tracer = trace.get_tracer("test")
        with tracer.start_as_current_span("unrelated"):
            score(name="relevance", trace_id=TRACE_ID)
        lr = _one_event(eval_events).log_record
        assert lr.trace_id == 0 and lr.span_id == 0
        assert lr.attributes["opensearch.evaluation.trace_id"] == TRACE_ID

    def test_trace_id_normalized(self, eval_events):
        score(name="relevance", trace_id="0x" + TRACE_ID.upper())
        lr = _one_event(eval_events).log_record
        assert lr.attributes["opensearch.evaluation.trace_id"] == TRACE_ID

    def test_no_context_no_level(self, eval_events):
        score(name="test", value=0.5)
        lr = _one_event(eval_events).log_record
        assert lr.trace_id == 0
        assert "opensearch.evaluation.level" not in lr.attributes

    def test_invalid_trace_id_standalone(self, eval_events):
        score(name="test", value=0.5, trace_id="not-hex!", span_id=SPAN_ID)
        lr = _one_event(eval_events).log_record
        assert lr.trace_id == 0
        assert "opensearch.evaluation.trace_id" not in lr.attributes

    def test_invalid_span_id_falls_back_to_trace_level(self, eval_events):
        score(name="test", value=0.5, trace_id=TRACE_ID, span_id="not-hex!")
        lr = _one_event(eval_events).log_record
        assert lr.span_id == 0
        assert lr.attributes["opensearch.evaluation.trace_id"] == TRACE_ID
        assert lr.attributes["opensearch.evaluation.level"] == "trace"

    def test_span_id_without_trace_id_ignored(self, eval_events):
        tracer = trace.get_tracer("test")
        with tracer.start_as_current_span("unrelated"):
            score(name="test", span_id=SPAN_ID)
        lr = _one_event(eval_events).log_record
        assert lr.trace_id == 0 and lr.span_id == 0


class TestEventAttributes:
    def test_standard_attributes(self, eval_events):
        score(
            name="helpfulness",
            value=0.83,
            label="helpful",
            explanation="Good response",
            response_id="resp_123",
            conversation_id="conv-1",
            agent_name="travel_agent",
        )
        a = _one_event(eval_events).log_record.attributes
        assert a["gen_ai.evaluation.name"] == "helpfulness"
        assert a["gen_ai.evaluation.score.value"] == 0.83
        assert a["gen_ai.evaluation.score.label"] == "helpful"
        assert a["gen_ai.evaluation.explanation"] == "Good response"
        assert a["gen_ai.response.id"] == "resp_123"
        assert a["gen_ai.conversation.id"] == "conv-1"
        assert a["gen_ai.agent.name"] == "travel_agent"

    def test_extension_attributes(self, eval_events):
        score(
            name="faithfulness",
            value=4,
            trace_id=TRACE_ID,
            span_id=SPAN_ID,
            level="tool_call",
            evaluator="faith_judge",
            evaluator_version="1.0",
            evaluator_type="llm_judge",
            direction="maximize",
            score_min=1,
            score_max=5,
            judge_trace_id="abcd" * 8,
        )
        a = _one_event(eval_events).log_record.attributes
        assert a["opensearch.evaluation.level"] == "tool_call"
        assert a["opensearch.evaluation.evaluator.name"] == "faith_judge"
        assert a["opensearch.evaluation.evaluator.version"] == "1.0"
        assert a["opensearch.evaluation.evaluator.type"] == "llm_judge"
        assert a["opensearch.evaluation.score.direction"] == "maximize"
        assert a["opensearch.evaluation.score.min"] == 1.0
        assert a["opensearch.evaluation.score.max"] == 5.0
        assert isinstance(a["opensearch.evaluation.score.min"], float)
        assert a["opensearch.evaluation.judge.trace_id"] == "abcd" * 8

    def test_value_is_double(self, eval_events):
        score(name="t", value=1)
        v = _one_event(eval_events).log_record.attributes["gen_ai.evaluation.score.value"]
        assert v == 1.0 and isinstance(v, float)

    def test_zero_value_kept(self, eval_events):
        score(name="toxicity", value=0.0)
        a = _one_event(eval_events).log_record.attributes
        assert a["gen_ai.evaluation.score.value"] == 0.0

    def test_minimal(self, eval_events):
        score(name="accuracy")
        a = dict(_one_event(eval_events).log_record.attributes)
        assert a == {"gen_ai.evaluation.name": "accuracy"}

    def test_error_type(self, eval_events):
        score(name="relevance", error_type="timeout")
        a = _one_event(eval_events).log_record.attributes
        assert a["error.type"] == "timeout"
        assert "gen_ai.evaluation.score.value" not in a

    def test_session_level_default(self, eval_events):
        score(name="user_satisfaction", value=0.9, conversation_id="session-456")
        a = _one_event(eval_events).log_record.attributes
        assert a["opensearch.evaluation.level"] == "session"
        assert a["gen_ai.conversation.id"] == "session-456"

    def test_explicit_level_overrides_default(self, eval_events):
        score(name="t", trace_id=TRACE_ID, span_id=SPAN_ID, level="trace")
        a = _one_event(eval_events).log_record.attributes
        assert a["opensearch.evaluation.level"] == "trace"

    def test_unknown_enum_recorded_with_warning(self, eval_events, caplog):
        score(name="t", evaluator_type="robot", direction="up", level="galaxy")
        a = _one_event(eval_events).log_record.attributes
        assert a["opensearch.evaluation.evaluator.type"] == "robot"
        assert "evaluator_type" in caplog.text

    def test_long_explanation_not_truncated_at_500(self, eval_events):
        score(name="t", explanation="x" * 5000)
        a = _one_event(eval_events).log_record.attributes
        assert len(a["gen_ai.evaluation.explanation"]) == 5000

    def test_explanation_safety_cap(self, eval_events):
        score(name="t", explanation="x" * (EXPLANATION_MAX_CHARS * 2))
        expl = _one_event(eval_events).log_record.attributes["gen_ai.evaluation.explanation"]
        assert len(expl) == EXPLANATION_MAX_CHARS == 8192
        assert expl.endswith("...[truncated]")

    def test_attributes_passthrough(self, eval_events):
        score(
            name="t",
            value=0.9,
            attributes={"test.suite.run.id": "run_001", "test.case.id": "case_001"},
        )
        a = _one_event(eval_events).log_record.attributes
        assert a["test.suite.run.id"] == "run_001"
        assert a["test.case.id"] == "case_001"


# ---------------------------------------------------------------------------
# Carrier selection and the deprecated span carrier
# ---------------------------------------------------------------------------


class TestCarrierSelection:
    def test_invalid_carrier_raises(self):
        with pytest.raises(ValueError, match="Unknown carrier"):
            score(name="t", carrier="kafka")  # type: ignore[arg-type]

    def test_env_var_span(self, monkeypatch, exporter, eval_events):
        monkeypatch.setenv("OPENSEARCH_GENAI_EVAL_CARRIER", "span")
        score(name="t", value=0.5)
        assert len(exporter.get_finished_spans()) == 1
        assert eval_events() == []

    def test_env_var_both(self, monkeypatch, exporter, eval_events):
        monkeypatch.setenv("OPENSEARCH_GENAI_EVAL_CARRIER", "BOTH")
        score(name="t", value=0.5)
        assert len(exporter.get_finished_spans()) == 1
        assert len(eval_events()) == 1

    def test_env_var_invalid_uses_event(self, monkeypatch, exporter, eval_events):
        monkeypatch.setenv("OPENSEARCH_GENAI_EVAL_CARRIER", "nope")
        score(name="t", value=0.5)
        assert exporter.get_finished_spans() == ()
        assert len(eval_events()) == 1

    def test_param_overrides_env(self, monkeypatch, exporter, eval_events):
        monkeypatch.setenv("OPENSEARCH_GENAI_EVAL_CARRIER", "span")
        score(name="t", value=0.5, carrier="event")
        assert exporter.get_finished_spans() == ()
        assert len(eval_events()) == 1

    def test_both_emits_event_and_span(self, exporter, eval_events):
        score(name="t", value=0.5, trace_id=TRACE_ID, span_id=SPAN_ID, carrier="both")
        span = exporter.get_finished_spans()[0]
        rec = _one_event(eval_events)
        # The event is parented to the evaluated span, not the score span.
        assert _ids(rec) == (TRACE_ID, SPAN_ID)
        assert format(span.parent.span_id, "016x") == SPAN_ID

    def test_both_with_active_span_parents_event_to_evaluated_span(self, exporter, eval_events):
        tracer = trace.get_tracer("test")
        with tracer.start_as_current_span("chat") as evaluated:
            score(name="t", value=0.5, carrier="both")
            sc = evaluated.get_span_context()
        rec = _one_event(eval_events)
        assert rec.log_record.span_id == sc.span_id

    def test_deprecation_warning_once(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            score(name="a", carrier="span")
            score(name="b", carrier="span")
            score(name="c", carrier="both")
        deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert len(deprecations) == 1
        assert "deprecated" in str(deprecations[0].message)

    def test_deprecation_warning_from_env(self, monkeypatch):
        monkeypatch.setenv("OPENSEARCH_GENAI_EVAL_CARRIER", "span")
        with pytest.warns(DeprecationWarning, match="span carrier"):
            score(name="a")

    def test_no_warning_for_event_carrier(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            score(name="a")
        assert not [w for w in caught if issubclass(w.category, DeprecationWarning)]


class TestSpanCarrier:
    def test_span_name_and_operation(self, exporter):
        score(name="relevance", value=0.9, carrier="span")
        span = exporter.get_finished_spans()[0]
        assert span.name == "evaluation relevance"
        assert span.attributes["gen_ai.operation.name"] == "evaluation"
        assert span.attributes["gen_ai.evaluation.name"] == "relevance"
        assert span.attributes["gen_ai.evaluation.score.value"] == 0.9

    def test_span_event(self, exporter):
        score(name="helpfulness", value=0.83, label="ok", response_id="r1", carrier="span")
        span = exporter.get_finished_spans()[0]
        assert len(span.events) == 1
        event = span.events[0]
        assert event.name == "gen_ai.evaluation.result"
        assert event.attributes["gen_ai.evaluation.score.label"] == "ok"
        assert event.attributes["gen_ai.response.id"] == "r1"

    def test_span_level_parent(self, exporter):
        score(name="accuracy", value=0.95, trace_id=TRACE_ID, span_id=SPAN_ID, carrier="span")
        span = exporter.get_finished_spans()[0]
        assert format(span.context.trace_id, "032x") == TRACE_ID
        assert format(span.parent.span_id, "016x") == SPAN_ID

    def test_trace_level_no_derived_parent(self, exporter):
        """Regression: the span carrier no longer invents a root span id."""
        score(name="relevance", value=0.92, trace_id=TRACE_ID, carrier="span")
        span = exporter.get_finished_spans()[0]
        assert span.parent is None
        assert span.attributes["opensearch.evaluation.trace_id"] == TRACE_ID
        assert span.attributes["opensearch.evaluation.level"] == "trace"

    def test_active_span_parent(self, exporter):
        tracer = trace.get_tracer("test")
        with tracer.start_as_current_span("chat") as evaluated:
            score(name="t", carrier="span")
            sc = evaluated.get_span_context()
        score_span = next(s for s in exporter.get_finished_spans() if s.name == "evaluation t")
        assert score_span.parent.span_id == sc.span_id

    def test_explanation_truncated_at_500(self, exporter):
        score(name="t", explanation="x" * 1000, carrier="span")
        span = exporter.get_finished_spans()[0]
        assert len(span.attributes["gen_ai.evaluation.explanation"]) == 500

    def test_new_attributes_mirrored(self, exporter):
        score(name="t", evaluator="j", evaluator_type="code", carrier="span")
        span = exporter.get_finished_spans()[0]
        assert span.attributes["opensearch.evaluation.evaluator.name"] == "j"
        assert span.attributes["opensearch.evaluation.evaluator.type"] == "code"

    def test_attributes_passthrough(self, exporter):
        score(name="t", attributes={"test.case.id": "c1"}, carrier="span")
        span = exporter.get_finished_spans()[0]
        assert span.attributes["test.case.id"] == "c1"


# ---------------------------------------------------------------------------
# _parse_hex helper
# ---------------------------------------------------------------------------


class TestParseHex:
    def test_plain_hex(self):
        assert _parse_hex("ff") == 255

    def test_0x_prefix(self):
        assert _parse_hex("0xff") == 255

    def test_uppercase_0x_prefix(self):
        assert _parse_hex("0XFF") == 255

    def test_invalid_returns_none(self):
        assert _parse_hex("not-hex!") is None

    def test_empty_string_returns_zero(self):
        assert _parse_hex("") == 0
