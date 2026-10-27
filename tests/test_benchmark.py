# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
#
# Modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

"""Tests for opensearch_genai_observability_sdk_py.benchmark."""

import importlib
import warnings

import pytest
from opentelemetry import trace

from opensearch_genai_observability_sdk_py.benchmark import (
    Benchmark,
    BenchmarkResult,
    BenchmarkSummary,
    EvalScore,
    _make_case_id,
    evaluate,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TRACE_ID = "6ebb9835f43af1552f2cebb9f5165e39"
SPAN_ID = "89829115c2128845"


def _get_spans_by_name(exporter, name):
    return [s for s in exporter.get_finished_spans() if s.name == name]


def _get_root_span(exporter, benchmark_name):
    return next(
        s for s in exporter.get_finished_spans() if s.name == f"test_suite_run {benchmark_name}"
    )


def _get_case_spans(exporter):
    return [s for s in exporter.get_finished_spans() if s.name.startswith("test_case ")]


_score_mod = importlib.import_module("opensearch_genai_observability_sdk_py.score")


@pytest.fixture(autouse=True)
def _reset_carrier(monkeypatch):
    monkeypatch.delenv("OPENSEARCH_GENAI_EVAL_CARRIER", raising=False)
    monkeypatch.setattr(_score_mod, "_span_carrier_warned", False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        yield


# ---------------------------------------------------------------------------
# _make_case_id
# ---------------------------------------------------------------------------


class TestMakeCaseId:
    def test_deterministic(self):
        assert _make_case_id("hello") == _make_case_id("hello")

    def test_different_inputs_different_ids(self):
        assert _make_case_id("hello") != _make_case_id("world")

    def test_length_is_16(self):
        assert len(_make_case_id("test")) == 16

    def test_dict_input(self):
        id1 = _make_case_id({"a": 1, "b": 2})
        id2 = _make_case_id({"b": 2, "a": 1})
        assert id1 == id2  # sort_keys=True

    def test_none_input(self):
        result = _make_case_id(None)
        assert len(result) == 16


# ---------------------------------------------------------------------------
# EvalScore dataclass
# ---------------------------------------------------------------------------


class TestEvalScore:
    def test_minimal(self):
        s = EvalScore(name="accuracy", value=0.95)
        assert s.name == "accuracy"
        assert s.value == 0.95
        assert s.label is None
        assert s.explanation is None
        assert s.metadata is None

    def test_full(self):
        s = EvalScore(
            name="relevance",
            value=0.8,
            label="good",
            explanation="Relevant answer",
            metadata={"model": "gpt-4"},
        )
        assert s.label == "good"
        assert s.explanation == "Relevant answer"
        assert s.metadata == {"model": "gpt-4"}


# ---------------------------------------------------------------------------
# Benchmark — root span structure
# ---------------------------------------------------------------------------


class TestBenchmarkRootSpan:
    def test_creates_root_span(self, exporter):
        with Benchmark(trace_structure=True, name="test-bench"):
            pass
        root = _get_root_span(exporter, "test-bench")
        assert root is not None

    def test_root_span_attributes(self, exporter):
        with Benchmark(trace_structure=True, name="my-bench"):
            pass
        root = _get_root_span(exporter, "my-bench")
        assert root.attributes["test.suite.name"] == "my-bench"
        assert root.attributes["test.suite.run.id"].startswith("run_")
        # Benchmark spans are not evaluation results (deprecated span carrier).
        assert "gen_ai.operation.name" not in root.attributes

    def test_root_span_success_status(self, exporter):
        with Benchmark(trace_structure=True, name="ok-bench") as b:
            b.log(scores={"a": 1.0})
        root = _get_root_span(exporter, "ok-bench")
        assert root.attributes["test.suite.run.status"] == "success"

    def test_root_span_failure_status(self, exporter):
        with Benchmark(trace_structure=True, name="fail-bench") as b:
            b.log(error="boom")
        root = _get_root_span(exporter, "fail-bench")
        assert root.attributes["test.suite.run.status"] == "failure"

    def test_metadata_on_root_span(self, exporter):
        with Benchmark(trace_structure=True, name="meta-bench", metadata={"model": "gpt-4"}):
            pass
        root = _get_root_span(exporter, "meta-bench")
        assert root.attributes["model"] == "gpt-4"

    def test_reserved_metadata_filtered(self, exporter):
        with Benchmark(
            trace_structure=True,
            name="reserved-bench",
            metadata={"test.suite.name": "oops", "model": "gpt-4"},
        ):
            pass
        root = _get_root_span(exporter, "reserved-bench")
        # Reserved key should not overwrite SDK attribute
        assert root.attributes["test.suite.name"] == "reserved-bench"
        # Non-reserved key should be present
        assert root.attributes["model"] == "gpt-4"

    def test_custom_test_prefix_allowed(self, exporter):
        """User metadata with test.* prefix is allowed if not an SDK key."""
        with Benchmark(
            trace_structure=True,
            name="custom-test",
            metadata={"test.custom_field": "hello"},
        ):
            pass
        root = _get_root_span(exporter, "custom-test")
        assert root.attributes["test.custom_field"] == "hello"

    def test_run_id_property(self):
        b = Benchmark(trace_structure=True, name="prop-test")
        assert b.run_id.startswith("run_")
        assert b.name == "prop-test"
        b.close()


# ---------------------------------------------------------------------------
# Benchmark.log() — case spans
# ---------------------------------------------------------------------------


class TestBenchmarkLog:
    def test_creates_case_span(self, exporter):
        with Benchmark(trace_structure=True, name="log-test") as b:
            b.log(input="q1", scores={"a": 1.0})
        cases = _get_case_spans(exporter)
        assert len(cases) == 1

    def test_case_span_attributes(self, exporter):
        with Benchmark(trace_structure=True, name="attr-test") as b:
            b.log(
                input="What is Python?",
                scores={"accuracy": 0.9},
                case_name="python_q",
            )
        case = _get_case_spans(exporter)[0]
        assert case.attributes["test.suite.name"] == "attr-test"
        assert case.attributes["test.suite.run.id"].startswith("run_")
        assert case.attributes["test.case.id"] == _make_case_id("What is Python?")
        assert case.attributes["test.case.name"] == "python_q"
        assert case.attributes["test.case.result.status"] == "pass"

    def test_explicit_case_id(self, exporter):
        with Benchmark(trace_structure=True, name="id-test") as b:
            b.log(case_id="custom_123", scores={"a": 1.0})
        case = _get_case_spans(exporter)[0]
        assert case.attributes["test.case.id"] == "custom_123"

    def test_auto_case_id_from_input(self, exporter):
        with Benchmark(trace_structure=True, name="auto-id") as b:
            b.log(input="hello", scores={"a": 1.0})
        case = _get_case_spans(exporter)[0]
        assert case.attributes["test.case.id"] == _make_case_id("hello")

    def test_error_sets_fail_status(self, exporter):
        with Benchmark(trace_structure=True, name="err-test") as b:
            b.log(input="q1", error="something broke")
        case = _get_case_spans(exporter)[0]
        assert case.attributes["test.case.result.status"] == "fail"

    def test_multiple_cases(self, exporter):
        with Benchmark(trace_structure=True, name="multi") as b:
            b.log(input="q1", scores={"a": 1.0})
            b.log(input="q2", scores={"a": 0.5})
            b.log(input="q3", scores={"a": 0.0})
        cases = _get_case_spans(exporter)
        assert len(cases) == 3

    def test_case_metadata(self, exporter):
        with Benchmark(trace_structure=True, name="meta") as b:
            b.log(input="q1", scores={"a": 1.0}, metadata={"model": "gpt-4"})
        case = _get_case_spans(exporter)[0]
        assert case.attributes["model"] == "gpt-4"

    def test_case_reserved_metadata_filtered(self, exporter):
        with Benchmark(trace_structure=True, name="res-meta") as b:
            b.log(
                input="q1",
                scores={"a": 1.0},
                metadata={"gen_ai.operation.name": "oops", "model": "gpt-4"},
            )
        case = _get_case_spans(exporter)[0]
        # Reserved key is filtered out
        assert "gen_ai.operation.name" not in case.attributes
        assert case.attributes["model"] == "gpt-4"

    def test_case_span_name_uses_case_name(self, exporter):
        with Benchmark(trace_structure=True, name="name-test") as b:
            b.log(input="q1", case_name="greeting")
        assert _get_case_spans(exporter)[0].name == "test_case greeting"

    def test_case_span_name_falls_back_to_case_id(self, exporter):
        with Benchmark(trace_structure=True, name="name-test") as b:
            b.log(input="q1", case_id="c-7")
        assert _get_case_spans(exporter)[0].name == "test_case c-7"

    def test_error_recorded_under_opensearch_namespace(self, exporter):
        with Benchmark(trace_structure=True, name="err-ns") as b:
            b.log(input="q1", error="boom")
        case = _get_case_spans(exporter)[0]
        assert case.attributes["opensearch.test_case.error"] == "boom"
        assert "test.case.error" not in case.attributes

    def test_suite_and_case_span_kinds(self, exporter):
        from opentelemetry.trace import SpanKind

        with Benchmark(trace_structure=True, name="kind-test") as b:
            b.log(input="q1")
        assert _get_root_span(exporter, "kind-test").kind == SpanKind.INTERNAL
        assert _get_case_spans(exporter)[0].kind == SpanKind.INTERNAL

    def test_log_after_close_raises(self):
        b = Benchmark(trace_structure=True, name="closed")
        b.close()
        with pytest.raises(RuntimeError, match="already closed"):
            b.log(input="q1")

    def test_double_close_raises(self):
        b = Benchmark(trace_structure=True, name="double")
        b.close()
        with pytest.raises(RuntimeError, match="already closed"):
            b.close()


# ---------------------------------------------------------------------------
# Benchmark.log() — score events
# ---------------------------------------------------------------------------


class TestBenchmarkScoreEvents:
    def test_one_log_event_per_score(self, exporter, eval_events):
        with Benchmark(trace_structure=True, name="evt-test") as b:
            b.log(input="q1", scores={"accuracy": 0.9, "relevance": 0.8})
        names = sorted(r.log_record.attributes["gen_ai.evaluation.name"] for r in eval_events())
        assert names == ["accuracy", "relevance"]
        # No deprecated span events by default
        case = _get_case_spans(exporter)[0]
        assert [e for e in case.events if e.name == "gen_ai.evaluation.result"] == []

    def test_event_parented_to_case_span(self, exporter, eval_events):
        with Benchmark(trace_structure=True, name="evt-parent") as b:
            b.log(input="q1", scores={"accuracy": 0.95})
        case = _get_case_spans(exporter)[0]
        lr = eval_events()[0].log_record
        assert lr.trace_id == case.context.trace_id
        assert lr.span_id == case.context.span_id
        assert lr.attributes["gen_ai.evaluation.score.value"] == 0.95
        assert lr.attributes["opensearch.evaluation.level"] == "span"

    def test_grouping_attributes_copied(self, eval_events):
        with Benchmark(trace_structure=True, name="evt-group") as b:
            b.log(input="q1", case_id="c1", case_name="greeting", scores={"accuracy": 1.0})
        a = eval_events()[0].log_record.attributes
        assert a["test.suite.name"] == "evt-group"
        assert a["test.suite.run.id"] == b.run_id
        assert a["test.case.id"] == "c1"
        assert a["test.case.name"] == "greeting"
        # Case content never goes on the event
        assert not [k for k in a if k.startswith("opensearch.test_case.")]

    def test_event_parented_to_linked_agent_span(self, eval_events):
        with Benchmark(trace_structure=True, name="evt-link") as b:
            b.log(input="q1", scores={"accuracy": 1.0}, trace_id=TRACE_ID, span_id=SPAN_ID)
        lr = eval_events()[0].log_record
        assert format(lr.trace_id, "032x") == TRACE_ID
        assert format(lr.span_id, "016x") == SPAN_ID

    def test_no_scores_no_events(self, eval_events):
        with Benchmark(trace_structure=True, name="no-score") as b:
            b.log(input="q1")
        assert eval_events() == []

    def test_span_carrier_env(self, monkeypatch, exporter, eval_events):
        monkeypatch.setenv("OPENSEARCH_GENAI_EVAL_CARRIER", "span")
        with Benchmark(trace_structure=True, name="evt-span") as b:
            b.log(input="q1", scores={"accuracy": 0.9})
        case = _get_case_spans(exporter)[0]
        events = [e for e in case.events if e.name == "gen_ai.evaluation.result"]
        assert len(events) == 1
        assert events[0].attributes["gen_ai.evaluation.score.value"] == 0.9
        assert eval_events() == []

    def test_both_carrier_env(self, monkeypatch, exporter, eval_events):
        monkeypatch.setenv("OPENSEARCH_GENAI_EVAL_CARRIER", "both")
        with Benchmark(trace_structure=True, name="evt-both") as b:
            b.log(input="q1", scores={"accuracy": 0.9})
        case = _get_case_spans(exporter)[0]
        assert len([e for e in case.events if e.name == "gen_ai.evaluation.result"]) == 1
        assert len(eval_events()) == 1


# ---------------------------------------------------------------------------
# Benchmark.log() — record_io
# ---------------------------------------------------------------------------


class TestBenchmarkRecordIO:
    def test_record_io_false_by_default(self, exporter):
        with Benchmark(trace_structure=True, name="no-io") as b:
            b.log(input="secret", output="answer", expected="truth")
        case = _get_case_spans(exporter)[0]
        assert not [k for k in case.attributes if k.startswith("opensearch.test_case.")]
        assert "test.case.input" not in case.attributes

    def test_record_io_true(self, exporter):
        with Benchmark(trace_structure=True, name="with-io", record_io=True) as b:
            b.log(input="question", output="answer", expected="truth")
        case = _get_case_spans(exporter)[0]
        assert case.attributes["opensearch.test_case.input"] == '"question"'
        assert case.attributes["opensearch.test_case.output"] == '"answer"'
        assert case.attributes["opensearch.test_case.expected"] == '"truth"'
        # Invented test.* names are gone
        assert "test.case.input" not in case.attributes
        assert "test.case.output" not in case.attributes
        assert "test.case.expected" not in case.attributes

    def test_record_io_dict_input(self, exporter):
        with Benchmark(trace_structure=True, name="dict-io", record_io=True) as b:
            b.log(input={"q": "hello"}, output="world")
        case = _get_case_spans(exporter)[0]
        assert '"q"' in case.attributes["opensearch.test_case.input"]

    def test_record_io_truncation_marker(self, exporter):
        long_input = "x" * 20_000
        with Benchmark(trace_structure=True, name="trunc-io", record_io=True) as b:
            b.log(input=long_input, output="short")
        case = _get_case_spans(exporter)[0]
        assert case.attributes["opensearch.test_case.input"].endswith("...[truncated]")
        assert len(case.attributes["opensearch.test_case.input"]) <= 10_000


# ---------------------------------------------------------------------------
# Benchmark.log() — span links
# ---------------------------------------------------------------------------


class TestBenchmarkSpanLinks:
    def test_trace_and_span_id_creates_link(self, exporter):
        with Benchmark(trace_structure=True, name="link-test") as b:
            b.log(input="q1", trace_id=TRACE_ID, span_id=SPAN_ID)
        case = _get_case_spans(exporter)[0]
        assert len(case.links) == 1
        link = case.links[0]
        assert format(link.context.trace_id, "032x") == TRACE_ID
        assert format(link.context.span_id, "016x") == SPAN_ID

    def test_trace_id_without_span_id_raises(self, exporter):
        with Benchmark(trace_structure=True, name="link-no-span") as b:
            with pytest.raises(ValueError, match="Both trace_id and span_id"):
                b.log(input="q1", trace_id=TRACE_ID)

    def test_span_id_without_trace_id_raises(self, exporter):
        with Benchmark(trace_structure=True, name="link-no-trace") as b:
            with pytest.raises(ValueError, match="Both trace_id and span_id"):
                b.log(input="q1", span_id=SPAN_ID)

    def test_no_trace_id_no_link(self, exporter):
        with Benchmark(trace_structure=True, name="no-link") as b:
            b.log(input="q1")
        case = _get_case_spans(exporter)[0]
        assert len(case.links) == 0

    def test_invalid_trace_id_no_link(self, exporter):
        with Benchmark(trace_structure=True, name="bad-link") as b:
            b.log(input="q1", trace_id="not-hex!", span_id=SPAN_ID)
        case = _get_case_spans(exporter)[0]
        assert len(case.links) == 0


# ---------------------------------------------------------------------------
# Benchmark — case spans are children of root
# ---------------------------------------------------------------------------


class TestBenchmarkParentChild:
    def test_case_spans_are_children_of_root(self, exporter):
        with Benchmark(trace_structure=True, name="parent-test") as b:
            b.log(input="q1", scores={"a": 1.0})
            b.log(input="q2", scores={"a": 0.5})

        root = _get_root_span(exporter, "parent-test")
        cases = _get_case_spans(exporter)

        for case in cases:
            assert case.parent is not None
            assert case.parent.span_id == root.context.span_id
            assert case.context.trace_id == root.context.trace_id


# ---------------------------------------------------------------------------
# Benchmark.close() — summary
# ---------------------------------------------------------------------------


class TestBenchmarkSummary:
    def test_close_returns_summary(self):
        b = Benchmark(trace_structure=True, name="sum-test")
        b.log(input="q1", scores={"accuracy": 1.0, "relevance": 0.8})
        b.log(input="q2", scores={"accuracy": 0.5, "relevance": 0.9})
        summary = b.close()

        assert isinstance(summary, BenchmarkSummary)
        assert summary.benchmark_name == "sum-test"
        assert summary.total_cases == 2
        assert summary.error_count == 0
        assert summary.scores["accuracy"].avg == 0.75
        assert summary.scores["accuracy"].min == 0.5
        assert summary.scores["accuracy"].max == 1.0
        assert summary.scores["accuracy"].count == 2
        assert summary.scores["relevance"].avg == pytest.approx(0.85)

    def test_summary_with_errors(self):
        b = Benchmark(trace_structure=True, name="err-sum")
        b.log(input="q1", scores={"a": 1.0})
        b.log(input="q2", error="failed")
        summary = b.close()
        assert summary.error_count == 1
        assert summary.total_cases == 2

    def test_summary_str_format(self):
        b = Benchmark(trace_structure=True, name="fmt-test")
        b.log(input="q1", scores={"accuracy": 0.9})
        summary = b.close()
        text = str(summary)
        assert "BENCHMARK: fmt-test" in text
        assert "accuracy" in text
        assert "avg=0.90" in text

    def test_prints_summary_to_stdout(self, capsys):
        """close() prints the benchmark summary for developer visibility."""
        with Benchmark(trace_structure=True, name="print-test") as b:
            b.log(input="q1", scores={"a": 1.0})
        captured = capsys.readouterr()
        assert "BENCHMARK: print-test" in captured.out


# ---------------------------------------------------------------------------
# evaluate() — full runner
# ---------------------------------------------------------------------------


class TestEvaluate:
    def test_basic_evaluate(self, exporter):
        def my_task(input_val):
            return f"answer to {input_val}"

        def accuracy(input_val, output, expected) -> EvalScore:
            return EvalScore(
                name="accuracy",
                value=1.0 if expected in output else 0.0,
            )

        result = evaluate(
            name="eval-test",
            task=my_task,
            data=[
                {"input": "q1", "expected": "q1"},
                {"input": "q2", "expected": "q2"},
            ],
            scores=[accuracy],
        )

        assert isinstance(result, BenchmarkResult)
        assert result.summary.total_cases == 2
        assert result.summary.scores["accuracy"].avg == 1.0
        assert len(result.cases) == 2

    def test_evaluate_creates_spans(self, exporter):
        evaluate(
            trace_structure=True,
            name="span-eval",
            task=lambda x: x,
            data=[{"input": "q1"}],
            scores=[],
        )

        root = _get_root_span(exporter, "span-eval")
        assert root is not None
        assert root.attributes["test.suite.name"] == "span-eval"

        case_spans = _get_case_spans(exporter)
        assert len(case_spans) == 1

    def test_evaluate_task_error(self, exporter):
        def failing_task(x):
            raise ValueError("boom")

        result = evaluate(
            name="err-eval",
            task=failing_task,
            data=[{"input": "q1"}],
            scores=[],
        )

        assert result.summary.error_count == 1
        assert result.cases[0].status == "fail"
        assert result.cases[0].error == "boom"

    def test_evaluate_scorer_returns_float(self, exporter):
        def my_scorer(input_val, output, expected):
            return 0.75

        result = evaluate(
            name="float-eval",
            task=lambda x: x,
            data=[{"input": "q1"}],
            scores=[my_scorer],
        )

        assert result.cases[0].scores["my_scorer"] == 0.75

    def test_evaluate_scorer_returns_list(self, exporter):
        def multi_scorer(input_val, output, expected):
            return [
                EvalScore(name="a", value=0.9),
                EvalScore(name="b", value=0.8),
            ]

        result = evaluate(
            name="list-eval",
            task=lambda x: x,
            data=[{"input": "q1"}],
            scores=[multi_scorer],
        )

        assert result.cases[0].scores == {"a": 0.9, "b": 0.8}

    def test_evaluate_record_io(self, exporter):
        evaluate(
            trace_structure=True,
            name="io-eval",
            task=lambda x: f"answer: {x}",
            data=[{"input": "question", "expected": "answer"}],
            scores=[],
            record_io=True,
        )

        case_spans = _get_case_spans(exporter)
        case = case_spans[0]
        assert case.attributes["opensearch.test_case.input"] == '"question"'
        assert case.attributes["opensearch.test_case.expected"] == '"answer"'
        assert '"answer: question"' in case.attributes["opensearch.test_case.output"]

    def test_evaluate_case_id_from_data(self, exporter):
        evaluate(
            trace_structure=True,
            name="cid-eval",
            task=lambda x: x,
            data=[{"input": "q1", "case_id": "my_custom_id"}],
            scores=[],
        )

        case_spans = _get_case_spans(exporter)
        assert case_spans[0].attributes["test.case.id"] == "my_custom_id"

    def test_evaluate_agent_spans_are_children(self, exporter):
        """Agent spans created inside task() should be children of the case span."""
        from opensearch_genai_observability_sdk_py import observe

        @observe(name="my_agent")
        def traced_task(x):
            return f"result: {x}"

        evaluate(
            trace_structure=True,
            name="child-eval",
            task=traced_task,
            data=[{"input": "q1"}],
            scores=[],
        )

        spans = exporter.get_finished_spans()
        agent_span = next(s for s in spans if s.name == "my_agent")
        case_span = next(s for s in spans if s.name.startswith("test_case "))

        # Agent span should be a child of the case span
        assert agent_span.parent is not None
        assert agent_span.parent.span_id == case_span.context.span_id
        assert agent_span.context.trace_id == case_span.context.trace_id

    def test_evaluate_prints_summary_to_stdout(self, capsys):
        """evaluate() prints the summary for developer visibility."""
        evaluate(
            name="print-eval",
            task=lambda x: x,
            data=[{"input": "q1"}],
            scores=[],
        )
        captured = capsys.readouterr()
        assert "BENCHMARK: print-eval" in captured.out

    def test_evaluate_scorer_error_surfaces(self, exporter):
        """A failing scorer should set status to fail and be in scorer_errors."""

        def bad_scorer(input_val, output, expected):
            raise RuntimeError("scorer broke")

        result = evaluate(
            name="scorer-err",
            task=lambda x: x,
            data=[{"input": "q1"}],
            scores=[bad_scorer],
        )

        assert result.cases[0].status == "fail"
        assert result.cases[0].scorer_errors is not None
        assert any("scorer broke" in e for e in result.cases[0].scorer_errors)

    def test_evaluate_emits_events_with_label_and_explanation(self, exporter, eval_events):
        def judge(input_val, output, expected):
            return EvalScore(name="relevance", value=0.7, label="relevant", explanation="ok")

        evaluate(
            trace_structure=True,
            name="evt-eval",
            task=lambda x: x,
            data=[{"input": "q1", "case_id": "c1", "case_name": "first"}],
            scores=[judge],
        )
        case = _get_case_spans(exporter)[0]
        rec = eval_events()[0]
        lr = rec.log_record
        assert lr.span_id == case.context.span_id
        assert lr.attributes["gen_ai.evaluation.name"] == "relevance"
        assert lr.attributes["gen_ai.evaluation.score.label"] == "relevant"
        assert lr.attributes["gen_ai.evaluation.explanation"] == "ok"
        assert lr.attributes["test.suite.name"] == "evt-eval"
        assert lr.attributes["test.case.id"] == "c1"
        assert lr.attributes["test.case.name"] == "first"
        assert lr.attributes["opensearch.evaluation.evaluator.name"] == "judge"
        assert rec.instrumentation_scope.name == "judge"

    def test_evaluate_scorer_error_emits_error_event(self, eval_events):
        def bad_scorer(input_val, output, expected):
            raise RuntimeError("scorer broke")

        evaluate(name="err-evt", task=lambda x: x, data=[{"input": "q1"}], scores=[bad_scorer])
        a = eval_events()[0].log_record.attributes
        assert a["gen_ai.evaluation.name"] == "bad_scorer"
        assert a["error.type"] == "RuntimeError"
        assert "gen_ai.evaluation.score.value" not in a

    def test_evaluate_suite_and_case_structure(self, exporter):
        evaluate(
            trace_structure=True,
            name="struct-eval",
            task=lambda x: x,
            data=[{"input": "q1", "case_name": "one"}],
            scores=[lambda i, o, e: 1.0],
        )
        root = _get_root_span(exporter, "struct-eval")
        case = _get_case_spans(exporter)[0]
        assert case.name == "test_case one"
        assert case.parent.span_id == root.context.span_id
        assert root.attributes["test.suite.run.status"] == "success"
        assert case.attributes["test.case.result.status"] == "pass"
        assert "gen_ai.operation.name" not in case.attributes

    def test_evaluate_accepts_generator(self, exporter):
        """data can be a generator (single-pass iterable)."""

        def gen():
            yield {"input": "q1"}
            yield {"input": "q2"}

        result = evaluate(
            name="gen-eval",
            task=lambda x: x,
            data=gen(),
            scores=[],
        )

        assert result.summary.total_cases == 2

    def test_evaluate_reserved_metadata_filtered(self, exporter):
        evaluate(
            trace_structure=True,
            name="res-meta-eval",
            task=lambda x: x,
            data=[{"input": "q1"}],
            scores=[],
            metadata={"test.suite.name": "oops", "model": "gpt-4"},
        )

        root = _get_root_span(exporter, "res-meta-eval")
        assert root.attributes["test.suite.name"] == "res-meta-eval"
        assert root.attributes["model"] == "gpt-4"


# ---------------------------------------------------------------------------
# Offline runs as log events (default, trace_structure=False)
# ---------------------------------------------------------------------------

CASE_EVENT = "opensearch.test_case.result"
RUN_EVENT = "opensearch.test_suite_run.result"
EVALUATION = "gen_ai.evaluation.result"
GROUPING = ("test.suite.name", "test.suite.run.id", "test.case.id", "test.case.name")


@pytest.fixture()
def events(log_exporter):
    """Return a callable listing emitted records with a given event name."""

    def _get(name):
        return [r for r in log_exporter.get_finished_logs() if r.log_record.event_name == name]

    return _get


def _harness_spans(exporter):
    return [
        s
        for s in exporter.get_finished_spans()
        if s.name.startswith("test_suite_run ") or s.name.startswith("test_case ")
    ]


def _traced_agent():
    from opensearch_genai_observability_sdk_py import observe

    @observe(name="agent")
    def agent(x):
        return _tool(x)

    @observe(name="tool")
    def _tool(x):
        if x == "boom":
            raise ValueError("task broke")
        return f"answer: {x}"

    return agent


def _spans_named(exporter, name):
    return [s for s in exporter.get_finished_spans() if s.name == name]


class TestEvaluateEvents:
    def test_no_harness_spans_by_default(self, exporter):
        evaluate(
            name="no-harness",
            task=_traced_agent(),
            data=[{"input": "q1"}, {"input": "q2"}],
            scores=[lambda i, o, e: 1.0],
        )
        assert _harness_spans(exporter) == []
        agents = _spans_named(exporter, "agent")
        assert len(agents) == 2
        assert all(a.parent is None for a in agents)
        # Each case's agent run is its own trace
        assert agents[0].context.trace_id != agents[1].context.trace_id

    def test_root_span_stamped_children_not(self, exporter):
        result = evaluate(
            name="stamp",
            task=_traced_agent(),
            data=[{"input": "q1", "case_id": "c1", "case_name": "first"}],
            scores=[],
        )
        root = _spans_named(exporter, "agent")[0]
        child = _spans_named(exporter, "tool")[0]
        assert root.attributes["test.suite.name"] == "stamp"
        assert root.attributes["test.suite.run.id"] == result.summary.run_id
        assert root.attributes["test.case.id"] == "c1"
        assert root.attributes["test.case.name"] == "first"
        assert child.parent.span_id == root.context.span_id
        assert not [k for k in child.attributes if k.startswith("test.")]

    def test_case_name_omitted_when_absent(self, exporter, events):
        evaluate(name="no-name", task=_traced_agent(), data=[{"input": "q1"}], scores=[])
        root = _spans_named(exporter, "agent")[0]
        assert "test.case.name" not in root.attributes
        assert "test.case.name" not in events(CASE_EVENT)[0].log_record.attributes

    def test_events_parented_to_agent_root(self, exporter, events):
        def judge(i, o, e):
            return EvalScore(name="relevance", value=0.7, label="relevant", explanation="ok")

        result = evaluate(
            name="parent",
            task=_traced_agent(),
            data=[{"input": "q1", "case_id": "c1", "case_name": "first"}],
            scores=[judge],
        )
        root = _spans_named(exporter, "agent")[0]
        score_rec = events("gen_ai.evaluation.result")[0]
        case_rec = events(CASE_EVENT)[0]
        for rec in (score_rec, case_rec):
            assert rec.log_record.trace_id == root.context.trace_id
            assert rec.log_record.span_id == root.context.span_id
        a = score_rec.log_record.attributes
        for key in GROUPING:
            assert a[key] == root.attributes[key]
        assert a["opensearch.evaluation.level"] == "span"
        assert a["gen_ai.evaluation.score.label"] == "relevant"
        assert score_rec.instrumentation_scope.name == "judge"
        case = result.cases[0]
        assert case.trace_id == format(root.context.trace_id, "032x")
        assert case.span_id == format(root.context.span_id, "016x")

    def test_case_event_shape(self, events):
        evaluate(
            name="shape",
            task=_traced_agent(),
            data=[{"input": "q1", "case_id": "c1"}],
            scores=[lambda i, o, e: 1.0],
            metadata={"model": "gpt-4", "test.suite.name": "oops"},
        )
        rec = events(CASE_EVENT)[0]
        lr = rec.log_record
        a = lr.attributes
        assert a["test.suite.name"] == "shape"
        assert a["test.case.id"] == "c1"
        assert a["test.case.result.status"] == "pass"
        assert "error.type" not in a
        assert isinstance(a["opensearch.test_case.duration_ms"], float)
        assert a["model"] == "gpt-4"
        # record_io is off by default
        assert not [
            k for k in a if k in ("opensearch.test_case.input", "opensearch.test_case.output")
        ]
        assert lr.timestamp is not None
        assert lr.severity_number.value == 9
        assert rec.instrumentation_scope.name == "opensearch-genai-observability-sdk-py"

    def test_case_event_record_io(self, events):
        evaluate(
            name="io",
            task=_traced_agent(),
            data=[{"input": "question", "expected": "answer"}],
            scores=[],
            record_io=True,
        )
        a = events(CASE_EVENT)[0].log_record.attributes
        assert a["opensearch.test_case.input"] == '"question"'
        assert a["opensearch.test_case.expected"] == '"answer"'
        assert a["opensearch.test_case.output"] == '"answer: question"'

    def test_crashed_case_gets_error_event_and_run_continues(self, exporter, events):
        result = evaluate(
            name="crash",
            task=_traced_agent(),
            data=[{"input": "boom", "case_id": "bad"}, {"input": "q2", "case_id": "good"}],
            scores=[lambda i, o, e: 1.0],
        )
        assert [c.case_id for c in result.cases] == ["bad", "good"]
        assert result.cases[0].error == "task broke"
        assert result.cases[0].status == "fail"  # historical value
        assert result.cases[0].error_type == "ValueError"

        by_case = {
            r.log_record.attributes["test.case.id"]: r.log_record for r in events(CASE_EVENT)
        }
        bad = by_case["bad"]
        assert bad.attributes["test.case.result.status"] == "error"
        assert bad.attributes["error.type"] == "ValueError"
        # Still parented to the crashed agent run's root span
        bad_root = next(
            s for s in _spans_named(exporter, "agent") if s.attributes["test.case.id"] == "bad"
        )
        assert bad.span_id == bad_root.context.span_id
        assert by_case["good"].attributes["test.case.result.status"] == "pass"
        # Scorers are skipped for the crashed case only
        scored = [r.log_record.attributes["test.case.id"] for r in events(EVALUATION)]
        assert scored == ["good"]

    def test_scorer_error_marks_case_error(self, events):
        def bad_scorer(i, o, e):
            raise RuntimeError("scorer broke")

        result = evaluate(
            name="scorer-err-evt",
            task=_traced_agent(),
            data=[{"input": "q1"}],
            scores=[lambda i, o, e: 1.0, bad_scorer],
        )
        assert result.cases[0].status == "fail"
        assert result.cases[0].scorer_errors
        a = events(CASE_EVENT)[0].log_record.attributes
        assert a["test.case.result.status"] == "error"
        assert a["error.type"] == "RuntimeError"
        failed = [
            r.log_record.attributes
            for r in events(EVALUATION)
            if r.log_record.attributes["gen_ai.evaluation.name"] == "bad_scorer"
        ]
        assert failed[0]["error.type"] == "RuntimeError"
        run = events(RUN_EVENT)[0].log_record.attributes
        assert run["test.suite.run.status"] == "failure"
        assert run["opensearch.test_suite_run.error_count"] == 1

    def test_untraced_task_events_without_trace_context(self, exporter, events):
        result = evaluate(
            name="untraced",
            task=lambda x: x,
            data=[{"input": "q1", "case_id": "c1"}],
            scores=[lambda i, o, e: 1.0],
        )
        assert exporter.get_finished_spans() == ()
        case_lr = events(CASE_EVENT)[0].log_record
        score_lr = events(EVALUATION)[0].log_record
        for lr in (case_lr, score_lr):
            assert lr.trace_id == 0 and lr.span_id == 0
            assert lr.attributes["test.case.id"] == "c1"
            assert lr.attributes["test.suite.name"] == "untraced"
        assert "opensearch.evaluation.level" not in score_lr.attributes
        assert result.cases[0].trace_id is None

    def test_untraced_task_ignores_ambient_span(self, events):
        tracer = trace.get_tracer("test")
        with tracer.start_as_current_span("outer"):
            evaluate(name="ambient", task=lambda x: x, data=[{"input": "q1"}], scores=[])
        lr = events(CASE_EVENT)[0].log_record
        assert lr.trace_id == 0 and lr.span_id == 0

    def test_agent_run_is_own_trace_inside_active_span(self, exporter):
        tracer = trace.get_tracer("test")
        with tracer.start_as_current_span("outer") as outer:
            evaluate(name="nested", task=_traced_agent(), data=[{"input": "q1"}], scores=[])
        agent = _spans_named(exporter, "agent")[0]
        assert agent.parent is None
        assert agent.context.trace_id != outer.get_span_context().trace_id
        assert agent.attributes["test.suite.name"] == "nested"

    def test_run_event_counts(self, events):
        def scorer(i, o, e):
            if i == "q3":
                raise RuntimeError("nope")
            return 1.0

        result = evaluate(
            name="counts",
            task=_traced_agent(),
            data=[{"input": "q1"}, {"input": "boom"}, {"input": "q3"}, {"input": "q4"}],
            scores=[scorer],
            metadata={"agent_version": "v2"},
        )
        recs = events(RUN_EVENT)
        assert len(recs) == 1
        lr = recs[0].log_record
        a = lr.attributes
        assert lr.trace_id == 0 and lr.span_id == 0
        assert a["test.suite.name"] == "counts"
        assert a["test.suite.run.id"] == result.summary.run_id
        assert a["test.suite.run.status"] == "failure"
        assert a["opensearch.test_suite_run.case_count"] == 4
        assert a["opensearch.test_suite_run.pass_count"] == 2
        assert a["opensearch.test_suite_run.fail_count"] == 0
        assert a["opensearch.test_suite_run.error_count"] == 2
        assert isinstance(a["opensearch.test_suite_run.duration_ms"], float)
        assert a["agent_version"] == "v2"
        assert len(events(CASE_EVENT)) == 4

    def test_run_event_success(self, events):
        evaluate(name="ok", task=lambda x: x, data=[{"input": "q1"}], scores=[])
        a = events(RUN_EVENT)[0].log_record.attributes
        assert a["test.suite.run.status"] == "success"
        assert a["opensearch.test_suite_run.pass_count"] == 1

    def test_span_carrier_without_case_span(self, monkeypatch, exporter, events):
        monkeypatch.setenv("OPENSEARCH_GENAI_EVAL_CARRIER", "span")
        evaluate(
            name="span-carrier",
            task=_traced_agent(),
            data=[{"input": "q1"}],
            scores=[lambda i, o, e: 1.0],
        )
        root = _spans_named(exporter, "agent")[0]
        eval_span = next(
            s for s in exporter.get_finished_spans() if s.name.startswith("evaluation ")
        )
        assert eval_span.parent.span_id == root.context.span_id
        assert events(EVALUATION) == []
        assert len(events(CASE_EVENT)) == 1

    def test_trace_structure_restores_spans_and_emits_events(self, exporter, events):
        result = evaluate(
            name="legacy",
            task=_traced_agent(),
            data=[{"input": "q1", "case_id": "c1"}],
            scores=[lambda i, o, e: 1.0],
            trace_structure=True,
        )
        root = _get_root_span(exporter, "legacy")
        case = _get_case_spans(exporter)[0]
        agent = _spans_named(exporter, "agent")[0]
        assert agent.parent.span_id == case.context.span_id
        # The agent span is not a root here, so it is not stamped
        assert "test.case.id" not in agent.attributes
        case_lr = events(CASE_EVENT)[0].log_record
        score_lr = events(EVALUATION)[0].log_record
        for lr in (case_lr, score_lr):
            assert lr.span_id == case.context.span_id
        assert case_lr.attributes["test.case.result.status"] == "pass"
        run = events(RUN_EVENT)[0].log_record.attributes
        assert run["test.suite.run.id"] == root.attributes["test.suite.run.id"]
        assert result.cases[0].span_id == format(case.context.span_id, "016x")

    def test_spans_outside_evaluate_not_stamped(self, exporter):
        evaluate(name="before", task=_traced_agent(), data=[{"input": "q1"}], scores=[])
        exporter.clear()
        _traced_agent()("q2")
        agent = _spans_named(exporter, "agent")[0]
        assert not [k for k in agent.attributes if k.startswith("test.")]

    def test_online_score_unaffected(self, exporter, log_exporter, events):
        from opensearch_genai_observability_sdk_py import score

        evaluate(name="warmup", task=_traced_agent(), data=[{"input": "q1"}], scores=[])
        exporter.clear()
        log_exporter.clear()
        tracer = trace.get_tracer("test")
        with tracer.start_as_current_span("chat") as span:
            score("relevance", 0.9)
        lr = events(EVALUATION)[-1].log_record
        assert lr.span_id == span.get_span_context().span_id
        assert not [k for k in lr.attributes if k.startswith("test.")]
        assert not [k for k in exporter.get_finished_spans()[0].attributes if k.startswith("test.")]
        assert events(CASE_EVENT) == []


class TestBenchmarkEvents:
    def test_no_spans_by_default(self, exporter):
        with Benchmark(name="b-no-spans") as b:
            b.log(input="q1", scores={"a": 1.0})
        assert exporter.get_finished_spans() == ()

    def test_case_and_score_events_without_link(self, events):
        with Benchmark(name="b-evt", metadata={"model": "gpt-4"}, record_io=True) as b:
            b.log(
                input="q1",
                output="a1",
                expected="e1",
                case_id="c1",
                case_name="one",
                scores={"accuracy": 0.9},
                metadata={"split": "dev"},
                duration_ms=12.5,
            )
        case_lr = events(CASE_EVENT)[0].log_record
        score_lr = events(EVALUATION)[0].log_record
        for lr in (case_lr, score_lr):
            assert lr.trace_id == 0 and lr.span_id == 0
            assert lr.attributes["test.suite.run.id"] == b.run_id
            assert lr.attributes["test.case.id"] == "c1"
            assert lr.attributes["test.case.name"] == "one"
        a = case_lr.attributes
        assert a["test.case.result.status"] == "pass"
        assert a["opensearch.test_case.input"] == '"q1"'
        assert a["opensearch.test_case.output"] == '"a1"'
        assert a["opensearch.test_case.expected"] == '"e1"'
        assert a["opensearch.test_case.duration_ms"] == 12.5
        assert a["model"] == "gpt-4"
        assert a["split"] == "dev"
        # Case content never goes on score events
        assert not [k for k in score_lr.attributes if k.startswith("opensearch.test_case.")]

    def test_events_parented_to_linked_span(self, events):
        with Benchmark(name="b-link") as b:
            b.log(input="q1", scores={"a": 1.0}, trace_id=TRACE_ID, span_id=SPAN_ID)
        for lr in (events(CASE_EVENT)[0].log_record, events(EVALUATION)[0].log_record):
            assert format(lr.trace_id, "032x") == TRACE_ID
            assert format(lr.span_id, "016x") == SPAN_ID

    def test_status_and_run_event(self, events):
        b = Benchmark(name="b-status")
        b.log(input="q1", scores={"a": 1.0})
        b.log(input="q2", status="fail")
        b.log(input="q3", error="boom", error_type="TimeoutError")
        summary = b.close()
        statuses = {
            r.log_record.attributes["test.case.id"]: r.log_record.attributes
            for r in events(CASE_EVENT)
        }
        err = statuses[_make_case_id("q3")]
        assert err["test.case.result.status"] == "error"
        assert err["error.type"] == "TimeoutError"
        assert statuses[_make_case_id("q2")]["test.case.result.status"] == "fail"
        run = events(RUN_EVENT)[0].log_record.attributes
        assert run["test.suite.run.status"] == "failure"
        assert run["opensearch.test_suite_run.case_count"] == 3
        assert run["opensearch.test_suite_run.pass_count"] == 1
        assert run["opensearch.test_suite_run.fail_count"] == 1
        assert run["opensearch.test_suite_run.error_count"] == 1
        assert summary.error_count == 1

    def test_invalid_status_raises(self):
        with Benchmark(name="b-bad-status") as b:
            with pytest.raises(ValueError, match="status must be one of"):
                b.log(input="q1", status="skipped")

    def test_trace_structure_emits_events_too(self, exporter, events):
        with Benchmark(name="b-legacy", trace_structure=True) as b:
            b.log(input="q1", scores={"a": 1.0})
        case = _get_case_spans(exporter)[0]
        assert events(CASE_EVENT)[0].log_record.span_id == case.context.span_id
        assert len(events(RUN_EVENT)) == 1
