# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
#
# Modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

"""Tests for opensearch_genai_observability_sdk_py.register."""

import importlib
from unittest.mock import MagicMock, patch

import pytest

from opensearch_genai_observability_sdk_py.register import _infer_protocol, _resolve_endpoint

# importlib.import_module is needed because opensearch_genai_observability_sdk_py.__init__ imports
# `register` (the function) into the package namespace, shadowing the submodule name.
# Using importlib bypasses that and returns the actual module object.
_register_mod = importlib.import_module("opensearch_genai_observability_sdk_py.register")


# ---------------------------------------------------------------------------
# _infer_protocol
# ---------------------------------------------------------------------------


class TestInferProtocol:
    """Unit tests for protocol inference and validation."""

    def test_http_scheme(self):
        assert _infer_protocol("http://localhost:4318/v1/traces", None) == "http"

    def test_https_scheme(self):
        assert _infer_protocol("https://collector.example.com/v1/traces", None) == "http"

    def test_grpc_scheme(self):
        assert _infer_protocol("grpc://localhost:4317", None) == "grpc"

    def test_grpcs_scheme(self):
        assert _infer_protocol("grpcs://collector.example.com:4317", None) == "grpc"

    def test_explicit_grpc_overrides_http_scheme(self):
        """http:// is ambiguous — explicit grpc wins."""
        assert _infer_protocol("http://localhost:4317", "grpc") == "grpc"

    def test_explicit_http_protobuf_accepted(self):
        assert _infer_protocol("http://localhost:4318/v1/traces", "http/protobuf") == "http"

    def test_contradicting_http_on_grpc_scheme_raises(self):
        """grpc:// unambiguously means gRPC — contradicting it is an error."""
        with pytest.raises(ValueError, match="Conflicting protocol"):
            _infer_protocol("grpc://localhost:4317", "http")

    def test_contradicting_http_protobuf_on_grpcs_scheme_raises(self):
        with pytest.raises(ValueError, match="Conflicting protocol"):
            _infer_protocol("grpcs://localhost:4317", "http/protobuf")

    def test_http_json_raises(self):
        with pytest.raises(ValueError, match="http/json.*not supported"):
            _infer_protocol("http://localhost:4318/v1/traces", "http/json")

    def test_unknown_protocol_raises(self):
        with pytest.raises(ValueError, match="Unknown OTLP protocol"):
            _infer_protocol("http://localhost:4318/v1/traces", "thrift")


# ---------------------------------------------------------------------------
# _resolve_endpoint
# ---------------------------------------------------------------------------


class TestResolveEndpoint:
    """Unit tests for endpoint resolution priority."""

    def test_code_endpoint_wins(self, monkeypatch):
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://traces:4318/v1/traces")
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://base:4318")
        assert (
            _resolve_endpoint("http://explicit:4318/v1/traces") == "http://explicit:4318/v1/traces"
        )

    def test_traces_endpoint_env_var(self, monkeypatch):
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://traces:4318/v1/traces")
        assert _resolve_endpoint(None) == "http://traces:4318/v1/traces"

    def test_base_endpoint_env_var_appends_path(self, monkeypatch):
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
        assert _resolve_endpoint(None) == "http://localhost:4318/v1/traces"

    def test_base_endpoint_trailing_slash_handled(self, monkeypatch):
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318/")
        assert _resolve_endpoint(None) == "http://localhost:4318/v1/traces"

    def test_traces_endpoint_takes_priority_over_base(self, monkeypatch):
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://traces:4318/v1/traces")
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://base:4318")
        assert _resolve_endpoint(None) == "http://traces:4318/v1/traces"

    def test_data_prepper_default(self, monkeypatch):
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
        assert _resolve_endpoint(None) == _register_mod.DEFAULT_ENDPOINT


# ---------------------------------------------------------------------------
# register()
# ---------------------------------------------------------------------------


class TestRegister:
    """Tests for the register() function."""

    @patch.object(_register_mod, "_create_http_exporter")
    def test_register_returns_tracer_provider(self, mock_create_http):
        from opentelemetry.sdk.trace import TracerProvider

        from opensearch_genai_observability_sdk_py.register import register

        mock_create_http.return_value = MagicMock()
        provider = register(set_global=False, auto_instrument=False)
        assert isinstance(provider, TracerProvider)

    @patch.object(_register_mod, "_create_http_exporter")
    def test_register_uses_default_endpoint(self, mock_create_http, monkeypatch):
        from opensearch_genai_observability_sdk_py.register import (
            DEFAULT_ENDPOINT,
            register,
        )

        monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
        mock_create_http.return_value = MagicMock()
        register(set_global=False, auto_instrument=False)
        mock_create_http.assert_called_once_with(DEFAULT_ENDPOINT, None)

    @patch.object(_register_mod, "_create_http_exporter")
    def test_register_uses_provided_endpoint(self, mock_create_http):
        from opensearch_genai_observability_sdk_py.register import register

        mock_create_http.return_value = MagicMock()
        register(
            endpoint="http://my-collector:4318/v1/traces",
            set_global=False,
            auto_instrument=False,
        )
        mock_create_http.assert_called_once_with("http://my-collector:4318/v1/traces", None)

    @patch.object(_register_mod, "_create_http_exporter")
    def test_register_uses_otel_traces_endpoint_env_var(self, mock_create_http, monkeypatch):
        from opensearch_genai_observability_sdk_py.register import register

        monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://otel:4318/v1/traces")
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
        mock_create_http.return_value = MagicMock()
        register(set_global=False, auto_instrument=False)
        mock_create_http.assert_called_once_with("http://otel:4318/v1/traces", None)

    @patch.object(_register_mod, "_create_http_exporter")
    def test_register_uses_otel_base_endpoint_env_var(self, mock_create_http, monkeypatch):
        from opensearch_genai_observability_sdk_py.register import register

        monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
        mock_create_http.return_value = MagicMock()
        register(set_global=False, auto_instrument=False)
        mock_create_http.assert_called_once_with("http://localhost:4318/v1/traces", None)

    @patch.object(_register_mod, "_create_grpc_exporter")
    def test_register_uses_grpc_for_grpc_scheme(self, mock_create_grpc):
        from opensearch_genai_observability_sdk_py.register import register

        mock_create_grpc.return_value = MagicMock()
        register(
            endpoint="grpc://localhost:4317",
            set_global=False,
            auto_instrument=False,
        )
        mock_create_grpc.assert_called_once()

    @patch.object(_register_mod, "_create_grpc_exporter")
    def test_register_uses_grpc_via_protocol_env_var(self, mock_create_grpc, monkeypatch):
        from opensearch_genai_observability_sdk_py.register import register

        monkeypatch.setenv("OTEL_EXPORTER_OTLP_PROTOCOL", "grpc")
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL", raising=False)
        mock_create_grpc.return_value = MagicMock()
        register(
            endpoint="http://localhost:4317",
            set_global=False,
            auto_instrument=False,
        )
        mock_create_grpc.assert_called_once()

    @patch.object(_register_mod, "_create_grpc_exporter")
    def test_register_traces_protocol_env_var_takes_priority(self, mock_create_grpc, monkeypatch):
        from opensearch_genai_observability_sdk_py.register import register

        monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL", "grpc")
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf")
        mock_create_grpc.return_value = MagicMock()
        register(
            endpoint="http://localhost:4317",
            set_global=False,
            auto_instrument=False,
        )
        mock_create_grpc.assert_called_once()

    @patch.object(_register_mod, "_create_http_exporter")
    def test_register_sets_service_version(self, mock_create_http):
        from opentelemetry.sdk.trace import TracerProvider

        from opensearch_genai_observability_sdk_py.register import register

        mock_create_http.return_value = MagicMock()
        provider = register(service_version="1.0.0", set_global=False, auto_instrument=False)
        assert isinstance(provider, TracerProvider)
        resource_attrs = dict(provider.resource.attributes)
        assert resource_attrs["service.version"] == "1.0.0"

    @patch.object(_register_mod, "_create_http_exporter")
    def test_register_service_version_from_env(self, mock_create_http, monkeypatch):
        from opensearch_genai_observability_sdk_py.register import register

        monkeypatch.setenv("OTEL_SERVICE_VERSION", "2.0.0")
        mock_create_http.return_value = MagicMock()
        provider = register(set_global=False, auto_instrument=False)
        resource_attrs = dict(provider.resource.attributes)
        assert resource_attrs["service.version"] == "2.0.0"

    @patch.object(_register_mod, "_create_http_exporter")
    def test_register_service_version_param_over_env(self, mock_create_http, monkeypatch):
        from opensearch_genai_observability_sdk_py.register import register

        monkeypatch.setenv("OTEL_SERVICE_VERSION", "2.0.0")
        mock_create_http.return_value = MagicMock()
        provider = register(service_version="3.0.0", set_global=False, auto_instrument=False)
        resource_attrs = dict(provider.resource.attributes)
        assert resource_attrs["service.version"] == "3.0.0"

    def test_register_uses_custom_exporter(self):
        from opensearch_genai_observability_sdk_py.register import register

        custom_exporter = MagicMock()
        register(exporter=custom_exporter, set_global=False, auto_instrument=False)
        custom_exporter.export.assert_not_called()  # accepted but not called

    @patch.object(_register_mod, "_create_http_exporter")
    def test_register_passes_headers(self, mock_create_http, monkeypatch):
        from opensearch_genai_observability_sdk_py.register import register

        monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
        mock_create_http.return_value = MagicMock()
        register(
            headers={"X-Custom": "value"},
            set_global=False,
            auto_instrument=False,
        )
        mock_create_http.assert_called_once_with(
            _register_mod.DEFAULT_ENDPOINT, {"X-Custom": "value"}
        )


# ---------------------------------------------------------------------------
# Logs pipeline for evaluation events
# ---------------------------------------------------------------------------

_LOG_ENV_VARS = (
    "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT",
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
    "OTEL_EXPORTER_OTLP_LOGS_PROTOCOL",
    "OTEL_EXPORTER_OTLP_PROTOCOL",
    "OTEL_EXPORTER_OTLP_TRACES_PROTOCOL",
)


@pytest.fixture()
def clean_otlp_env(monkeypatch):
    for var in _LOG_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def _in_memory_log_exporter():
    try:
        from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter as InMemoryLogs
    except ImportError:  # pragma: no cover
        from opentelemetry.sdk._logs.export import InMemoryLogExporter as InMemoryLogs
    return InMemoryLogs()


class TestDeriveLogsEndpoint:
    @pytest.mark.parametrize(
        ("traces", "logs"),
        [
            ("http://localhost:4318/v1/traces", "http://localhost:4318/v1/logs"),
            ("http://localhost:4318/v1/traces/", "http://localhost:4318/v1/logs"),
            (
                "http://localhost:21890/opentelemetry/v1/traces",
                "http://localhost:21890/opentelemetry/v1/logs",
            ),
            (
                "https://p-abc.us-east-1.osis.amazonaws.com/v1/traces",
                "https://p-abc.us-east-1.osis.amazonaws.com/v1/logs",
            ),
            ("http://collector:4318", "http://collector:4318/v1/logs"),
            ("grpc://localhost:4317", "grpc://localhost:4317"),
            ("grpcs://collector:4317", "grpcs://collector:4317"),
        ],
    )
    def test_derivation(self, traces, logs):
        from opensearch_genai_observability_sdk_py._internal import derive_logs_endpoint

        assert derive_logs_endpoint(traces) == logs

    def test_unknown_path_warns(self, caplog):
        from opensearch_genai_observability_sdk_py._internal import derive_logs_endpoint

        assert derive_logs_endpoint("http://h:1/custom") == "http://h:1/custom/v1/logs"
        assert "Cannot derive" in caplog.text


class TestResolveLogsEndpoint:
    def test_param_wins(self, clean_otlp_env, monkeypatch):
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_LOGS_ENDPOINT", "http://env:4318/v1/logs")
        assert (
            _register_mod._resolve_logs_endpoint("http://x:4318/v1/traces", "http://p/v1/logs")
            == "http://p/v1/logs"
        )

    def test_derived_from_endpoint_param(self, clean_otlp_env, monkeypatch):
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_LOGS_ENDPOINT", "http://env:4318/v1/logs")
        assert (
            _register_mod._resolve_logs_endpoint("http://x:4318/v1/traces", None)
            == "http://x:4318/v1/logs"
        )

    def test_logs_env(self, clean_otlp_env, monkeypatch):
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_LOGS_ENDPOINT", "http://env:4318/v1/logs")
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://base:4318")
        assert _register_mod._resolve_logs_endpoint(None, None) == "http://env:4318/v1/logs"

    def test_base_env(self, clean_otlp_env, monkeypatch):
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://base:4318/")
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://t:4318/v1/traces")
        assert _register_mod._resolve_logs_endpoint(None, None) == "http://base:4318/v1/logs"

    def test_derived_from_traces_env(self, clean_otlp_env, monkeypatch):
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://t:4318/v1/traces")
        assert _register_mod._resolve_logs_endpoint(None, None) == "http://t:4318/v1/logs"

    def test_default(self, clean_otlp_env):
        assert _register_mod._resolve_logs_endpoint(None, None) == (
            "http://localhost:21890/opentelemetry/v1/logs"
        )
        assert _register_mod.DEFAULT_LOGS_ENDPOINT.endswith("/opentelemetry/v1/logs")

    def test_logs_protocol_env(self, monkeypatch):
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_LOGS_PROTOCOL", "grpc")
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf")
        assert _register_mod._resolve_logs_protocol(None) == "grpc"
        assert _register_mod._resolve_logs_protocol("http") == "http"


class TestCreateLogExporter:
    def test_http(self):
        from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter

        exp = _register_mod._create_log_exporter("http://localhost:4318/v1/logs", None, None)
        assert isinstance(exp, OTLPLogExporter)
        assert exp._endpoint == "http://localhost:4318/v1/logs"

    def test_grpc(self):
        from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter

        exp = _register_mod._create_log_exporter("grpc://localhost:4317", None, None)
        assert isinstance(exp, OTLPLogExporter)
        exp.shutdown()


@pytest.fixture()
def fresh_globals():
    """Pretend no global LoggerProvider is set and capture what register() installs."""
    with (
        patch.object(_register_mod, "_global_logger_provider_is_set", return_value=False),
        patch.object(_register_mod._logs, "set_logger_provider") as set_lp,
        patch.object(_register_mod.trace, "set_tracer_provider"),
    ):
        yield set_lp


class TestRegisterLogs:
    @patch.object(_register_mod, "_create_log_exporter")
    @patch.object(_register_mod, "_create_http_exporter")
    def test_default_logs_exporter_mirrors_traces(
        self, mock_http, mock_log, fresh_globals, clean_otlp_env
    ):
        from opentelemetry.sdk._logs import LoggerProvider

        from opensearch_genai_observability_sdk_py.register import register

        mock_http.return_value = MagicMock()
        mock_log.return_value = _in_memory_log_exporter()
        register(
            endpoint="http://my-collector:4318/v1/traces",
            headers={"Authorization": "Basic x"},
            auto_instrument=False,
        )
        mock_log.assert_called_once_with(
            endpoint="http://my-collector:4318/v1/logs",
            protocol=None,
            headers={"Authorization": "Basic x"},
        )
        fresh_globals.assert_called_once()
        assert isinstance(fresh_globals.call_args.args[0], LoggerProvider)

    @patch.object(_register_mod, "_create_log_exporter")
    @patch.object(_register_mod, "_create_grpc_exporter")
    def test_grpc_protocol_passed_to_logs(self, mock_grpc, mock_log, fresh_globals, clean_otlp_env):
        from opensearch_genai_observability_sdk_py.register import register

        mock_grpc.return_value = MagicMock()
        mock_log.return_value = _in_memory_log_exporter()
        register(endpoint="http://localhost:4317", protocol="grpc", auto_instrument=False)
        assert mock_log.call_args.kwargs["protocol"] == "grpc"

    @patch.object(_register_mod, "_create_log_exporter")
    @patch.object(_register_mod, "_create_http_exporter")
    def test_logs_endpoint_param(self, mock_http, mock_log, fresh_globals, clean_otlp_env):
        from opensearch_genai_observability_sdk_py.register import register

        mock_http.return_value = MagicMock()
        mock_log.return_value = _in_memory_log_exporter()
        register(logs_endpoint="http://logs:4318/v1/logs", auto_instrument=False)
        assert mock_log.call_args.kwargs["endpoint"] == "http://logs:4318/v1/logs"

    @patch.object(_register_mod, "_create_http_exporter")
    def test_events_flow_and_flush_through_tracer_provider(self, mock_http, fresh_globals):
        from opensearch_genai_observability_sdk_py.register import register

        mock_http.return_value = MagicMock()
        log_exp = _in_memory_log_exporter()
        provider = register(log_exporter=log_exp, batch=True, auto_instrument=False)
        logger_provider = fresh_globals.call_args.args[0]

        logger_provider.get_logger("t").emit(
            event_name="gen_ai.evaluation.result", attributes={"gen_ai.evaluation.name": "x"}
        )
        assert provider.force_flush()
        records = log_exp.get_finished_logs()
        assert [r.log_record.event_name for r in records] == ["gen_ai.evaluation.result"]
        # Same resource as traces. ReadableLogRecord.resource (>=1.39) or
        # LogData.log_record.resource (1.38).
        resource = getattr(records[0], "resource", None) or records[0].log_record.resource
        assert resource.attributes["service.name"] == provider.resource.attributes["service.name"]

        with patch.object(logger_provider, "shutdown") as lp_shutdown:
            provider.shutdown()
        lp_shutdown.assert_called_once()

    @patch.object(_register_mod, "_create_http_exporter")
    def test_existing_global_logger_provider_untouched(self, mock_http):
        """conftest installs a global LoggerProvider: register() must reuse it."""
        from opensearch_genai_observability_sdk_py.register import register

        mock_http.return_value = MagicMock()
        with (
            patch.object(_register_mod._logs, "set_logger_provider") as set_lp,
            patch.object(_register_mod.trace, "set_tracer_provider"),
            patch.object(_register_mod, "_create_log_exporter") as mock_log,
        ):
            register(auto_instrument=False)
        set_lp.assert_not_called()
        mock_log.assert_not_called()

    def test_global_logger_provider_detection(self):
        # The test session has an SDK LoggerProvider installed (conftest).
        assert _register_mod._global_logger_provider_is_set() is True
        from opentelemetry._logs._internal import ProxyLoggerProvider

        with patch.object(
            _register_mod._logs, "get_logger_provider", return_value=ProxyLoggerProvider()
        ):
            assert _register_mod._global_logger_provider_is_set() is False

    @patch.object(_register_mod, "_create_http_exporter")
    def test_enable_logs_false(self, mock_http, fresh_globals):
        from opensearch_genai_observability_sdk_py.register import register

        mock_http.return_value = MagicMock()
        register(enable_logs=False, auto_instrument=False)
        fresh_globals.assert_not_called()

    @patch.object(_register_mod, "_create_log_exporter")
    @patch.object(_register_mod, "_create_http_exporter")
    def test_set_global_false_skips_logs(self, mock_http, mock_log, fresh_globals):
        from opensearch_genai_observability_sdk_py.register import register

        mock_http.return_value = MagicMock()
        register(set_global=False, auto_instrument=False)
        fresh_globals.assert_not_called()
        mock_log.assert_not_called()

    def test_custom_exporter_without_log_exporter_warns(self, fresh_globals, caplog):
        from opensearch_genai_observability_sdk_py.register import register

        register(exporter=MagicMock(), auto_instrument=False)
        fresh_globals.assert_not_called()
        assert "log_exporter=" in caplog.text

    def test_custom_exporter_with_log_exporter(self, fresh_globals):
        from opensearch_genai_observability_sdk_py.register import register

        register(
            exporter=MagicMock(), log_exporter=_in_memory_log_exporter(), auto_instrument=False
        )
        fresh_globals.assert_called_once()

    def test_sigv4_exporter_gets_sigv4_log_exporter(self, fresh_globals):
        from opensearch_genai_observability_sdk_py.exporters import AWSSigV4OTLPExporter
        from opensearch_genai_observability_sdk_py.register import register

        span_exp = MagicMock(spec=AWSSigV4OTLPExporter)
        span_exp.create_log_exporter.return_value = _in_memory_log_exporter()
        register(
            exporter=span_exp,
            logs_endpoint="https://logs.osis.amazonaws.com/v1/logs",
            auto_instrument=False,
        )
        span_exp.create_log_exporter.assert_called_once_with(
            endpoint="https://logs.osis.amazonaws.com/v1/logs"
        )
        fresh_globals.assert_called_once()
