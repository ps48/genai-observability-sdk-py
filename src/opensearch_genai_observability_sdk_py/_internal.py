# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
#
# Modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

"""Shared internal utilities for the SDK.

Not part of the public API — subject to change without notice.
"""

from __future__ import annotations

import logging
from importlib.metadata import PackageNotFoundError, version
from urllib.parse import urlparse, urlunparse

logger = logging.getLogger(__name__)

# Instrumentation scope used when the SDK itself emits telemetry (for
# example evaluation events without a named evaluator).
SDK_SCOPE_NAME = "opensearch-genai-observability-sdk-py"

try:
    SDK_VERSION: str | None = version(SDK_SCOPE_NAME)
except PackageNotFoundError:  # pragma: no cover - only when running from a bare checkout
    SDK_VERSION = None

# Exact attribute keys written by the SDK — user metadata must not use these.
_RESERVED_KEYS = frozenset(
    {
        "test.suite.name",
        "test.suite.run.id",
        "test.suite.run.status",
        "test.case.id",
        "test.case.name",
        "test.case.result.status",
        "test.case.error",
        "test.case.input",
        "test.case.output",
        "test.case.expected",
        "opensearch.test_case.input",
        "opensearch.test_case.output",
        "opensearch.test_case.expected",
        "opensearch.test_case.error",
        "opensearch.test_case.duration_ms",
        "opensearch.test_suite_run.case_count",
        "opensearch.test_suite_run.pass_count",
        "opensearch.test_suite_run.fail_count",
        "opensearch.test_suite_run.error_count",
        "opensearch.test_suite_run.duration_ms",
        "gen_ai.operation.name",
    }
)


def parse_hex(value: str) -> int | None:
    """Parse a hex string (with or without 0x prefix) to int.

    Returns ``None`` on failure.
    """
    try:
        return int(value.lstrip("0x").lstrip("0X") or "0", 16)
    except ValueError:
        return None


def validate_metadata_keys(metadata: dict, *, context: str = "") -> None:
    """Warn if metadata keys collide with SDK-reserved attribute keys."""
    for key in metadata:
        if key in _RESERVED_KEYS:
            logger.warning(
                "%sMetadata key '%s' is reserved by the SDK and will be ignored",
                f"{context}: " if context else "",
                key,
            )


def derive_logs_endpoint(traces_endpoint: str) -> str:
    """Derive the OTLP logs endpoint from an OTLP traces endpoint.

    Mirrors how the traces endpoint is built: the trailing ``/v1/traces``
    (or ``/traces``) signal path is swapped for ``/v1/logs`` (or ``/logs``),
    keeping scheme, host, port and any path prefix (for example
    ``/opentelemetry`` for Data Prepper). gRPC endpoints carry no signal path
    and are returned unchanged.

    Examples::

        http://localhost:4318/v1/traces                  -> http://localhost:4318/v1/logs
        http://localhost:21890/opentelemetry/v1/traces   -> .../opentelemetry/v1/logs
        https://x.us-east-1.osis.amazonaws.com/v1/traces -> .../v1/logs
        grpc://localhost:4317                            -> grpc://localhost:4317
    """
    parsed = urlparse(traces_endpoint)
    if parsed.scheme.lower() in ("grpc", "grpcs"):
        return traces_endpoint

    path = parsed.path.rstrip("/")
    if path.endswith("/traces"):
        new_path = path[: -len("traces")] + "logs"
    elif path == "":
        new_path = "/v1/logs"
    else:
        logger.warning(
            "Cannot derive an OTLP logs endpoint from '%s' (no /v1/traces suffix); "
            "appending /v1/logs. Pass logs_endpoint= to override.",
            traces_endpoint,
        )
        new_path = path + "/v1/logs"
    return urlunparse(parsed._replace(path=new_path))
