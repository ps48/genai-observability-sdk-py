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

from opentelemetry import _logs

logger = logging.getLogger(__name__)

# Instrumentation scope used when the SDK itself emits telemetry (for
# example evaluation events without a named evaluator).
SDK_SCOPE_NAME = "opensearch-genai-observability-sdk-py"

try:
    SDK_VERSION: str = version(SDK_SCOPE_NAME)
except PackageNotFoundError:  # pragma: no cover - only when running from a bare checkout
    SDK_VERSION = "0.0.0+unknown"


# Instrumentation scope attributes set on every logger the SDK uses for
# evaluation, test case and test suite run events. Data Prepper derives
# ``derived.evaluation.source = opensearch-sdk`` from them, even when the scope
# name is a user-named evaluator.
SDK_SCOPE_ATTRIBUTES: dict[str, str] = {
    "opensearch.genai.sdk.name": SDK_SCOPE_NAME,
    "opensearch.genai.sdk.version": SDK_VERSION,
}


def event_scope(
    evaluator: str | None = None, evaluator_version: str | None = None
) -> tuple[str, str]:
    """Instrumentation scope ``(name, version)`` for an SDK-emitted event.

    The evaluator identity when an evaluator is named, otherwise the SDK
    scope. Data Prepper and OpenSearch Dashboards derive the grade source
    from the scope, so it is always set.
    """
    if evaluator:
        return evaluator, evaluator_version or ""
    return SDK_SCOPE_NAME, SDK_VERSION


def get_event_logger(
    evaluator: str | None = None, evaluator_version: str | None = None
) -> _logs.Logger:
    """OTel Logger for SDK events: ``event_scope()`` plus ``SDK_SCOPE_ATTRIBUTES``."""
    name, scope_version = event_scope(evaluator, evaluator_version)
    return _logs.get_logger(name, scope_version, attributes=SDK_SCOPE_ATTRIBUTES)


# Exact attribute keys written by the SDK — user metadata must not use these.
_RESERVED_KEYS = frozenset(
    {
        "test.suite.name",
        "test.suite.run.id",
        "test.suite.run.status",
        "test.case.id",
        "test.case.name",
        "test.case.result.status",
        "opensearch.test_case.input",
        "opensearch.test_case.output",
        "opensearch.test_case.expected",
        "opensearch.test_case.duration_ms",
        "opensearch.test_suite_run.case_count",
        "opensearch.test_suite_run.pass_count",
        "opensearch.test_suite_run.fail_count",
        "opensearch.test_suite_run.error_count",
        "opensearch.test_suite_run.duration_ms",
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


def derive_logs_endpoint(traces_endpoint: str) -> str | None:
    """Derive the OTLP logs endpoint from an OTLP traces endpoint, if possible.

    This is the only derivation the SDK does: a traces URL whose path ends
    with ``/v1/traces`` maps to the same base with ``/v1/logs`` (scheme,
    host, port and any path prefix such as ``/opentelemetry`` are kept).
    Any other path returns ``None``: the SDK does not guess.

    Examples::

        http://localhost:4318/v1/traces                  -> http://localhost:4318/v1/logs
        http://localhost:21890/opentelemetry/v1/traces   -> .../opentelemetry/v1/logs
        https://x.us-east-1.osis.amazonaws.com/v1/traces -> .../v1/logs
        http://collector:4318/custom                     -> None
    """
    parsed = urlparse(traces_endpoint)
    path = parsed.path.rstrip("/")
    if not path.endswith("/v1/traces"):
        return None
    return urlunparse(parsed._replace(path=path[: -len("/v1/traces")] + "/v1/logs"))
