# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
#
# Modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

"""AWS SigV4 authentication with opensearch-genai-observability-sdk-py.

When your Data Prepper or OpenSearch Ingestion pipeline is hosted on AWS,
pass AWSSigV4OTLPExporter directly to register() via the exporter= parameter.

OpenSearch Ingestion (OSIS) has one pipeline per signal: traces go to the
traces pipeline, evaluation events (score(), evaluate(), Benchmark) go to a
logs pipeline. Pass the logs pipeline URL as logs_endpoint=.

Requires: pip install opensearch-genai-observability-sdk-py[aws]
"""

from opensearch_genai_observability_sdk_py import Op, observe, register, score
from opensearch_genai_observability_sdk_py.exporters import AWSSigV4OTLPExporter

TRACES_ENDPOINT = "https://my-traces-pipeline.us-east-1.osis.amazonaws.com/v1/traces"
LOGS_ENDPOINT = "https://my-logs-pipeline.us-east-1.osis.amazonaws.com/v1/logs"

# Pass AWSSigV4OTLPExporter explicitly.
# Uses the standard boto3 credential chain (env vars, ~/.aws/credentials, IAM role).
# The log exporter is created automatically with the same service and region.
provider = register(
    project_name="my-llm-app",
    exporter=AWSSigV4OTLPExporter(
        endpoint=TRACES_ENDPOINT,
        service="osis",  # "osis" for OpenSearch Ingestion Service
        # region="us-east-1",   # optional, auto-detected from botocore if not set
    ),
    logs_endpoint=LOGS_ENDPOINT,
)


@observe(name="qa_pipeline", op=Op.INVOKE_AGENT)
def run(question: str) -> str:
    answer = f"Answer to: {question}"
    # Inline evaluation: the event is parented to the active qa_pipeline span.
    score("relevance", 0.9, label="relevant", evaluator_type="human")
    return answer


if __name__ == "__main__":
    print(run("What is OpenSearch?"))
    provider.force_flush()
