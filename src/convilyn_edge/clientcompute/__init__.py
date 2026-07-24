"""Client-compute — the on-device extractor keystone (contract v1 consumer).

Productizes the device half of the frozen ``client_compute`` interrupt contract:
the cloud delegates the extractor role to the device, the device runs a local
model over its OWN copy of the file and returns grounded anchors, the server
re-grounds them. ``convilyn-edge`` **confirms-and-consumes** the contract.

    from convilyn_edge.clientcompute import (
        ClientComputeBridge, EdgeModelOperator, HttpLocalExtractor,
    )

    operator = EdgeModelOperator(HttpLocalExtractor.from_env(os.environ))
    bridge = ClientComputeBridge(operator, my_file_resolver)
    job = await bridge.handle_if_present(client.goals, job)   # client: AsyncConvilyn
"""

from __future__ import annotations

from convilyn_edge.clientcompute.bridge import (
    ClientComputeBridge,
    ClientComputeError,
    FileTextResolver,
    GoalClientPort,
    GoalJobLike,
    MissingLocalSourceError,
    find_client_compute_interrupt,
)
from convilyn_edge.clientcompute.contract import (
    DEFAULT_MAX_TOTAL_CHARS,
    DEFAULT_MAX_VALUE_CHARS,
    INTERRUPT_TYPE,
    MISSING_SENTINEL,
    STRATEGY_V1,
    AnchorsContract,
    ClientComputeRequest,
    build_resume_answer,
    ground_anchors,
)
from convilyn_edge.clientcompute.engine import (
    ExtractorOutputError,
    ExtractorTransportError,
    HttpLocalExtractor,
    LocalExtractor,
    ModelAvailability,
    build_extract_messages,
    resolve_local_model_tag,
)
from convilyn_edge.clientcompute.operator import EdgeModelOperator, ExtractInput

__all__ = [
    # contract
    "INTERRUPT_TYPE",
    "STRATEGY_V1",
    "MISSING_SENTINEL",
    "DEFAULT_MAX_VALUE_CHARS",
    "DEFAULT_MAX_TOTAL_CHARS",
    "AnchorsContract",
    "ClientComputeRequest",
    "ground_anchors",
    "build_resume_answer",
    # engine
    "LocalExtractor",
    "HttpLocalExtractor",
    "ExtractorTransportError",
    "ExtractorOutputError",
    "ModelAvailability",
    "resolve_local_model_tag",
    "build_extract_messages",
    # operator
    "EdgeModelOperator",
    "ExtractInput",
    # bridge
    "ClientComputeBridge",
    "ClientComputeError",
    "MissingLocalSourceError",
    "FileTextResolver",
    "GoalJobLike",
    "GoalClientPort",
    "find_client_compute_interrupt",
]
