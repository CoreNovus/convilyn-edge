"""Scaffold generators for ``convilyn-edge init adapter|workflow``.

Writes a conventional skeleton so a developer adding a device adapter or a
workflow starts from a consistent layout, not a blank directory. Pure stdlib
(pathlib); every file is a commented placeholder the author fills in.
"""

from __future__ import annotations

import re
from pathlib import Path

# A scaffold name is a single path component: alnum-led, then [A-Za-z0-9._-]. This
# rejects path separators, ``..``, absolute/anchored names (which would escape the
# target root — ``root / "adapters" / "/tmp/x"`` == ``/tmp/x``), and newlines /
# quotes / ``: `` that would inject into the generated adapter YAML or the
# workflow.py string literal.
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _validate_name(name: str) -> None:
    if not _SAFE_NAME.match(name):
        raise ValueError(
            f"invalid name {name!r}: use [A-Za-z0-9._-] (alnum-led), no path separators or '..'"
        )


_ADAPTER_YAML = """\
# Device adapter manifest for '{name}' (Device Capability Manifest).
apiVersion: convilyn.io/v1alpha1
kind: DeviceProfile
metadata:
  name: {name}
  vendor: generic
  model: "*"
capabilities:
  properties: {{}}
  events: {{}}
  actions: {{}}
bindings: []
"""

_ADAPTER_README = """\
# {name} adapter

An `EventSource` (+ `Normalizer`) that turns this device's raw output into the
canonical `EventEnvelope`. Implement `convilyn_edge.spi.EventSource` in `src/`.

- `schemas/`         — the canonical event schema(s) this adapter emits
- `src/`             — the adapter implementation
- `fixtures/`        — recorded raw payloads for tests
- `contract-tests/`  — assert the adapter emits the canonical envelope
"""

_WORKFLOW_PY = '''\
"""Edge workflow '{name}' — a runtime ``Pipeline`` composition.

A workflow's only executable form on the device is this Python composition of
the SDK's runtime primitives, driven by ``WorkflowDriver`` — there is no
standalone workflow manifest the runtime consumes. See the reference Solution
Pack (``examples/pet_monitoring`` in the SDK source distribution) for a
complete, working composition.
"""

from convilyn_edge.runtime import Pipeline


def build_pipeline() -> Pipeline:
    """Compose the '{name}' workflow. Every builder call returns a NEW Pipeline."""
    return (
        Pipeline("{name}")
        # .normalize(...)  # raw device payload -> canonical fields
        # .state(...)      # durable aggregation (e.g. ThresholdAggregator)
        # .decide(...)     # deterministic rule table
        # .model(...)      # grounded model node (ContractModelOperator)
        # .review(...)     # human review with an expires_at watchdog
        # .act(...)        # ActionGate-authorized sinks
    )
'''

_WORKFLOW_README = """\
# {name} workflow

Compose deterministic operators, a model operator, human review and gated
actions over device events — fill in the `Pipeline` stages in `workflow.py`.
Each stage declares its input binding, placement, deadline and authorization.

- `workflow.py`    — the executable composition (`build_pipeline()`)
- `schemas/`       — input/output schemas
- `policies/`      — retry / timeout / human-review / fallback policy
- `prompts/`       — model-operator prompt(s)
- `tests/`         — unit tests
- `golden-cases/`  — recorded input → expected decision
- `deployment/`    — resource limits + safety shut-off config
"""

_ADAPTER_DIRS = ("schemas", "src", "fixtures", "contract-tests")
_WORKFLOW_DIRS = ("schemas", "policies", "prompts", "tests", "golden-cases", "deployment")


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _scaffold(base: Path, dirs: tuple[str, ...], files: dict[str, str]) -> Path:
    for directory in dirs:
        (base / directory).mkdir(parents=True, exist_ok=True)
        (base / directory / ".gitkeep").write_text("", encoding="utf-8")
    for filename, content in files.items():
        _write(base / filename, content)
    return base


def scaffold_adapter(root: Path, name: str) -> Path:
    """Create ``<root>/adapters/<name>/`` with the adapter skeleton. Returns it.

    Raises ``ValueError`` for an unsafe ``name`` and ``FileExistsError`` if the
    target already exists (a scaffold never silently overwrites authored files)."""
    _validate_name(name)
    base = root / "adapters" / name
    if base.exists():
        raise FileExistsError(f"{base} already exists — remove it or choose another name")
    return _scaffold(
        base,
        _ADAPTER_DIRS,
        {
            "adapter.yaml": _ADAPTER_YAML.format(name=name),
            "README.md": _ADAPTER_README.format(name=name),
        },
    )


def scaffold_workflow(root: Path, name: str) -> Path:
    """Create ``<root>/workflows/<name>/`` with the workflow skeleton. Returns it.

    Raises ``ValueError`` for an unsafe ``name`` and ``FileExistsError`` if the
    target already exists."""
    _validate_name(name)
    base = root / "workflows" / name
    if base.exists():
        raise FileExistsError(f"{base} already exists — remove it or choose another name")
    return _scaffold(
        base,
        _WORKFLOW_DIRS,
        {
            "workflow.py": _WORKFLOW_PY.format(name=name),
            "README.md": _WORKFLOW_README.format(name=name),
        },
    )


__all__ = ["scaffold_adapter", "scaffold_workflow"]
