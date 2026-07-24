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
# target root — ``root / "adapters" / "/tmp/x"`` == ``/tmp/x``), and newlines / ``: ``
# that would inject into the generated YAML.
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

_WORKFLOW_YAML = """\
# Edge workflow '{name}' (a typed DAG, never an agent prompt).
apiVersion: convilyn.io/v1alpha1
kind: EdgeWorkflow
metadata:
  name: {name}
  version: 0.1.0
trigger:
  event: device.example.received
steps: []
"""

_WORKFLOW_README = """\
# {name} workflow

Compose deterministic operators, a model operator, human review and gated
actions over device events. Each step declares its input/output schema,
placement, deadline and required permissions.

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
            "workflow.yaml": _WORKFLOW_YAML.format(name=name),
            "README.md": _WORKFLOW_README.format(name=name),
        },
    )


__all__ = ["scaffold_adapter", "scaffold_workflow"]
