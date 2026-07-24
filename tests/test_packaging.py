"""Smoke tests for the package surface (``convilyn_edge``).

These are the PR1 "does it assemble" checks: version present, every advertised
public symbol importable, ``py.typed`` shipped, zero heavy imports.
"""

from __future__ import annotations

import re
from pathlib import Path

import convilyn_edge


def test_version_is_pep440_prerelease():
    assert re.fullmatch(r"\d+\.\d+\.\d+b\d+", convilyn_edge.__version__)


def test_every_public_symbol_is_importable():
    # A missing name in __all__ is a packaging regression — assert each resolves.
    missing = [name for name in convilyn_edge.__all__ if not hasattr(convilyn_edge, name)]

    assert missing == []


def test_seven_primitives_are_exported():
    seven = {
        "EventSource",
        "Normalizer",
        "StateProvider",
        "DeterministicOperator",
        "ModelOperator",
        "HumanReview",
        "ActionSink",
    }

    assert seven <= set(convilyn_edge.__all__)


def test_py_typed_marker_ships():
    marker = Path(convilyn_edge.__file__).parent / "py.typed"

    assert marker.is_file()


def test_no_runtime_dependency_on_backend_or_other_sdk():
    # The zero-dep / no-cross-import invariant: importing the package must not
    # have pulled in backend-api or a sibling SDK.
    import sys

    leaked = [m for m in sys.modules if m.startswith(("app.", "convilyn.", "backend"))]

    assert leaked == []
