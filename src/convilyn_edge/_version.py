"""Single source of truth for ``convilyn-edge``'s package version.

Read by both:

* ``pyproject.toml`` via ``[tool.hatch.version]`` (so the wheel metadata +
  ``importlib.metadata.version("convilyn-edge")`` agree)
* ``convilyn_edge.__init__`` for the in-process ``__version__`` attribute.

The SDK versions its *surface* independently of any device / adapter /
workflow / runtime version — this is only the
Core API + SPI version. Bump in CHANGELOG-bumping commits; never edit a
hardcoded constant in two places.
"""

__version__ = "0.1.0b20"
