"""``python -m convilyn_edge.cli`` — the console-script entry for installs
without pip (a vendored copy on an offline device has no entry-point shim, so
the module itself must be runnable)."""

from convilyn_edge.cli.main import main

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
