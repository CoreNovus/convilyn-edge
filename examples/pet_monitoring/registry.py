"""An adapter registry — device I/O binds by capability KEY, not by a code branch.

Charter rule 2: *device capabilities bind only via the capability registry + a deterministic
resolver.* Swapping an adapter — camera brand X → Y, notify LINE → SMS — must be a **registry
operation**, never a raw ``if brand == …`` swap inside the workflow. This is the pack-side
registry for that: a logical capability key (``"camera"``, ``"notify.primary"``, …) maps to a
factory that builds the concrete adapter, and the declarative assembly resolves adapters from it
by key. The workflow never names a brand; it names a capability.

Immutable: :meth:`register` returns a NEW registry (never mutates), so a built registry is a
stable declaration.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

#: A factory that builds a concrete adapter (an ``EventSource`` / ``ActionSink`` / ``ModelOperator``
#: …) from call-time arguments. Kept ``Any`` because the registry is capability-agnostic — the
#: resolver's caller knows the concrete type it asked for.
AdapterFactory = Callable[..., Any]


class AdapterRegistry:
    """A frozen key → factory table for device adapters, with a deterministic resolver."""

    def __init__(self, factories: Mapping[str, AdapterFactory] | None = None) -> None:
        self._factories: dict[str, AdapterFactory] = dict(factories or {})

    def register(self, key: str, factory: AdapterFactory) -> AdapterRegistry:
        """Return a NEW registry with ``key`` bound to ``factory`` (immutable; overwrites a dup)."""
        return AdapterRegistry({**self._factories, key: factory})

    def resolve(self, key: str, *args: Any, **kwargs: Any) -> Any:
        """Build the adapter registered under ``key`` — a dict lookup, never a brand branch.

        Raises ``KeyError`` (with the known keys) for an unregistered capability, so a missing
        binding is a loud declaration error, not a silent ``None``.
        """
        factory = self._factories.get(key)
        if factory is None:
            raise KeyError(f"no adapter registered for {key!r}; known: {sorted(self._factories)}")
        return factory(*args, **kwargs)

    def keys(self) -> frozenset[str]:
        """The registered capability keys."""
        return frozenset(self._factories)

    def __contains__(self, key: object) -> bool:
        return key in self._factories


__all__ = ["AdapterFactory", "AdapterRegistry"]
