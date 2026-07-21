"""``Result[T, E]`` — a typed, exception-free outcome carrier.

The deterministic edge primitives (``Normalizer``, ``DeterministicOperator``)
report failure as a *value*, not by raising. A normalize failure (an unreadable
barcode) or a rule-engine rejection is an expected, data-driven outcome — part
of the contract, not an exceptional control-flow event. Encoding it in the type
means a caller *cannot* forget to handle it: there is no silent ``try/except``
that swallows the failure path.

Design: this is a **tagged union**
of two frozen generic dataclasses — ``Ok(value)`` and ``Err(error)`` — aliased
as ``Result[T, E] = Ok[T] | Err[E]``, *not* a single ``{ok, value, error}``
struct. A single struct leaves ``value: T | None`` un-narrowable, so every
consumer re-checks ``is not None``. The tagged union narrows exhaustively under
static analysis::

    match source_result:
        case Ok(canonical):
            ...   # `canonical` is T here — no None check
        case Err(error):
            ...   # `error` is E here

This module has ZERO imports beyond ``typing`` / ``dataclasses`` — it is the
leaf of the dependency graph.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, NoReturn, TypeVar

T = TypeVar("T")
E = TypeVar("E")
U = TypeVar("U")
F = TypeVar("F")


class _ResultError(Exception):
    """Raised by ``.unwrap()`` / ``.unwrap_err()`` on the wrong variant.

    Escapes the exception-free contract only for a programmer error (unwrapping
    the branch you did not check) — never for the domain failure itself, which
    is carried in ``Err.error``.
    """


@dataclass(frozen=True)
class Ok(Generic[T]):
    """The success variant, carrying ``value: T``."""

    value: T

    # Class-level discriminants — deliberately UNANNOTATED so ``@dataclass`` keeps
    # them off the field list (an annotated ``ClassVar`` is silently treated as a
    # field under ``from __future__ import annotations``). Narrow via ``match``/
    # ``case Ok(v)`` rather than ``if r.is_ok:``.
    is_ok = True
    is_err = False

    def unwrap(self) -> T:
        """Return the contained value."""
        return self.value

    def unwrap_err(self) -> NoReturn:
        """Always raises — an ``Ok`` has no error."""
        raise _ResultError(f"called unwrap_err() on an Ok: {self.value!r}")

    def unwrap_or(self, _default: T) -> T:
        """Return the contained value (the default is unused for ``Ok``)."""
        return self.value

    def map(self, fn: Callable[[T], U]) -> Ok[U]:
        """Apply ``fn`` to the contained value, re-wrapping in ``Ok``."""
        return Ok(fn(self.value))

    def map_err(self, _fn: Callable[[E], F]) -> Ok[T]:
        """No-op for ``Ok`` — there is no error to transform."""
        return self


@dataclass(frozen=True)
class Err(Generic[E]):
    """The failure variant, carrying ``error: E``."""

    error: E

    is_ok = False
    is_err = True

    def unwrap(self) -> NoReturn:
        """Always raises — an ``Err`` has no value."""
        raise _ResultError(f"called unwrap() on an Err: {self.error!r}")

    def unwrap_err(self) -> E:
        """Return the contained error."""
        return self.error

    def unwrap_or(self, default: T) -> T:
        """Return ``default`` — an ``Err`` has no value."""
        return default

    def map(self, _fn: Callable[[T], U]) -> Err[E]:
        """No-op for ``Err`` — there is no value to transform."""
        return self

    def map_err(self, fn: Callable[[E], F]) -> Err[F]:
        """Apply ``fn`` to the contained error, re-wrapping in ``Err``."""
        return Err(fn(self.error))


# The public type. Consumers annotate with ``Result[Canonical, NormalizeError]``
# and construct with ``Ok(value)`` / ``Err(error)``. ``Ok[T] | Err[E]`` yields a
# ``typing.Union`` (``_GenericAlias.__or__``), so it stays subscriptable as
# ``Result[int, str]`` and is safe on Python 3.10.
Result = Ok[T] | Err[E]

__all__ = ["Ok", "Err", "Result"]
