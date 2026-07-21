"""Unit tests for the ``Result`` tagged union (``convilyn_edge.result``).

Four categories per the repo unit-testing skill: logic, boundary, error,
object-state. One assertion per test; AAA structure.
"""

from __future__ import annotations

import pytest

from convilyn_edge.result import Err, Ok, _ResultError

# ── logic ────────────────────────────────────────────────────────────────────


def test_ok_unwrap_returns_value():
    result = Ok(42)

    assert result.unwrap() == 42


def test_err_unwrap_err_returns_error():
    result = Err("bad")

    assert result.unwrap_err() == "bad"


def test_ok_map_transforms_value():
    result = Ok(3).map(lambda v: v * 2)

    assert result.unwrap() == 6


def test_err_map_err_transforms_error():
    result = Err("e").map_err(lambda e: e.upper())

    assert result.unwrap_err() == "E"


def test_ok_map_err_is_noop():
    result = Ok(3).map_err(lambda e: "changed")

    assert result.unwrap() == 3


def test_err_map_is_noop():
    result = Err("e").map(lambda v: v * 2)

    assert result.unwrap_err() == "e"


# ── boundary ─────────────────────────────────────────────────────────────────


def test_ok_unwrap_or_ignores_default():
    result = Ok(1)

    assert result.unwrap_or(999) == 1


def test_err_unwrap_or_returns_default():
    result: Err[str] = Err("e")

    assert result.unwrap_or(999) == 999


def test_ok_holds_falsy_value_without_confusing_variant():
    # A falsy payload (0 / "" / None) must not be mistaken for an Err.
    result = Ok(0)

    assert result.is_ok is True


# ── error ────────────────────────────────────────────────────────────────────


def test_ok_unwrap_err_raises():
    result = Ok(1)

    with pytest.raises(_ResultError):
        result.unwrap_err()


def test_err_unwrap_raises():
    result = Err("boom")

    with pytest.raises(_ResultError):
        result.unwrap()


# ── object-state ─────────────────────────────────────────────────────────────


def test_ok_discriminants():
    result = Ok(1)

    assert (result.is_ok, result.is_err) == (True, False)


def test_err_discriminants():
    result = Err("e")

    assert (result.is_ok, result.is_err) == (False, True)


def test_ok_is_frozen():
    result = Ok(1)

    with pytest.raises(Exception):  # noqa: B017 — FrozenInstanceError
        result.value = 2  # type: ignore[misc]


def test_ok_equality_by_value():
    assert Ok(1) == Ok(1)


def test_ok_not_equal_to_err():
    assert Ok(1) != Err(1)


def test_match_narrows_to_ok_branch():
    # Exhaustive pattern-match is the ergonomic the tagged union exists for.
    matched = None
    match Ok("v"):
        case Ok(value):
            matched = value
        case Err(_):
            matched = "err"

    assert matched == "v"
