#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""carla_port/carla_api.py — the single import site for the CARLA module.

Every other module does `from carla_port.carla_api import carla`. That keeps
the real/fake decision in one place and makes the offline validation suite
possible on machines with no CARLA server (see fake_carla/).

The fake is NEVER selected silently. It is used only when the caller asks for
it explicitly, either with CARLA_PORT_FAKE=1 in the environment or by calling
use_fake() before the first `carla` access.
"""
from __future__ import annotations

import os
import sys

_carla = None
_is_fake = False


def _load_real():
    try:
        import carla as _real                      # noqa: F401
        return _real
    except ImportError as exc:                     # pragma: no cover
        raise ImportError(
            "CARLA 0.9.16 python API not importable. Install the wheel/egg for "
            "your interpreter, or run against the offline test double with "
            "CARLA_PORT_FAKE=1 (validation only; it does not simulate CARLA)."
        ) from exc


def use_fake() -> None:
    """Bind the CARLA name to the offline test double. Must be called before
    the first attribute access on `carla`."""
    global _carla, _is_fake
    from . import fake_carla
    _carla = fake_carla
    _is_fake = True
    sys.modules.setdefault("carla", fake_carla)


def is_fake() -> bool:
    return _is_fake


class _Lazy:
    """Defers the CARLA import until first use, so `import carla_port.*` works
    on a machine without the CARLA python API (e.g. to read the code, or to run
    the fake-backed validation)."""

    def __getattr__(self, name):
        global _carla
        if _carla is None:
            if os.environ.get("CARLA_PORT_FAKE") == "1":
                use_fake()
            else:
                _carla = _load_real()
        return getattr(_carla, name)


carla = _Lazy()

__all__ = ["carla", "use_fake", "is_fake"]
