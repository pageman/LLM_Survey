"""Public package wrapper for the repository's installable Python surface."""

from __future__ import annotations

import importlib
import sys

from src import core as core
from src import modules as modules
from src.core import *  # noqa: F401,F403

__version__ = "0.3.0"

sys.modules[__name__ + ".core"] = core
sys.modules[__name__ + ".modules"] = modules

__all__ = ["__version__", "core", "modules", *getattr(importlib.import_module("src.core"), "__all__", [])]
