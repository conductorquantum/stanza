"""Bokeh backend type alias."""

from __future__ import annotations

from typing import Union

from stanza.plotter.backends.inline import InlineBackend
from stanza.plotter.backends.server import ServerBackend

BokehBackend = Union["ServerBackend", "InlineBackend"]
