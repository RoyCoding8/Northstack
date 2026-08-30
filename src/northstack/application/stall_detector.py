"""Stall detector.

A run that is *alive but not progressing* inside the configured window would
stay pinned forever: a cell that hangs emits no terminal event, so the wave
loop never advances and the run never reaches an outcome. The stall detector
turns that into an honest terminal: when the elapsed time since the last
per-cell heartbeat exceeds the configured window, the run abstains with a
typed ``StallDetected`` event (a stuck run is an unknown-outcome run, so
abstention -- not a guessed failure -- is the honest terminal).

The detector is a clock-injectable unit so tests advance time deterministically
rather than sleeping. ``heartbeat()`` records a progress beat (driven by the
per-cell loop); ``is_stalled()`` compares "now" against the last beat. A
``window_seconds`` of 0 means no configured cap: the detector never trips,
mirroring the ``BudgetAuthority`` "None == unlimited" semantics -- an operator
who sets no stall window has opted out of stall detection.
"""

from __future__ import annotations

from collections.abc import Callable
from time import monotonic


class StallDetector:
    """Owns the "alive but not progressing" signal for one run.

    Constructed once per run with the configured window and a clock callable
    (defaults to ``time.monotonic``). ``heartbeat()`` is the progress seam the
    per-cell loop drives; ``is_stalled()`` is True only when the window is
    configured (>0) and the elapsed time since the last beat exceeds it.
    """

    __slots__ = ("_clock", "_last_beat", "_window")

    def __init__(
        self,
        *,
        window_seconds: float,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._window = window_seconds
        self._clock = clock or monotonic
        self._last_beat = self._clock()

    def heartbeat(self) -> None:
        """Record a per-cell progress beat (the stall detector's input)."""
        self._last_beat = self._clock()

    def is_stalled(self) -> bool:
        """True when the window is configured and no beat has arrived in it."""
        if self._window <= 0:
            return False
        return (self._clock() - self._last_beat) > self._window
