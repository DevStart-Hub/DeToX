"""Experimental GUI and live-monitor helpers for DeToX (internal).

These helpers and their platform behavior may change; validate the overlay and
live monitor on the intended tracker/display setup before use.

This folder holds the on-screen visual feedback used during an experiment:

- ``TrackBox.py``          shared drawing base (track box + distance bar)
- ``PositioningOverlay.py`` setup overlay shown before recording
- ``LiveMonitor.py``       client that drives the detached monitor
- ``MonitorWindow.py``     the window shown by that detached process

Nothing here is part of the public API; these pieces are used internally by
``ETracker.show_status(overlay=True)`` and ``ETracker.enable_live_monitor()``.
"""
