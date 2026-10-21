# Manual examples

These are interactive demonstrations, **not automated tests**. Run them explicitly
with DeToX and PsychoPy installed, after checking the window/monitor settings
(except the standalone Tobii process probe below, which needs only the SDK).
Recording examples write HDF5 files in the current working directory.

| Example | Requirements / purpose |
| --- | --- |
| `record_simulated.py` | Mouse simulation, periodic save and final stop/save |
| `record_tobii.py` | Connected Tobii tracker, repeated recording saves |
| `gaze_contingent.py` | Mouse simulation and visual gaze feedback; Escape exits |
| `calibration.py` | Connected Tobii tracker and interactive calibration |
| `read_hdf5.R` | R with `hdf5r`; adjust the recording filename before running |
| `tobii_two_processes.py` | Standalone SDK-only probe: gaze in parent, user position in a spawned child |
| `tobii_process_lifecycle.py` | Windows-only parent-failure/child-cleanup probe; optional live Tobii subscriber |
| `test_monitor_window.py` | Standalone real eye-position GUI; no DeToX/PsychoPy imports |
| `live_monitor.py` | Integrated ETracker monitor without recording; requires full runtime and fullscreen PsychoPy |

Example, from the repository root:

```powershell
python examples/record_simulated.py
```

## Two-process Tobii connection probe (no GUI)

Connect and power on one tracker, then run from the repository root:

```powershell
python examples/tobii_two_processes.py
```

Use Python 3.10 with `tobii-research==2.1.0` and Pillow installed:

```powershell
python -m pip install tobii-research==2.1.0 pillow
```

The standalone probes do not need DeToX or PsychoPy.

The parent receives gaze, then a fresh child connects to the **same tracker URI**
and receives user-position-guide samples for 10 seconds. It prints both process
IDs and sample counts, closes the child, and checks gaze delivery for another
2 seconds. Exit code 0 / `PASS` means both streams delivered samples and gaze
continued after the child disconnected. It does **not** prove gap-free recording,
continuous overlap, or crash/freeze safety. No samples are saved, no calibration
is performed, and no device settings are changed. Valid eyes are not required;
this checks packet delivery, not tracking validity.

Use `--duration 20` to change the overlap interval (maximum 30 seconds).
If multiple trackers are discovered, the script lists their URIs; select one
with `--address "URI"`. No hardware found or any failed phase returns exit code 1.

## Parent-failure cleanup probe (Windows)

```powershell
# Test OS ownership without hardware:
python examples/tobii_process_lifecycle.py
# Repeat with a real position subscriber, then check fresh gaze/position connections:
python examples/tobii_process_lifecycle.py --sdk
```

A supervisor launches a disposable parent, which owns its child through a private
Windows Job Object with `KILL_ON_JOB_CLOSE`. The child cannot initialize the SDK
until assignment to the job succeeds. Only the parent owns the job handle.
The supervisor acquires a child process handle before triggering each failure,
then checks that handle for termination (5-second timeout):

- Unhandled main-thread exception, allowing `finally` cleanup.
- Abrupt `os._exit()`, bypassing Python cleanup.
- External termination of **only the parent**, not its process tree.
- Abrupt parent exit while the child deliberately ignores shutdown/pipe EOF.

The exception case prints an **intentional traceback**. Each successful case
prints `PASS`; any failure returns a nonzero exit code. `--sdk` requires receipt
of a real position packet before triggering the failure. It ends by launching
the two-process connection probe in a fresh process to check that the tracker
still works without a restart. `--address "URI"` selects a particular tracker.

The unresponsive case ignores cooperative shutdown; it is not an OS-suspended
process. These tests validate the standalone Windows ownership mechanism, not
DeToX integration, parent-hang detection, or gaze continuity during a child crash.
No GUI, calibration, device setting changes, or data files are involved.

## Independent eye-position GUI

```powershell
python examples/test_monitor_window.py
```

The existing `DeToX/gui/MonitorWindow.py` now supports its own Tobii
user-position subscription. This example launches that script directly; it does
not import the DeToX package, create an ETracker, record gaze, or change tracker
settings. Tkinter must be available (it is included in the Windows Pixi Python).

Move your head to check the eye dots and relative distance bar. The display uses
only the latest sample and hides feedback after one second without packets.
It refreshes at 20 Hz; this is feedback, not a physical distance measurement or
saved research data. Close the window, press Ctrl+C, or wait 30 seconds to stop.

- `--duration 10`: stop after ten seconds.
- `--update-rate 10`: refresh the display ten times per second.
- `--show-images`: add live camera panels (real hardware only). Images are
  decoded in the GUI process, never saved, and cleared when stale. Camera IDs
  are not anatomical left/right eye labels. Full/cropped frames depend on the
  tracker and tracking state; each panel displays its camera's latest frame.
  By default the left panel uses camera 1 (preferring region 1); the right uses
  camera 0 (preferring region 0). If that region is absent, another fresh region from the
  same camera is used. Camera/region metadata is not shown. Missing cameras
  stay blank rather than duplicate the other feed. These are not verified
  anatomical left/right eye assignments.
  Unsupported streams or unavailable Pillow leave position feedback working.
  If images are requested, the child checks the tracker's eye-image capability.
  Missing support or a failed image subscription hides the camera panels and
  prints one warning to stderr. With images disabled, no image check or warning
  is performed. No recording settings or gaze callbacks are changed.
- `--reverse-eye`: swap the two camera panel positions when `--show-images` is
  enabled. Does not mirror the images or change position dots or gaze data.
- `--size 500 350`: initial window content width/height in pixels. The title bar
  is excluded. Smaller sizes are clamped to the natural layout minimum.
- Drag the window edges to resize the position display and distance bar.
  Fonts, dot radius, and padding remain controlled by `scale` rather than resizing.
  Use `--fixed-size` to disable edge dragging. The positioning overlay remains
  non-resizable by default.
- `--address "URI"`: choose a tracker explicitly.
- `--duration 5 --crash`: abruptly exit the launcher after five seconds. Exit
  code 23 is intentional; on Windows the owned GUI should disappear immediately.

The launcher reuses the production Windows ownership helper in
`DeToX/gui/ProcessLifecycle.py`. Other platforms have cooperative pipe-EOF cleanup
only, not the Windows hard-kill guarantee. The startup pipe gate ensures that
the launcher assigns ownership before Tk or the SDK connection is created.

ETracker now uses this independent connection through `LiveMonitor.py` on real
hardware. The positioning overlay keeps its existing subscriptions and fixed-size
default, and shares the updated layout. Omitting `tracker_address` selects binary
stdin packets for mouse simulation.

## Integrated ETracker monitor

Use an environment with DeToX and PsychoPy installed:

```powershell
python examples/live_monitor.py --screen 1 --size 500 350
```

This opens a fullscreen PsychoPy window on screen 0 and independent feedback on
the selected screen. It does not start a recording or write data. Escape or 30
seconds ends the test. Choose screen 0 on a single-display setup.

In an existing experiment:

```python
controller.enable_live_monitor(screen=1, size=(500, 350), update_rate=20)
controller.start_recording('trial.h5')
# Run the experiment, then stop/save while keeping feedback open:
controller.stop_recording(stop_monitor=False)
controller.stop_live_monitor()  # Explicit, safe to repeat; does not stop recording.
```

Enable works before/during recording; repeating it does not create duplicates.
By default, `stop_recording()` also requests monitor shutdown. Use
`stop_recording(stop_monitor=False)` to keep feedback between recordings.
Controller shutdown always closes a remaining monitor.
Use `enable_live_monitor(show_images=True, size=(700, 350))` for camera feedback;
add `reverse_eye=True` to swap the two image panels. The default is False.
The default remains position-only. With a running monitor, stop and enable it
again to change options. The standalone preview is:

```powershell
python examples/test_monitor_window.py --show-images --size 700 350
```
Explicitly enable again after closing/crashing the child to reopen it. Real gaze
callbacks no longer forward display packets. Simulation retains latest-only
packets, written by a background thread rather than acquisition code.

Monitor startup is asynchronous. Child SDK/GUI errors appear on stderr and do
not raise into recording. Stopping the monitor schedules bounded cleanup off the
calling thread. On Windows a private Job Object kills the child when the parent
dies; other platforms currently have cooperative EOF cleanup only.

Hardware-free checks for both input modes, invalid/stale feedback, and cleanup:

```powershell
python -m unittest discover -s tests -p test_monitor_window.py -v
```

Do not import these scripts during automated test discovery: their demonstration
code runs at module level. The former combined `General_Test.py` was retired;
use the separate calibration and recording examples instead.
