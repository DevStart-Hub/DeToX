"""Launch the independent eye-position GUI without DeToX or PsychoPy.

Examples
--------
```powershell
python examples/test_monitor_window.py
python examples/test_monitor_window.py --duration 5 --crash
```

On Windows, reuse the production monitor's Job Object ownership helper.
On other platforms, pipe EOF provides cooperative cleanup only. No settings are
changed and no samples are saved. Close the window or press Ctrl+C to stop early.
"""

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from tobii_process_lifecycle import read_ready

# Import only the small ownership helper, without importing DeToX or PsychoPy.
GUI_DIRECTORY = Path(__file__).resolve().parents[1] / 'DeToX' / 'gui'
sys.path.insert(0, str(GUI_DIRECTORY))
from ProcessLifecycle import WindowsProcessJob


def main():
    """Discover a tracker, launch the GUI, and clean up after this manual test."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--address', help='Tracker URI; otherwise discover exactly one tracker')
    parser.add_argument('--duration', type=float, default=30, help='Display seconds (default 30)')
    parser.add_argument('--update-rate', type=float, default=20, help='Display refresh Hz (default 20)')
    parser.add_argument('--crash', action='store_true', help='Abruptly exit the parent after duration')
    parser.add_argument('--size', type=int, nargs=2, metavar=('WIDTH', 'HEIGHT'),
                        help='Initial window content size in pixels, e.g. --size 500 350')
    parser.add_argument('--fixed-size', action='store_true', help='Disable resizing by dragging edges')
    parser.add_argument('--show-images', action='store_true', help='Show optional tracker camera images')
    parser.add_argument('--reverse-eye', action='store_true', help='Swap camera image panels only')
    args = parser.parse_args()
    if args.size is not None and any(value <= 0 for value in args.size):
        parser.error('size dimensions must be positive')
    for name, value in (('duration', args.duration), ('update-rate', args.update_rate)):
        if not math.isfinite(value) or value <= 0:
            parser.error(f'{name} must be finite and greater than zero')

    import tobii_research as tr
    address = args.address
    if not address:
        trackers = tr.find_all_eyetrackers()
        if len(trackers) != 1:
            parser.error('Connect exactly one tracker or specify --address URI')
        address = trackers[0].address
        del trackers

    job = None
    process = None
    try:
        if sys.platform == 'win32':
            job = WindowsProcessJob()
        else:
            print('No OS ownership on this platform: parent EOF cleanup only.', flush=True)

        script = GUI_DIRECTORY / 'MonitorWindow.py'
        config = {'tracker_address': address, 'update_rate': args.update_rate,
                  'report_ready': True, 'wait_for_start': True,
                  'size': args.size, 'resizable': not args.fixed_size,
                  'show_images': args.show_images, 'reverse_eye': args.reverse_eye}
        process = subprocess.Popen([sys.executable, str(script), json.dumps(config)],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        if job:
            job.assign(process.pid)
        process.stdin.write(b'start\n')
        process.stdin.flush()
        ready = read_ready(process.stdout, 20)
        if not ready.get('ready') or not ready.get('direct'):
            raise RuntimeError(f'Monitor did not establish its subscription: {ready}')
        print(f'Parent PID {os.getpid()}, GUI PID {process.pid}; subscribed to {address}', flush=True)
        if args.show_images:
            print(f'Camera image subscription active: {ready.get("images", False)}', flush=True)
        print('Move in front of the tracker to check the eye dots and distance bar.', flush=True)
        deadline = time.monotonic() + args.duration
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.1)
        if args.crash and process.poll() is None:
            print('Crashing parent now; the GUI should disappear.', flush=True)
            os._exit(23)
    except KeyboardInterrupt:
        print('Stopping monitor.')
    finally:
        try:
            if process is not None:
                process.stdin.close()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=2)
                process.stdout.close()
        finally:
            if job:
                job.close()
    if process.returncode != 0:
        raise RuntimeError(f'Monitor exited with code {process.returncode}')
    print('Monitor stopped cleanly.')


if __name__ == '__main__':
    main()
