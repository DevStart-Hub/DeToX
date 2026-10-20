"""Manual Tobii hardware probe: gaze in this process, position in a fresh child.

Run: python examples/tobii_two_processes.py
Requires a connected tracker. No GUI, DeToX imports, calibration, settings changes,
or data files. PASS checks sample delivery, not timing quality or crash safety.
"""

import argparse
import math
import multiprocessing as mp
import os
import threading
import time


class SampleCounter:
    def __init__(self):
        self._count = 0
        self._lock = threading.Lock()

    def callback(self, sample):
        with self._lock:
            self._count += 1

    def read(self):
        with self._lock:
            return self._count


def position_worker(address, reports):
    """Create the SDK object here; never inherit/pass the parent's tracker."""
    tracker = None
    subscribed = False
    counter = SampleCounter()
    error = None
    try:
        import tobii_research as tr

        tracker = tr.EyeTracker(address)
        tracker.subscribe_to(tr.EYETRACKER_USER_POSITION_GUIDE,
                             counter.callback, as_dictionary=True)
        subscribed = True
        reports.send(('ready', os.getpid()))
        # EOF lets the child exit if the parent disappears, without shared locks.
        if reports.poll(60):
            reports.recv()
        else:
            raise RuntimeError('Parent did not stop the probe within 60 seconds')
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    finally:
        if subscribed:
            try:
                tracker.unsubscribe_from(tr.EYETRACKER_USER_POSITION_GUIDE,
                                         counter.callback)
            except Exception as exc:
                error = f'Position unsubscribe failed: {exc}'
        try:
            reports.send(('error', error) if error else ('samples', counter.read()))
        except (BrokenPipeError, EOFError, OSError):
            pass  # The parent may already have exited.
        reports.close()


def receive(reports, timeout):
    if not reports.poll(timeout):
        raise RuntimeError('Timed out waiting for the position process')
    try:
        kind, value = reports.recv()
    except EOFError as exc:
        raise RuntimeError('Position process exited without a report') from exc
    if kind == 'error':
        raise RuntimeError(f'Position process: {value}')
    return kind, value


def run_probe(address, seconds):
    import tobii_research as tr

    if address:
        tracker = tr.EyeTracker(address)
    else:
        trackers = tr.find_all_eyetrackers()
        if len(trackers) != 1:
            for tracker in trackers:
                print(f'  {tracker.serial_number}: {tracker.address}')
            raise RuntimeError(
                f'Found {len(trackers)} trackers; connect one tracker or select --address URI'
            )
        tracker = trackers[0]

    print(f'SDK {tr.__version__} | {tracker.model} | {tracker.serial_number}', flush=True)
    print(f'Tracker: {tracker.address}', flush=True)
    print(f'Parent PID {os.getpid()}: subscribing to GAZE_DATA', flush=True)
    counter = SampleCounter()
    ctx = mp.get_context('spawn')
    reports, child_reports = ctx.Pipe()
    child = ctx.Process(target=position_worker,
                        args=(tracker.address, child_reports))
    subscribed = False
    started = False
    try:
        tracker.subscribe_to(tr.EYETRACKER_GAZE_DATA,
                             counter.callback, as_dictionary=True)
        subscribed = True
        time.sleep(2)
        baseline = counter.read()
        print(f'Before child: {baseline} gaze samples in 2 seconds', flush=True)
        if not baseline:
            raise RuntimeError('No gaze samples even before starting the child')

        child.start()
        started = True
        child_reports.close()
        kind, pid = receive(reports, 20)
        if kind != 'ready':
            raise RuntimeError(f'Unexpected startup report: {kind}')
        print(f'Child PID {pid}: subscribed to USER_POSITION_GUIDE', flush=True)
        before = counter.read()
        time.sleep(seconds)
        during = counter.read() - before
        reports.send('stop')
        kind, positions = receive(reports, 10)
        if kind != 'samples':
            raise RuntimeError(f'Unexpected final report: {kind}')
        child.join(timeout=5)
        if child.exitcode != 0:
            raise RuntimeError(f'Position process did not exit cleanly: {child.exitcode}')
        print(f'Together ({seconds:g}s): {during} gaze samples; '
              f'{positions} position samples over the child subscription', flush=True)

        before = counter.read()
        time.sleep(2)
        after = counter.read() - before
        print(f'After child exits: {after} gaze samples in 2 seconds', flush=True)
        if not during or not positions or not after:
            raise RuntimeError('One or more phases received no samples')
        print('PASS: both processes received samples and gaze continued after child exit.')
        print('This does not prove gap-free recording or crash/freeze safety.')
    finally:
        # Closing the control pipe also signals EOF to a waiting child.
        reports.close()
        child_reports.close()
        if started:
            child.join(timeout=2)
            if child.is_alive():
                child.terminate()
                child.join(timeout=2)
            if child.is_alive():
                child.kill()
                child.join(timeout=2)
        if subscribed:
            tracker.unsubscribe_from(tr.EYETRACKER_GAZE_DATA, counter.callback)


def duration(value):
    seconds = float(value)
    if not math.isfinite(seconds) or not 0 < seconds <= 30:
        raise argparse.ArgumentTypeError('duration must be greater than 0 and at most 30 seconds')
    return seconds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--address', help='Tracker URI; otherwise exactly one must be discovered')
    parser.add_argument('--duration', type=duration, default=10,
                        help='Seconds with both subscriptions active (default: 10; max: 30)')
    args = parser.parse_args()
    try:
        run_probe(args.address, args.duration)
    except KeyboardInterrupt:
        print('Interrupted.')
        return 130
    except Exception as exc:
        print(f'FAIL: {exc}', flush=True)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
