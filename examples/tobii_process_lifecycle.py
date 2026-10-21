"""Windows-only parent-failure probe; no GUI or DeToX imports.

Run with Pixi's tracker-test environment. Add --sdk for real Tobii subscriptions.
The supervisor kills only the parent, then independently checks the child handle.
"""

import argparse
import ctypes as ct
from ctypes import wintypes as wt
import json
import os
import subprocess
import sys
import threading
import time


class BasicLimits(ct.Structure):
    _fields_ = [('process_time', ct.c_int64), ('job_time', ct.c_int64),
                ('flags', wt.DWORD), ('min_working_set', ct.c_size_t),
                ('max_working_set', ct.c_size_t), ('active_processes', wt.DWORD),
                ('affinity', ct.c_size_t), ('priority', wt.DWORD),
                ('scheduling', wt.DWORD)]


class IoCounters(ct.Structure):
    _fields_ = [(name, ct.c_uint64) for name in
                ('read_ops', 'write_ops', 'other_ops', 'read_bytes',
                 'write_bytes', 'other_bytes')]


class ExtendedLimits(ct.Structure):
    _fields_ = [('basic', BasicLimits), ('io', IoCounters),
                ('process_memory', ct.c_size_t), ('job_memory', ct.c_size_t),
                ('peak_process_memory', ct.c_size_t), ('peak_job_memory', ct.c_size_t)]


def windows():
    api = ct.WinDLL('kernel32', use_last_error=True)
    signatures = {
        'CreateJobObjectW': ([ct.c_void_p, wt.LPCWSTR], wt.HANDLE),
        'SetInformationJobObject': ([wt.HANDLE, ct.c_int, ct.c_void_p, wt.DWORD], wt.BOOL),
        'AssignProcessToJobObject': ([wt.HANDLE, wt.HANDLE], wt.BOOL),
        'OpenProcess': ([wt.DWORD, wt.BOOL, wt.DWORD], wt.HANDLE),
        'CloseHandle': ([wt.HANDLE], wt.BOOL),
        'WaitForSingleObject': ([wt.HANDLE, wt.DWORD], wt.DWORD),
    }
    for name, (args, result) in signatures.items():
        function = getattr(api, name)
        function.argtypes = args
        function.restype = result
    return api


def checked(result):
    if not result:
        raise ct.WinError(ct.get_last_error())
    return result


def command(role, args):
    cmd = [sys.executable, os.path.abspath(__file__), '--role', role,
           '--scenario', args.scenario]
    if args.sdk:
        cmd += ['--sdk', '--address', args.address]
    return cmd


def child_main(args):
    # Do not connect until the parent has successfully established OS ownership.
    if sys.stdin.readline().strip() != 'start':
        return 1
    tracker = None
    received = threading.Event()

    def on_position(sample):
        received.set()

    try:
        if args.sdk:
            import tobii_research as tr
            tracker = tr.EyeTracker(args.address)
            tracker.subscribe_to(tr.EYETRACKER_USER_POSITION_GUIDE,
                                 on_position, as_dictionary=True)
            if not received.wait(8):
                raise RuntimeError('Position subscriber received no packets')
        print(json.dumps({'ready': True, 'sdk_packet': received.is_set()}), flush=True)
        if args.scenario == 'unresponsive':
            # Deliberately ignore pipe EOF: cooperative shutdown cannot help.
            while True:
                time.sleep(60)
        else:
            sys.stdin.read()
    finally:
        if tracker is not None:
            tracker.unsubscribe_from(tr.EYETRACKER_USER_POSITION_GUIDE, on_position)
    return 0


def parent_main(args):
    api = windows()
    job = checked(api.CreateJobObjectW(None, None))
    child = None
    try:
        limits = ExtendedLimits()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        checked(api.SetInformationJobObject(job, 9, ct.byref(limits), ct.sizeof(limits)))
        child = subprocess.Popen(command('child', args), stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE, text=True)
        # Open explicitly rather than relying on Popen's private handle attribute.
        handle = checked(api.OpenProcess(0x0001 | 0x0100, False, child.pid))
        try:
            checked(api.AssignProcessToJobObject(job, handle))
        finally:
            api.CloseHandle(handle)
        child.stdin.write('start\n')
        child.stdin.flush()
        ready = json.loads(child.stdout.readline())
        print(json.dumps({'child_pid': child.pid, **ready}), flush=True)
        # Supervisor acquires the child handle before permitting the failure.
        if sys.stdin.readline().strip() != 'go':
            return 1
        if args.scenario == 'exception':
            raise RuntimeError('Intentional main-thread failure')
        if args.scenario in ('abrupt', 'unresponsive'):
            os._exit(23)  # No finally, atexit, or cooperative child shutdown.
        time.sleep(60)  # Supervisor forcibly kills this parent, not its tree.
    finally:
        if child is not None:
            child.stdin.close()
            try:
                child.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
        api.CloseHandle(job)  # OS kills only members of our private job.
        if child is not None:
            child.wait(timeout=3)
            child.stdout.close()
    return 0


def read_ready(stream, timeout):
    result = []
    finished = threading.Event()

    def read():
        result.append(stream.readline())
        finished.set()

    threading.Thread(target=read, daemon=True).start()
    if not finished.wait(timeout):
        raise RuntimeError('Parent/child startup timed out')
    return json.loads(result[0])


def run_scenario(args):
    api = windows()
    parent = subprocess.Popen(command('parent', args), stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE, text=True)
    handle = None
    try:
        ready = read_ready(parent.stdout, 20)
        if not ready.get('ready') or (args.sdk and not ready.get('sdk_packet')):
            raise RuntimeError(f'Child not ready: {ready}')
        handle = checked(api.OpenProcess(0x00100000, False, ready['child_pid']))
        parent.stdin.write('go\n')
        parent.stdin.flush()
        started = time.monotonic()
        if args.scenario == 'killed':
            parent.kill()
        parent.wait(timeout=5)
        expected = 1 if args.scenario == 'exception' else 23
        if args.scenario != 'killed' and parent.returncode != expected:
            raise RuntimeError(f'Unexpected parent exit: {parent.returncode}')
        status = api.WaitForSingleObject(handle, 5000)
        if status != 0:
            raise RuntimeError(f'Child survived parent failure (wait status {status})')
        elapsed = time.monotonic() - started
        print(f'PASS {args.scenario}: parent {parent.pid}, child '
              f'{ready["child_pid"]} exited; confirmed within {elapsed:.3f}s', flush=True)
    finally:
        if parent.poll() is None:
            parent.kill()
        parent.wait(timeout=5)
        parent.stdin.close()
        parent.stdout.close()
        if handle:
            api.CloseHandle(handle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sdk', action='store_true', help='Use a real Tobii position subscriber')
    parser.add_argument('--address', help='Tracker URI (otherwise discover exactly one tracker)')
    parser.add_argument('--role', choices=('supervisor', 'parent', 'child'), default='supervisor',
                        help=argparse.SUPPRESS)
    parser.add_argument('--scenario', choices=('exception', 'abrupt', 'killed', 'unresponsive'),
                        help=argparse.SUPPRESS)
    args = parser.parse_args()
    if sys.platform != 'win32':
        parser.error('This probe tests Windows Job Objects and requires Windows')
    if args.role == 'child':
        return child_main(args)
    if args.role == 'parent':
        return parent_main(args)
    if args.sdk and not args.address:
        import tobii_research as tr
        trackers = tr.find_all_eyetrackers()
        if len(trackers) != 1:
            parser.error('Connect exactly one tracker or specify --address URI')
        args.address = trackers[0].address
        print(f'Tracker: {trackers[0].model} | SDK {tr.__version__}', flush=True)
        del trackers
    for scenario in ('exception', 'abrupt', 'killed', 'unresponsive'):
        args.scenario = scenario
        run_scenario(args)
    if args.sdk:
        # Fresh process, fresh SDK connection, after all forced failures.
        probe = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'tobii_two_processes.py')
        subprocess.run([sys.executable, probe, '--address', args.address,
                        '--duration', '2'], check=True, timeout=45)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
