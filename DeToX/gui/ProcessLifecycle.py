"""Windows ownership for the optional monitor, without owning the Tobii runtime.

The parent holds a private, non-inheritable Job Object handle. Windows kills
its assigned GUI process when that handle closes, including on parent death.
Other platforms use the launcher's cooperative stdin-EOF shutdown instead.
"""

import ctypes
from ctypes import wintypes


class _BasicLimits(ctypes.Structure):
    _fields_ = [('process_time', ctypes.c_int64), ('job_time', ctypes.c_int64),
                ('flags', wintypes.DWORD), ('minimum_memory', ctypes.c_size_t),
                ('maximum_memory', ctypes.c_size_t), ('active_processes', wintypes.DWORD),
                ('affinity', ctypes.c_size_t), ('priority', wintypes.DWORD),
                ('scheduling', wintypes.DWORD)]


class _IoCounters(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in
                ('read_ops', 'write_ops', 'other_ops', 'read_bytes',
                 'write_bytes', 'other_bytes')]


class _ExtendedLimits(ctypes.Structure):
    _fields_ = [('basic', _BasicLimits), ('io', _IoCounters),
                ('process_memory', ctypes.c_size_t), ('job_memory', ctypes.c_size_t),
                ('peak_process_memory', ctypes.c_size_t), ('peak_job_memory', ctypes.c_size_t)]


def _check(result):
    """Raise the Windows error immediately when an API operation fails."""
    if not result:
        raise ctypes.WinError(ctypes.get_last_error())
    return result


class WindowsProcessJob:
    """
    Own a monitor process through a Windows kill-on-close Job Object.

    Notes
    -----
    Construct only on Windows. Assign the child before releasing its startup
    gate. Keep this object in the parent only and call close during cleanup.
    The tracker runtime is not assigned to this job and must not be terminated.
    """

    def __init__(self):
        self.api = ctypes.WinDLL('kernel32', use_last_error=True)
        signatures = {
            'CreateJobObjectW': ([ctypes.c_void_p, wintypes.LPCWSTR], wintypes.HANDLE),
            'SetInformationJobObject': ([wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p,
                                         wintypes.DWORD], wintypes.BOOL),
            'AssignProcessToJobObject': ([wintypes.HANDLE, wintypes.HANDLE], wintypes.BOOL),
            'OpenProcess': ([wintypes.DWORD, wintypes.BOOL, wintypes.DWORD], wintypes.HANDLE),
            'CloseHandle': ([wintypes.HANDLE], wintypes.BOOL),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(self.api, name)
            function.argtypes = arguments
            function.restype = result
        self.handle = _check(self.api.CreateJobObjectW(None, None))
        try:
            limits = _ExtendedLimits()
            limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            _check(self.api.SetInformationJobObject(
                self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)))
        except Exception:
            self.close()
            raise

    def assign(self, process_id):
        """
        Assign the GUI child before it can create a window or connect to Tobii.

        Parameters
        ----------
        process_id : int
            Operating-system process ID of the newly launched child.
        """
        process = _check(self.api.OpenProcess(0x0001 | 0x0100, False, process_id))
        try:
            _check(self.api.AssignProcessToJobObject(self.handle, process))
        finally:
            self.api.CloseHandle(process)

    def close(self):
        """Release ownership, terminating a surviving child. Safe to repeat."""
        if self.handle is not None:
            _check(self.api.CloseHandle(self.handle))
            self.handle = None
