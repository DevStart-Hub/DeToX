"""Deterministic, already-adapted batches for storage tests."""

from pathlib import Path
import runpy

import numpy as np
import pandas as pd


SETTINGS = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'DeToX' / 'ETSettings.py'))


def fake_batch(raw_format, start=0, count=4, label=None):
    """Create adapted rows with 10ms sample spacing and an optional event.

    Raw timestamps use microseconds; simplified timestamps use milliseconds.
    Callback, timestamp-conversion, and event-matching tests use SDK-shaped
    samples instead of these preformatted batches.
    """
    columns = SETTINGS['RawDataColumns' if raw_format else 'SimplifiedDataColumns']
    gaze_dtype = columns.get_hdf5_dtype()
    events_dtype = SETTINGS['EventDataColumns'].get_hdf5_dtype(raw_format)
    sample = np.arange(start, start + count)
    timestamps = sample * (10_000 if raw_format else 10)
    values = {}

    for name in gaze_dtype.names:
        if name == 'Events':
            values[name] = [''] * count
        elif name == 'device_time_stamp':
            values[name] = sample * 10_000
        elif name in ('system_time_stamp', 'TimeStamp'):
            values[name] = timestamps
        elif 'validity' in name.lower():
            values[name] = np.ones(count, dtype=np.int8)
        else:
            values[name] = 0.25 + sample.astype(float) / 100

    if count:
        float_column = next(name for name in gaze_dtype.names if gaze_dtype[name].kind == 'f')
        values[float_column][0] = np.nan

    gaze = pd.DataFrame(values)
    events = None
    if label is not None:
        if not count:
            raise ValueError('Fake event needs a sample to attach to')
        gaze.loc[count - 1, 'Events'] = label
        timestamp_column = events_dtype.names[0]
        events = pd.DataFrame([
            {timestamp_column: int(timestamps[-1]), 'Events': label},
        ])

    return gaze, events
