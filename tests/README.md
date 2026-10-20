# Automated tests

Use Python 3.10 with `numpy`, `pandas`, and `tables` installed. From the repository
root, run:

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

| File | Coverage |
| --- | --- |
| `test_hdf5_settings.py` | Column order/types, schema generation, empty tables, DataFrame reads |
| `test_hdf5_recording.py` | Actual recording/storage methods, repeated saves, event matching, format/coordinate policy, raw/simplified timestamps, bounded synchronization, final stop/save, CSV compatibility |
| `hdf5_fixtures.py` | Shared fake-data helper; not a test suite |

The HDF5 suite uses temporary files that are removed afterward.
The integrated console-formatting suite adds 11 tests in `test_nice_print.py`.
It does not open windows or require a tracker. Settings are loaded directly;
recording tests compile selected methods unchanged from `DeToX/Base.py` and the
pixel-coordinate converter from `DeToX/Coords.py` to avoid importing PsychoPy/Tobii.
External clocks, streaming, and status display are
controlled by the tests. This does not validate the complete package import,
physical hardware, real-time callback concurrency, or failed-save recovery.

Raw timestamps preserve SDK microseconds. Simplified timestamps use relative
milliseconds. Event labels read through `pandas.read_hdf()` are Python strings.
Raw coordinates remain in Tobii ADCS; simplified coordinates use the selected
PsychoPy unit (pixels by default), not Tobii coordinates. Grouped event labels use
` | ` separators. The `relative_timestamps` HDF5 attribute describes the selected
format; it is not an accepted `start_recording()` argument.
The existing fixed-width event-text limit is not a truncation-prevention feature.

Manual hardware/window demonstrations live in [`../examples/`](../examples/),
not in the automated suite. Historical prototype writers and generated demo
files are not part of the maintained tests.


Raw recordings select optional track-box columns by field presence in the first
saved batch, not by validity or NaN. Missing later values remain NaN; the schema
stays fixed until the next recording. The raw CSV header/HDF5 gaze table is
created at the first save. An empty raw recording has no gaze header/table yet.
Simplified output is unchanged. Tests cover CSV/HDF5, missing fields within and
between batches, all-NaN fields, late fields and schema reset between recordings.
