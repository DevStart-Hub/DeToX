# Automated tests

Use Python 3.10 with `numpy`, `pandas`, and `tables` installed. From the repository
root, run:

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

| File | Coverage |
| --- | --- |
| `test_hdf5_settings.py` | Column order/types, schema generation, empty tables, DataFrame reads |
| `test_hdf5_recording.py` | Actual recording/storage methods, repeated saves, event matching, raw/simplified timestamps, final stop/save, CSV compatibility |
| `hdf5_fixtures.py` | Shared fake-data helper; not a test suite |

The suite contains 12 tests and uses temporary files that are removed afterward.
It does not open windows or require a tracker. Settings are loaded directly;
recording tests compile selected methods unchanged from `DeToX/Base.py` to avoid
importing PsychoPy/Tobii. External clocks, streaming, and status display are
controlled by the tests. This does not validate the complete package import,
physical hardware, real-time callback concurrency, or failed-save recovery.

Raw timestamps preserve SDK microseconds. Simplified timestamps use relative
milliseconds. Event labels read through `pandas.read_hdf()` are Python strings.
The existing fixed-width event-text limit is not a truncation-prevention feature.

Manual hardware/window demonstrations live in [`../examples/`](../examples/),
not in the automated suite. Historical prototype writers and generated demo
files are not part of the maintained tests.
