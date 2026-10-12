# Manual examples

These are interactive demonstrations, **not automated tests**. Run them explicitly
with DeToX and PsychoPy installed, after checking the window/monitor settings.
Recording examples write HDF5 files in the current working directory.

| Example | Requirements / purpose |
| --- | --- |
| `record_simulated.py` | Mouse simulation, periodic save and final stop/save |
| `record_tobii.py` | Connected Tobii tracker, repeated recording saves |
| `gaze_contingent.py` | Mouse simulation and visual gaze feedback; Escape exits |
| `calibration.py` | Connected Tobii tracker and interactive calibration |
| `read_hdf5.R` | R with `hdf5r`; adjust the recording filename before running |

Example, from the repository root:

```powershell
python examples/record_simulated.py
```

Do not import these scripts during automated test discovery: their demonstration
code runs at module level. The former combined `General_Test.py` was retired;
use the separate calibration and recording examples instead.
