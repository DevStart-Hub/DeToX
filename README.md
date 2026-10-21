# DeToX

DeToX (Developmental Tobii Experiment) is a user-friendly Python wrapper for the tobii-researcher library. It is designed to simplify the integration of Tobii eye-tracking hardware with PsychoPy, particularly for conducting developmental studies involving infants. By streamlining the process of data collection, DeToX aims to enhance the efficiency of developmental eye-tracking experiments.

## Quick Install

``` bash
pip install dvst-detox
```

or for the latest version:

``` bash
pip install git+https://github.com/DevStart-Hub/DeToX.git
```

<sub>Coming soon to PyPI for even easier installation!</sub>

## Why We Built DeToX

While the official Tobii SDK provides powerful features, we found it could be complex for routine research tasks. DeToX bridges this gap by offering a straightforward approach to eye-tracking data collection.

**We don't aim to be the most feature-rich package** - we built exactly what we needed for our infant-friendly studies: a simple, well-documented tool that just works

> This project didn't start from scratch—it builds upon an existing repository that we have used in the past: [**psychopy_tobii_infant**](https://github.com/yh-luo/psychopy_tobii_infant)

While the eye-tracking landscape offers many tools—from [PsychoPy](https://psychopy.org/hardware/eyeTracking.html)'s built-in Tobii integration to comprehensive packages like [Titta](https://github.com/marcus-nystrom/Titta)—DeToX carves out its own niche through thoughtful simplicity. We've prioritized clarity and ease-of-use without sacrificing the flexibility researchers need. When your codebase is straightforward and well-documented, it becomes a platform for innovation rather than an obstacle to overcome.

## Key Features

-   **Simple Data Recording**: Start and stop eye-tracking recordings with just one line of code. Collect gaze data, pupil measurements, and timestamped events automatically during your experiments.

-   **HDF5 Data Storage**: Save data in analysis-ready HDF5 format with events embedded directly in the gaze timeline for easy analysis, while preserving raw event data in a separate table for advanced processing. Includes comprehensive metadata like screen dimensions, framerate, and recording settings.

-   **Infant-Friendly Calibration**: Engage infants with animated calibration stimuli featuring zoom and trill effects. Use colorful, attention-getting images with optional sound feedback for better participant engagement.

-   **Calibration Management**: Save and load calibration files to reuse participant calibrations across multiple sessions.

-   **Simulation Mode**: Test your experiments using mouse input as a gaze proxy when hardware isn't available. Perfect for development, debugging, and pilot studies without requiring physical eye trackers.

-   **Gaze-Contingent Support**: Real-time gaze processing for interactive paradigms with configurable smoothing windows. Extract gaze positions using median, mean, or last-sample methods for different latency/accuracy trade-offs.

## Learn More

We believe good software requires great documentation. Explore our comprehensive guides and examples:

📚 [**DeToX Documentation**](https://devstart-hub.github.io/DeToX/) **(coming soon)**

------------------------------------------------------------------------

**Questions or found a bug?** Let us know by creating an issue or starting a discussion here on GitHub!

## Development environments

With Pixi installed, the checked-in `pixi.lock` provides separate environments:

- `pixi run python ...`: DeToX/PsychoPy runtime, using Python 3.10.
- `pixi run -e tracker-test python ...`: SDK-only environment with Tobii Research
  2.1.0 and Pillow; no DeToX or PsychoPy installation.
- `pixi run -e docs docs-build`: isolated documentation build on Python 3.12.

The runtime and SDK-only environments target Windows/Linux 64-bit. The Tobii
2.1.0 Python wheels require Python 3.10; documentation uses static introspection
and does not import the runtime. `pixi install --locked` installs the default
environment without updating the lockfile. Environment files stay in `.pixi/`,
which Git ignores. These setup changes do not change acquisition or saved data.
