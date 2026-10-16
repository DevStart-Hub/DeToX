import os
import time
import tables
import atexit
import warnings
import threading
from pathlib import Path
from datetime import datetime
from collections import deque

# Third party imports
import numpy as np
import pandas as pd
import tobii_research as tr
from psychopy import core, event, visual

# Local imports
from . import Coords
# Remove the old import and import both calibration classes
from .Calibration import TobiiCalibrationSession, MouseCalibrationSession
from . import ETSettings as cfg
from .Utils import NicePrint


class ETracker:
    """
    A high-level controller for running eye-tracking experiments with Tobii Pro and PsychoPy.

    The **ETracker** class is a simplified Python interface designed to streamline the process of running infant eye-tracking experiments. It acts as a bridge between the **Tobii Pro SDK** (version 3.0 or later) and the popular experiment-building framework, **PsychoPy**.

    This class is the central hub for your eye-tracking experiment. Instead of managing low-level SDK functions, the TobiiController provides a clean, unified workflow for key experimental tasks. It is designed to "detoxify" the process, abstracting away complex boilerplate code so you can focus on your research.

    Key features include:
     - **Experiment Control**: Start, stop, and manage eye-tracking recordings with simple method calls.
     - **Data Management**: Automatically save recorded gaze data to a specified file format.
     - **Calibration**: Easily run a calibration procedure or load an existing calibration file to prepare the eye-tracker.
     - **Seamless Integration**: Built specifically to integrate with PsychoPy's experimental loop, making it a natural fit for your existing research designs.

    This class is intended to be the first object you instantiate in your experiment script. It provides a minimal yet powerful set of methods that are essential for conducting a reliable and reproducible eye-tracking study.
    """

    # --- Core Methods ---

    def __init__(self, win, etracker_id=0, simulate=False, verbose=True):
        """
        Initializes the ETracker controller.

        This constructor sets up the ETracker, either by connecting to a physical
        Tobii eye tracker or by preparing for simulation mode. It initializes
        all necessary attributes for data collection, state management, and
        interaction with the hardware or simulated input.

        Parameters
        ----------
        win : psychopy.visual.Window
            The PsychoPy window object where stimuli will be displayed. This is
            required for coordinate conversions. Real tracking requires fullscreen
            on the display configured for the tracker; keep it fullscreen for the
            lifetime of this controller. Simulation also supports windowed mode.
        id : int, optional
            The index of the Tobii eye tracker to use if multiple are found.
            Default is 0. Ignored if `simulate` is True.
        simulate : bool, optional
            If True, the controller will run in simulation mode, using the mouse
            as a proxy for gaze data. If False (default), it will attempt to
            connect to a physical Tobii eye tracker.
        verbose : bool, optional
            If True (default), prints connection information to the console.

        Raises
        ------
        ValueError
            If real tracking is requested with a non-fullscreen window.
        RuntimeError
            If `simulate` is False and no Tobii eye trackers can be found.
        """
        # --- Core Attributes ---
        # Store essential configuration parameters provided at initialization.
        self.win = win
        self.simulate = simulate

        if not self.simulate and not self.win.fullscr:
            raise ValueError(
                "Real eye tracking currently requires a fullscreen window "
                "on the display configured for the tracker. "
                "Windowed mode is supported for simulation only."
            )

        self.eyetracker_id = etracker_id
        self.verbose = verbose

        # --- State Management ---
        # Flags and variables to track the current state of the controller.
        self.recording = False          # True when data is being collected.
        self.first_timestamp = None     # Stores the timestamp of the first gaze sample for relative timing.

        # --- Data Buffers ---
        # Use deques for efficient appending and popping from both ends.
        self._buf_lock = threading.Lock()  # Lock for thread-safe access to buffers.
        self.gaze_data = deque()        # Main buffer for incoming gaze data.
        self._position_data = deque(maxlen=1)  # Latest positioning sample only.
        self.event_data = deque()       # Buffer for timestamped experimental events.
        self.gaze_contingent_buffer = None # Buffer for real-time gaze-contingent logic.

        # --- Timing ---
        # Clocks for managing experiment timing.
        self.experiment_clock = core.Clock()

        # --- Hardware and Simulation Attributes ---
        # Initialize attributes for both real and simulated modes.
        self.eyetracker = None          # Tobii eyetracker object.
        self.calibration = None         # Tobii calibration object.
        self.mouse = None               # PsychoPy mouse object for simulation.
        self.fps = None                 # Frames per second (frequency) of the tracker.
        self.illum_mode = None          # Illumination mode of the tracker.
        self._stop_simulation = None    # Threading event to stop simulation loops.
        self._simulation_thread = None  # Thread object for running simulations.
        self.live_monitor = None        # Live eye position monitor window.

        # --- Setup based on Mode (Real vs. Simulation) ---
        # Configure the controller for either a real eyetracker or simulation.
        if self.simulate:
            # In simulation mode, use the mouse as the input device.
            self.mouse = event.Mouse(win=self.win)
        else:
            # In real mode, find and connect to a Tobii eyetracker.
            eyetrackers = tr.find_all_eyetrackers()
            if not eyetrackers:
                raise RuntimeError(
                    "No Tobii eyetrackers detected.\n"
                    "Verify the connection and make sure to power on the "
                    "eyetracker before starting your computer."
                )
            # Select the specified eyetracker and prepare the calibration API.
            self.eyetracker = eyetrackers[self.eyetracker_id]
            self.calibration = tr.ScreenBasedCalibration(self.eyetracker)

            # Check Tobii SDK version for feature compatibility
            self.sdk_version = tr.__version__
            self.sdk_major = int(self.sdk_version.split('.')[0])

        # --- Finalization ---
        # Display connection info and register the cleanup function to run on exit.
        self._get_info(moment='connection')
        atexit.register(self._close)

        
    def set_eyetracking_settings(self, desired_fps=None, desired_illumination_mode=None, use_gui=False, screen=-1, alwaysOnTop=True):
        """
        Configure and apply Tobii eye tracker settings.
        
        This method updates the eye tracker's sampling frequency (FPS) and illumination 
        mode, either programmatically or via a graphical interface. It ensures that 
        configuration changes are only made when the device is idle and connected.
        
        After applying settings, a summary is displayed showing which settings changed
        and which remained the same.
        
        Parameters
        ----------
        desired_fps : int, optional
            Desired sampling frequency in Hz (e.g., 60, 120, 300). If None, the current 
            frequency is retained. When `use_gui=True`, this value pre-populates the 
            dialog box dropdown.
        desired_illumination_mode : str, optional
            Desired illumination mode (e.g., 'Auto', 'Bright', 'Dark'). If None, the 
            current illumination mode is retained. When `use_gui=True`, this value 
            pre-populates the dialog box dropdown.
        use_gui : bool, optional
            If True, opens a PsychoPy GUI dialog with dropdown menus that allows users 
            to select settings interactively. The dropdowns are pre-populated with values 
            from `desired_fps` and `desired_illumination_mode` if provided, or current 
            settings if None. Defaults to False.
        screen : int, optional
            Screen number where the GUI dialog is displayed. Only used when `use_gui=True`.
            If -1 (default), the dialog appears on the primary screen. Use 0, 1, 2, etc. 
            to specify other monitors. Ignored when `use_gui=False`.
        alwaysOnTop : bool, optional
            Whether the GUI dialog stays on top of other windows. Only used when 
            `use_gui=True`. Default is True to prevent the dialog from being hidden 
            behind experiment windows. Ignored when `use_gui=False`.
        
        Raises
        ------
        RuntimeError
            If no physical eye tracker is connected or if the function is called in 
            simulation mode.
        ValueError
            If the specified FPS or illumination mode is not supported by the connected 
            device.
        
        Details
        -------
        - Settings cannot be changed during active recording. If an ongoing recording 
        is detected, a non-blocking warning is issued and the function exits safely.
        - When `use_gui=True`, a PsychoPy dialog window appears with dropdown menus.
        The `screen` and `alwaysOnTop` parameters control its display behavior.
        - After successfully applying new settings, the internal attributes `self.fps` 
        and `self.illum_mode` are updated to reflect the current device configuration.
        - A summary of applied changes is displayed using NicePrint, showing which 
        settings changed (with old --> new values) and which remained unchanged.
        
        Examples
        --------
        ### Programmatic Settings
        
        #### Set frequency to 120 Hz
        ```python
                ET_controller.set_eyetracking_settings(desired_fps=120)
        ```
        
        #### Set illumination mode to 'Bright'
        ```python
            ET_controller.set_eyetracking_settings(desired_illumination_mode='Bright')
        ```
        
        #### Set both frequency and illumination mode
        ```python
            ET_controller.set_eyetracking_settings(
                desired_fps=120,
                desired_illumination_mode='Bright'
            )
        ```
        
        ### GUI-Based Settings
        
        #### Open GUI with default settings
        ```python
            ET_controller.set_eyetracking_settings(use_gui=True)
        ```
        
        #### GUI on secondary monitor without always-on-top
        ```python
            ET_controller.set_eyetracking_settings(
                use_gui=True,
                screen=1,
                alwaysOnTop=False
            )
        ```
        
        #### GUI with pre-selected 120 Hz in dropdown
        ```python
            ET_controller.set_eyetracking_settings(
                desired_fps=120,
                use_gui=True
            )
        ```
        """
        # --- Pre-condition Check ---

        # Ensure we are not recording already, as settings cannot be changed mid-recording.
        # Raise a non-blocking warning instead of an error.
        if self.recording:
            warnings.warn(
                "|-- Ongoing recording!! --|\n"
                "Eye-tracking settings cannot be changed while recording is active.\n"
                "Skipping set_eyetracking_settings() call.",
                UserWarning
            )
            return

        # Ensure not in simulation mode, as settings require a physical tracker.
        if self.simulate:
            raise RuntimeError(
                "Cannot set eye-tracking settings in simulation mode. "
                "This operation requires a physical Tobii eye tracker."
            )
        
        # Ensure an eyetracker is connected before applying settings.
        if self.eyetracker is None:
            raise RuntimeError(
                "No eyetracker connected. Cannot set eye-tracking settings."
            )
        
        # --- Store old settings for summary ---
        old_fps = self.fps
        old_illum_mode = self.illum_mode
        
        # --- Apply Settings ---
        if use_gui:
            from psychopy import gui

            # Determine defaults
            default_fps = desired_fps if desired_fps is not None else self.fps
            default_illum = desired_illumination_mode if desired_illumination_mode is not None else self.illum_mode

            # Build choice lists with default first, then other options
            fps_choices = [default_fps] + [f for f in self.freqs if f != default_fps]
            illum_choices = [default_illum] + [m for m in self.illum_modes if m != default_illum]

            # Prepare options for GUI selection with dropdown menus
            desired_settings_dict = {
                'Hz': fps_choices,
                'Illumination mode': illum_choices
            }

            # Open GUI dialog for user to select settings
            desired_settings = gui.DlgFromDict(
                dictionary=desired_settings_dict, 
                title='Eye-Tracking Settings',
                order=['Hz', 'Illumination mode'],
                tip={'Hz': f'Available: {self.freqs}', 
                    'Illumination mode': f'Available: {self.illum_modes}'},
                screen=screen,
                alwaysOnTop=alwaysOnTop
            )
            
            # Handle cancellation
            if not desired_settings.OK:
                print("|-- Eye-tracking settings configuration cancelled by user. --|")
                return

            # Extract selected settings
            desired_fps = desired_settings_dict['Hz']
            desired_illumination_mode = desired_settings_dict['Illumination mode']

        else:
            # Validate desired FPS if provided
            if desired_fps is not None and desired_fps not in self.freqs:
                raise ValueError(
                    f"Desired FPS {desired_fps} not supported. "
                    f"Supported frequencies: {self.freqs}"
                )
            
            # Validate desired illumination mode if provided
            if desired_illumination_mode is not None and desired_illumination_mode not in self.illum_modes:
                raise ValueError(
                    f"Desired illumination mode '{desired_illumination_mode}' not supported. "
                    f"Supported modes: {self.illum_modes}"
                )
        
        # --- Apply changes and build summary ---
        changes = []
        
        if desired_fps is not None and desired_fps != self.fps:
            # Update eye tracker frequency 
            self.eyetracker.set_gaze_output_frequency(desired_fps)
            # Update internal FPS attribute 
            self.fps = desired_fps
            changes.append(f"- FPS: {old_fps} Hz --> {self.fps} Hz")
        else:
            changes.append(f"- FPS: kept at {self.fps} Hz")

        if desired_illumination_mode is not None and desired_illumination_mode != self.illum_mode:
            # Update eye tracker illumination mode 
            self.eyetracker.set_eye_tracking_mode(desired_illumination_mode)
            # Update internal illumination mode attribute
            self.illum_mode = desired_illumination_mode
            changes.append(f"- Illumination mode: {old_illum_mode} --> {self.illum_mode}")
        else:
            changes.append(f"- Illumination mode: kept at {self.illum_mode}")
        
        # --- Print summary ---
        summary = "\n".join(changes)
        NicePrint(summary, title="Applied Changes", verbose=self.verbose)


    # --- Calibration Methods ---

    def show_status(self, decision_key="space", video_help=True, overlay=False, scale=1.0, offset_y=30):
        """
        Real-time visualization of participant eye position in the track box.

        Creates an interactive positioning display showing left/right eye
        positions and distance from the screen. Updates continuously until the
        decision key is pressed.

        Parameters
        ----------
        decision_key : str, optional
            Key to press to exit visualization. Default is 'space'.
        video_help : bool or visual.MovieStim, optional
            Controls background video display. True uses the built-in
            instructional video, False disables it, and a MovieStim object uses
            that pre-loaded custom video.
        overlay : bool, optional
            Experimental: if True, use a separate Tkinter overlay with Tobii eye-camera images
            on real hardware. In simulation mode this falls back to the
            integrated PsychoPy-shapes display. Default is False.
        scale : float, optional
            Scale factor for the overlay GUI. Used only when overlay=True with a
            real eye tracker. Default is 1.0.
        offset_y : int, optional
            Vertical pixel offset for the overlay GUI relative to the PsychoPy
            window. Default is 30.

        Raises
        ------
        RuntimeError
            If positioning is requested during an active recording. Use the
            detached live monitor for eye-position feedback while recording.

        Notes
        -----
        The optional overlay is experimental: its API and platform behavior may
        change. Validate it on the intended tracker/display setup before use.
        The default integrated positioning display is not an experimental API.

        The default mode draws the track-box visualization directly into the
        PsychoPy window. The overlay mode draws eye-position feedback and eye
        images in a small Tkinter window while PsychoPy continues to show the
        optional help video.

        In simulation mode, the scroll wheel adjusts simulated distance. The
        simulation path uses a background data thread to populate user-position
        samples for the PsychoPy display.

        Positioning keeps only the latest user-position sample in a separate
        buffer; it does not update recorded gaze, gaze-contingent history, or
        the detached live monitor. Run positioning before recording; use
        `enable_live_monitor()` for eye-position feedback during recording.

        This method owns the PsychoPy flip loop until `decision_key` is pressed.
        Custom PsychoPy stimuli that should appear during positioning must be
        drawn inside this method rather than by the caller while it is running.

        Examples
        --------
        ### Integrated positioning display
        ```python
        ET_controller.show_status()
        ```

        ### Overlay display with eye-camera images
        ```python
        ET_controller.show_status(overlay=True)
        ```
        """
        if self.recording:
            raise RuntimeError(
                "Cannot show positioning during an active recording. "
                "Use enable_live_monitor() instead."
            )

        # Do not display a stale position from a previous positioning session.
        with self._buf_lock:
            self._position_data.clear()

        instructions_text = f"""Showing participant position:

    - The track box is shown as a white rectangle.
    - Both eyes are shown as colored circles.
        (try to center them within the box)
    - The green bar on the bottom indicates distance from screen.
        (try to have the black marker in the center of
        the green bar, around 60 cm from the screen)

    Press '{decision_key}' to finish positioning.
        """

        NicePrint(instructions_text, title="Participant Positioning", verbose=self.verbose)

        if not self.simulate and self.eyetracker is None:
            raise ValueError("Eye tracker not found and not in simulation mode")

        status_movie = None
        if isinstance(video_help, visual.MovieStim):
            status_movie = video_help
            status_movie.play()
        elif video_help:
            video_path = os.path.join(os.path.dirname(__file__), 'stimuli', 'ShowStatus.mp4')
            status_movie = visual.MovieStim(
                self.win,
                video_path,
                size=(1.06, 0.6),
                pos=(0, -0.08),
                loop=True,
                units='height'
            )
            status_movie.play()

        eye_gui = None
        use_overlay = overlay and not self.simulate
        subscribed_position = False
        sim_started = False

        bgrect = leye = reye = zbar = zc = zpos = None

        try:
            if use_overlay:
                from .gui.PositioningOverlay import PositioningOverlay

                eye_gui = PositioningOverlay(
                    self.eyetracker,
                    win=self.win,
                    scale=scale,
                    position='top-center',
                    offset_y=offset_y,
                    left_eye_color=cfg.colors.left_eye,
                    right_eye_color=cfg.colors.right_eye
                )
            else:
                if overlay and self.simulate:
                    print("Simulation mode: eye images are unavailable; using integrated positioning display")

                bgrect = visual.Rect(
                    self.win,
                    pos=(0, 0.4),
                    width=0.25,
                    height=0.2,
                    lineColor="white",
                    fillColor="black",
                    units="height"
                )
                leye = visual.Circle(
                    self.win,
                    size=0.02,
                    units="height",
                    lineColor=None,
                    fillColor=cfg.colors.left_eye,
                    colorSpace='rgb255'
                )
                reye = visual.Circle(
                    self.win,
                    size=0.02,
                    units="height",
                    lineColor=None,
                    fillColor=cfg.colors.right_eye,
                    colorSpace='rgb255'
                )
                zbar = visual.Rect(
                    self.win,
                    pos=(0, 0.28),
                    width=0.25,
                    height=0.03,
                    lineColor="green",
                    fillColor="green",
                    units="height"
                )
                zc = visual.Rect(
                    self.win,
                    pos=(0, 0.28),
                    width=0.01,
                    height=0.03,
                    lineColor="white",
                    fillColor="white",
                    units="height"
                )
                zpos = visual.Rect(
                    self.win,
                    pos=(0, 0.28),
                    width=0.005,
                    height=0.03,
                    lineColor="black",
                    fillColor="black",
                    units="height"
                )

                if self.simulate:
                    self.sim_z_position = 0.6
                    print("Simulation mode: Use scroll wheel to adjust Z-position (distance from screen)")
                    self._stop_simulation = threading.Event()
                    self._simulation_thread = threading.Thread(
                        target=self._simulate_data_loop,
                        args=('user_position',),
                        daemon=True
                    )
                    self.recording = True
                    self._simulation_thread.start()
                    sim_started = True
                else:
                    self.eyetracker.subscribe_to(
                        tr.EYETRACKER_USER_POSITION_GUIDE,
                        self._on_user_position,
                        as_dictionary=True
                    )
                    subscribed_position = True

            core.wait(1)

            b_show_status = True
            while b_show_status:
                if eye_gui:
                    eye_gui.update()
                    if not eye_gui.is_running():
                        break

                if status_movie:
                    status_movie.draw()

                if not eye_gui:
                    bgrect.draw()
                    zbar.draw()
                    zc.draw()

                    with self._buf_lock:
                        gaze_data = self._position_data[-1] if self._position_data else None
                    if gaze_data:
                        lv = gaze_data.get("left_user_position_validity", 0)
                        rv = gaze_data.get("right_user_position_validity", 0)
                        lx, ly, lz = gaze_data.get("left_user_position", (None, None, None))
                        rx, ry, rz = gaze_data.get("right_user_position", (None, None, None))

                        if lv:
                            lx_conv, ly_conv = Coords.get_psychopy_pos_from_user_position(
                                self.win,
                                [lx, ly],
                                "height"
                            )
                            leye.setPos((round(lx_conv * 0.25, 4), round(ly_conv * 0.2 + 0.4, 4)))
                            leye.draw()

                        if rv:
                            rx_conv, ry_conv = Coords.get_psychopy_pos_from_user_position(
                                self.win,
                                [rx, ry],
                                "height"
                            )
                            reye.setPos((round(rx_conv * 0.25, 4), round(ry_conv * 0.2 + 0.4, 4)))
                            reye.draw()

                        if lv or rv:
                            avg_z = (lz * int(lv) + rz * int(rv)) / (int(lv) + int(rv))
                            zpos.setPos((round((avg_z - 0.5) * 0.125, 4), 0.28))
                            zpos.draw()

                keys = event.getKeys()
                if eye_gui:
                    keys.extend(eye_gui.get_keys())

                for key in keys:
                    if key == decision_key or str(key).lower() == str(decision_key).lower():
                        b_show_status = False
                        break

                self.win.flip()

        finally:
            if status_movie:
                status_movie.stop()

            if eye_gui:
                eye_gui.close()

            self.win.flip()

            if sim_started:
                self.recording = False
                self._stop_simulation.set()
                if self._simulation_thread is not None and self._simulation_thread.is_alive():
                    self._simulation_thread.join(timeout=1.0)

            if subscribed_position:
                self.eyetracker.unsubscribe_from(
                    tr.EYETRACKER_USER_POSITION_GUIDE,
                    self._on_user_position
                )

            core.wait(0.5)


    def calibrate(self,
                calibration_points=5,
                infant_stims=True,
                shuffle=True,
                audio=True,
                anim_type='zoom',
                stim_size='big',
                visualization_style='circles'
    ):
        """
        Run infant-friendly calibration procedure.

        Performs eye tracker calibration using animated stimuli to engage infant 
        participants. The calibration establishes the mapping between eye position 
        and screen coordinates, which is essential for accurate gaze data collection.
        Automatically selects the appropriate calibration method based on operating 
        mode (real eye tracker vs. mouse simulation).

        Parameters
        ----------
        calibration_points : int or list of tuple, optional
            Calibration pattern specification. 

             - Use **5** for the standard 5-point pattern (4 corners + center; default). 
             - Use **9** for a comprehensive 9-point pattern (3x3 grid). 
             - Alternatively, provide a **list of tuples** with custom points in normalized 
             - coordinates [-1, 1]. Example: ``[(-0.4, 0.4), (0.4, 0.4), (0.0, 0.0)]``.

        infant_stims : True or str or list or visual stimulus, optional
            Calibration stimulus specification. Accepts multiple formats

             - **True** uses built-in stimuli from the package (default); 
             - **False** uses a default blue square; 
             - a **str** specifies a single image file path (e.g., ``'stimulus.png'``); 
             - a **list of str** provides multiple image paths; 
             - a **PsychoPy visual stimulus** (e.g., Circle, Rect, Polygon, ImageStim, ShapeStim) 
            can be used as a single object; 
             - or a **list of visual stimuli** may be provided. 
            If fewer stimuli than calibration points are given, they are automatically 
            repeated and optionally shuffled to cover all points. 
            Supported types: ImageStim, Circle, Rect, Polygon, ShapeStim. 
            TextStim and MovieStim are not supported.

        shuffle : bool, optional
            Whether to randomize stimulus presentation order after any necessary 
            repetition. Helps prevent habituation to stimulus sequence. Default is True.

        audio : True or False or None or psychopy.sound.Sound, optional
            Controls attention-getting audio during calibration:

             -**True** uses the built-in looping calibration sound (default). 
             -**False** or **None** disables audio. 
            A **psychopy.sound.Sound** object may be provided for custom audio 
            (ensure it is configured appropriately, e.g., ``loops=-1`` for continuous looping). 
            Audio plays when a calibration point is selected and fades out during data collection.

        anim_type : {'zoom', 'trill'}, optional
            Animation style for calibration stimuli:

             - **'zoom'** applies smooth size oscillation using a cosine function (default). 
             - **'trill'** uses rapid rotation with intermittent pauses.

        stim_size : {'big', 'small'}, optional
            Size preset for calibration stimuli:

             - **'big'** uses larger stimuli recommended for infants and children (default). 
             -**'small'** uses smaller stimuli for adults.

        visualization_style : {'circles', 'lines'}, optional
            How to display calibration results:

             - **'circles'** shows small filled circles at each gaze sample position. 
             - **'lines'** draws lines from targets to gaze samples. 

        Returns
        -------
        bool
            True if calibration completed successfully and was accepted by the user. 
            False if calibration was aborted (e.g., via ESC key) or failed.

        Raises
        ------
        ValueError
            If `calibration_points` is not 5, 9, or a valid list of coordinate tuples; 
            if `visualization_style` is not 'circles' or 'lines'; 
            or if `infant_stims` format is unrecognized.
        TypeError
            If pre-loaded stimuli include unsupported types (e.g., TextStim, MovieStim).

        Examples
        --------
        ### Calibration Point Patterns
        
        #### Standard 5-point calibration
        ```python
            controller.calibrate(5)
        ```
        
        #### Comprehensive 9-point calibration
        ```python
            controller.calibrate(9)
        ```
        
        #### Custom calibration points
        ```python
            custom_points = [
                (0.0, 0.0),      # Center
                (-0.5, 0.5),     # Top-left
                (0.5, 0.5),      # Top-right
                (-0.5, -0.5),    # Bottom-left
                (0.5, -0.5)      # Bottom-right
            ]
            controller.calibrate(custom_points)
        ```
        
        ### Stimulus Options
        
        #### Single image file
        ```python
            controller.calibrate(5, infant_stims='my_stimulus.png')
        ```
        
        #### Multiple image files
        ```python
            controller.calibrate(5, infant_stims=['stim1.png', 'stim2.png', 'stim3.png'])
        ```
        
        #### Single shape stimulus
        ```python
            red_square = visual.Rect(win, size=0.08, fillColor='red', units='height')
            controller.calibrate(5, infant_stims=red_square)
        ```
        
        #### Multiple shape stimuli
        ```python
            shapes = [
                visual.Circle(win, radius=0.04, fillColor='red', units='height'),
                visual.Rect(win, size=0.08, fillColor='blue', units='height'),
                visual.Polygon(win, edges=6, radius=0.04, fillColor='green', units='height')
            ]
            controller.calibrate(5, infant_stims=shapes, shuffle=True)
        ```
        
        ### Animation and Audio
        
        #### Custom audio
        ```python
            from psychopy import sound
            my_sound = sound.Sound('custom_beep.wav', loops=-1)
            controller.calibrate(5, audio=my_sound)
        ```
        
        #### Trill animation without audio
        ```python
            controller.calibrate(5, audio=False, anim_type='trill')
        ```
        
        ### Visualization Styles
        
        #### Lines visualization
        ```python
            controller.calibrate(5, visualization_style='lines')
        ```
        
        #### Circles visualization with small stimuli
        ```python
            controller.calibrate(5, stim_size='small', visualization_style='circles')
        ```
        
        ### Complete Workflows
        
        #### Full calibration workflow
        ```python
            # Position participant
            controller.show_status()
            
            # Run calibration
            success = controller.calibrate(
                calibration_points=9,
                infant_stims=['stim1.png', 'stim2.png'],
                shuffle=True,
                audio=True,
                anim_type='zoom',
                visualization_style='circles'
            )
            
            if success:
                controller.start_recording('data.h5')
                # ... run experiment ...
                controller.stop_recording()
        ```
        """
        # --- Visualization Style Validation ---
        valid_styles = ['lines', 'circles']
        if visualization_style not in valid_styles:
            raise ValueError(
                f"Invalid visualization_style: '{visualization_style}'. "
                f"Must be one of {valid_styles}."
            )
        
        # --- Calibration Points Processing ---
        if isinstance(calibration_points, int):
            # Load standard pattern from config
            if calibration_points == 5:
                norm_points = cfg.calibration.points_5
            elif calibration_points == 9:
                norm_points = cfg.calibration.points_9
            else:
                raise ValueError(
                    f"Invalid calibration_points={calibration_points}. "
                    f"Must be 5, 9, or a list of (x, y) coordinates."
                )

        elif isinstance(calibration_points, list):
            if not calibration_points:
                raise ValueError("calibration_points list cannot be empty.")
            
            for i, point in enumerate(calibration_points):
                if not isinstance(point, tuple) or len(point) != 2:
                    raise ValueError(f"Point {i} must be a tuple (x, y). Got: {point}")
                
                x, y = point
                if not (-1 <= x <= 1 and -1 <= y <= 1):
                    raise ValueError(
                        f"Point {i} ({x}, {y}) out of range [-1, 1]."
                    )
            # ✅ Set norm_points to the user-provided list
            norm_points = calibration_points
            
        else:
            raise ValueError(
                f"calibration_points must be int (5 or 9) or list of tuples. "
                f"Got: {type(calibration_points).__name__}"
            )
        
        # Number of calibration points
        num_points = len(norm_points)


        # --- Stimulus Size Validation ---
        valid_sizes = ['big', 'small']
        if stim_size not in valid_sizes:
            raise ValueError(
                f"Invalid stim_size: '{stim_size}'. "
                f"Must be one of {valid_sizes}."
            )
        
        # --- Stimuli Processing ---
        # Handle single stimulus or list of stimuli
        if infant_stims is False:
            # Create default square stimulus
            default_square = visual.Rect(
                self.win,
                size=0.08,  # 8% of screen height
                fillColor='#2e5576',  # Deep blue color
                lineColor=None,
                units='height'
            )
            stim_list = [default_square]
            
        elif isinstance(infant_stims, list):
            # Already a list - use as is
            if len(infant_stims) > 0:
                stim_list = infant_stims
            else:
                raise ValueError("infant_stims list cannot be empty.")
                
        elif infant_stims is True:
            # Load default stimuli from package
            import glob
            
            package_dir = os.path.dirname(__file__)
            stimuli_dir = os.path.join(package_dir, 'stimuli')
            
            # Get all PNG files
            stim_list = glob.glob(os.path.join(stimuli_dir, '*.png'))
            stim_list.sort()  # Consistent ordering
            
        elif hasattr(infant_stims, 'draw') or isinstance(infant_stims, str):
            # Single stimulus (shape object or file path) - wrap in list
            stim_list = [infant_stims]
            
        else:
            raise ValueError(
                f"infant_stims must be True, False, a stimulus object, a file path string, "
                f"or a list of stimuli. Got: {type(infant_stims).__name__}"
            )
        
        # --- Repeat stimuli if needed to cover all calibration points ---
        num_stims = len(stim_list)
        if num_stims < num_points:
            repetitions = (num_points // num_stims) + 1
            stim_list = stim_list * repetitions
        
        # --- Shuffle if requested ---
        if shuffle:
            import random
            random.shuffle(stim_list)
        
        # --- Subset to exact number needed ---
        stim_list = stim_list[:num_points]

        # --- Setup audio stimulus ---
        audio_stim = None

        if audio is not None and audio is not False:
            from psychopy import sound
            
            if isinstance(audio, sound.Sound):
                audio_stim = audio
            elif audio is True:
                audio_path = os.path.join(os.path.dirname(__file__), 'stimuli', 'CalibrationSound.wav')
                audio_stim = sound.Sound(audio_path, loops=-1)

        # --- Mode-specific calibration setup ---
        if self.simulate:
            session = MouseCalibrationSession(
                win=self.win,
                infant_stims=stim_list,
                mouse=self.mouse,
                audio=audio_stim,
                anim_type=anim_type,
                stim_size=stim_size,  
                visualization_style=visualization_style,
                verbose=self.verbose
            )
        else:
            session = TobiiCalibrationSession(
                win=self.win,
                calibration_api=self.calibration,
                infant_stims=stim_list,
                audio=audio_stim,
                anim_type=anim_type,
                stim_size=stim_size,  
                visualization_style=visualization_style,
                verbose=self.verbose
            )

        # --- Run calibration ---
        success = session.run(norm_points)

        return success

    def save_calibration(self, filename=None, use_gui=False, screen=-1, alwaysOnTop=True):
        """
        Save the current calibration data to a file.
        
        Retrieves the active calibration data from the connected Tobii eye tracker
        and saves it as a binary file. This can be reloaded later with
        `load_calibration()` to avoid re-calibrating the same participant.
        
        Parameters
        ----------
        filename : str | None, optional
            Desired output path. If None and `use_gui` is False, a timestamped default
            name is used (e.g., 'YYYY-mm-dd_HH-MM-SS_calibration.dat').
            If provided without an extension, '.dat' is appended.
            If an extension is already present, it is left unchanged.
        use_gui : bool, optional
            If True, opens a file-save dialog (Psychopy) where the user chooses the path.
            The suggested name respects the logic above. Default False.
        screen : int, optional
            Screen number where the GUI dialog is displayed. Only used when `use_gui=True`.
            If -1 (default), the dialog appears on the primary screen. Use 0, 1, 2, etc. 
            to specify other monitors. Ignored when `use_gui=False`.
        alwaysOnTop : bool, optional
            Whether the GUI dialog stays on top of other windows. Only used when 
            `use_gui=True`. Default is True to prevent the dialog from being hidden 
            behind experiment windows. Ignored when `use_gui=False`.
        
        Returns
        -------
        bool
            True if saved successfully; False if cancelled, no data available, in
            simulation mode, or on error.
        
        Details
        -------
        - In simulation mode, saving is skipped and a warning is issued.
        - If `use_gui` is True and the dialog is cancelled, returns False.
        
        Examples
        --------
        #### Save with default timestamped name
        ```python
        ET_controller.save_calibration()
        ```
        
        #### Save with specified filename
        ```python
        ET_controller.save_calibration('subject_01_calib.dat')
        ```
        
        #### Use GUI to choose save location
        ```python
        ET_controller.save_calibration(use_gui=True)
        ```
        
        #### GUI on secondary monitor
        ```python
        ET_controller.save_calibration(use_gui=True, screen=1, alwaysOnTop=False)
        ```
        """
        # --- Simulation guard ---
        if self.simulate:
            warnings.warn(
                "Skipping calibration save: running in simulation mode. "
                "Saving requires a real Tobii eye tracker."
            )
            return False

        # --- Recording guard ---
        if self.recording:
            warnings.warn(
                "|-- Ongoging recording!! --|\n"
                "Better to save calibration before or after recording.\n",
                UserWarning
            )
            return

        try:
            # --- Build a default or normalized filename ---
            if filename is None:
                # Default timestamped base name
                base = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_calibration"
                path = Path(base).with_suffix(".dat")
            else:
                p = Path(filename)
                # If no suffix, add .dat; otherwise, respect the existing extension
                path = p if p.suffix else p.with_suffix(".dat")

            if use_gui:
                from psychopy import gui
                # Use the computed name as the suggested default
                save_path = gui.fileSaveDlg(
                    prompt='Save calibration data as…',
                    # Psychopy expects a string path; supply our suggested default
                    initFilePath=str(path),
                    allowed='*.dat',
                    screen=screen,
                    alwaysOnTop=alwaysOnTop
                )
                if not save_path:
                    print("|-- Save calibration cancelled by user. --|")
                    return False
                # Normalize selection: ensure .dat if user omitted extension
                sp = Path(save_path)
                path = sp if sp.suffix else sp.with_suffix(".dat")

            # --- Retrieve calibration data ---
            calib_data = self.eyetracker.retrieve_calibration_data()
            if not calib_data:
                warnings.warn("No calibration data available to save.")
                return False

            # --- Write to disk ---
            with open(path, 'wb') as f:
                f.write(calib_data)

            NicePrint(f"Calibration data saved to:\n{path}", title="Calibration Saved", verbose=self.verbose)
            return True

        except Exception as e:
            warnings.warn(f"Failed to save calibration data: {e}")
            return False

    def load_calibration(self, filename=None, use_gui=False, screen=-1, alwaysOnTop=True):
        """
        Loads calibration data from a file and applies it to the eye tracker.
        
        This method allows reusing a previously saved calibration, which can save
        significant time for participants, especially in multi-session studies.
        The calibration data must be a binary file generated by a Tobii eye tracker,
        typically via the `save_calibration()` method. This operation is only
        available when connected to a physical eye tracker.

        Parameters
        ----------
        filename : str, optional
            The path to the calibration data file (e.g., "subject_01_calib.dat").
            If `use_gui` is `True`, this path is used as the default suggestion
            in the file dialog. If `use_gui` is `False`, this parameter is
            required.
        use_gui : bool, optional
            If `True`, a graphical file-open dialog is displayed for the user to
            select the calibration file. Defaults to `False`.
        screen : int, optional
            Screen number where the GUI dialog is displayed. Only used when `use_gui=True`.
            If -1 (default), the dialog appears on the primary screen. Use 0, 1, 2, etc. 
            to specify other monitors. Ignored when `use_gui=False`.
        alwaysOnTop : bool, optional
            Whether the GUI dialog stays on top of other windows. Only used when 
            `use_gui=True`. Default is True to prevent the dialog from being hidden 
            behind experiment windows. Ignored when `use_gui=False`.
        
        Returns
        -------
        bool
            Returns `True` if the calibration was successfully loaded and applied,
            and `False` otherwise (e.g., user cancelled the dialog, file not
            found, or data was invalid).
            
        Raises
        ------
        RuntimeError
            If the method is called while the ETracker is in simulation mode.
        ValueError
            If `use_gui` is `False` and `filename` is not provided.
        
        Examples
        --------
        #### Load calibration from specific file
        ```python
        success = ET_controller.load_calibration('subject_01_calib.dat')
        if success:
            ET_controller.start_recording('subject_01_data.h5')
        ```
        
        #### Use GUI to select file
        ```python
        success = ET_controller.load_calibration(use_gui=True)
        ```
        
        #### GUI on secondary monitor
        ```python
        success = ET_controller.load_calibration(
            use_gui=True, 
            screen=1, 
            alwaysOnTop=False
        )
        ```
        
        #### Multi-session workflow
        ```python
        # Session 1: Calibrate and save
        ET_controller.calibrate(5)
        ET_controller.save_calibration('participant_123.dat')
        ET_controller.start_recording('session_1.h5')
        # ... run experiment ...
        ET_controller.stop_recording()
        
        # Session 2: Load previous calibration
        ET_controller.load_calibration('participant_123.dat')
        ET_controller.start_recording('session_2.h5')
        # ... run experiment ...
        ET_controller.stop_recording()
        ```
        """
        # --- Pre-condition Check: Ensure not in simulation mode ---
        # Calibration can only be applied to a physical eye tracker.
        if self.simulate:
            raise RuntimeError(
                "Cannot load calibration in simulation mode. "
                "Calibration loading requires a real Tobii eye tracker."
            )
        
        # --- Recording guard ---
        if self.recording:
            warnings.warn(
                "|-- Ongoging recording!! --|\n"
                "Better to load calibration before or after recording.\n",
                UserWarning
            )
            return

        # --- Determine the file path to load ---
        load_path = None
        if use_gui:
            from psychopy import gui
            
            # Use the provided filename as the initial path, otherwise start in the current directory.
            start_path = filename if filename else '.'
            # Open a file dialog to let the user choose the calibration file.
            file_list = gui.fileOpenDlg(
                prompt='Select calibration file to load…',
                allowed='*.dat',
                tryFilePath=start_path,
                screen=screen,
                alwaysOnTop=alwaysOnTop
            )
                
            # The dialog returns a list; if cancelled, it's None.
            if file_list:
                load_path = file_list[0]
            else:
                # User cancelled the dialog, so we stop here.
                print("|-- Load calibration cancelled by user. --|")
                return False
        else:
            # If not using the GUI, a filename must be explicitly provided.
            if filename is None:
                raise ValueError(
                    "A filename must be provided when `use_gui` is False."
                )
            load_path = filename

        # --- Load and Apply Calibration Data ---
        try:
            # Open the file in binary read mode ('rb').
            with open(load_path, 'rb') as f:
                calib_data = f.read()

            # The tracker expects a non-empty bytestring.
            if not calib_data:
                warnings.warn(f"Calibration file is empty: {load_path}")
                return False

            # Apply the loaded data to the eye tracker.
            self.eyetracker.apply_calibration_data(calib_data)

            # --- Final Confirmation ---
            NicePrint(f"Calibration data loaded from:\n{load_path}",
                    title="Calibration Loaded", verbose=self.verbose)
            return True

        except FileNotFoundError:
            # Handle the case where the specified file does not exist.
            warnings.warn(f"Calibration file not found at: {load_path}")
            return False
        except Exception as e:
            # Catch any other errors during file I/O or from the Tobii SDK.
            warnings.warn(f"Failed to load and apply calibration data: {e}")
            return False

        
    # --- Recording Methods ---

    def start_recording(self, filename=None, raw_format=False, coordinate_units='default'):
        """
        Begin gaze data recording session.
        
        Clears pending gaze samples, resets the timestamp origin, prepares the
        output file, and starts tracker or simulation data collection. HDF5 files
        begin with metadata and empty typed gaze/events tables; CSV files begin
        with a header row. Data rows are written by save_data(), not continuously.
        
        Parameters
        ----------
        filename : str, optional
            Output filename for gaze data. If None, generates timestamp-based
            name. File extension determines format (.h5/.hdf5 for HDF5,
            .csv for CSV, defaults to .h5).
        raw_format : bool, optional
            If True, uses the configured raw SDK fields, splitting coordinate
            tuples into x/y/z columns and preserving SDK microsecond timestamps.
            If False (default), uses simplified column names and subset of columns
            (gaze positions, pupil diameters, validity flags only).
            See [Data Format Options](#data-format-options) for more information.
        coordinate_units : str, optional
            Unit for the 2D screen coordinates in the **simplified** format only
            (ignored—and warned—under raw format, which always keeps the Tobii
            ADCS 0-1 coordinates untouched). Default is 'default' (= 'pix').

            Simplified 2D coordinates are always written in the PsychoPy
            convention: origin at screen center, y-axis pointing up. This
            parameter only selects the *unit/scale*, never the origin:

            - **'default'** → 'pix' (PsychoPy pixels)
            - **'pix'**, **'height'**, **'norm'**, **'cm'**, **'deg'**

            See [Coordinate System](#coordinate-system) for more information.

        Raises
        ------
        ValueError
            If simplified coordinate_units or the filename extension is unsupported.

        Warns
        -----
        UserWarning
            - If a recording is already running, the call is ignored.
            - If the output file already exists, a timestamp is added to the
              filename so the old file is not overwritten.
            - If ``raw_format=True`` and ``coordinate_units`` is not 'default',
              ``coordinate_units`` is ignored.
        
        Notes
        -----
        ### Data Format: fidelity vs convenience

        `raw_format` is a *fidelity* switch. It fixes the coordinate system *and*
        the time representation, so the two formats never silently disagree:

        - **Simplified** (`raw_format=False`): the analysis-ready, PsychoPy-native
          view. 2D gaze is written in PsychoPy coordinates (origin at the screen
          center, y up) in the unit chosen with `coordinate_units`; time is
          standardised to milliseconds measured from the first sample.
        - **Raw** (`raw_format=True`): the data exactly as the Tobii Pro SDK
          delivered it. 2D gaze stays in Tobii ADCS (0-1, origin top-left, y down);
          3D eye data retains its SDK User Coordinate System units; time stays in
          the SDK's original units (microseconds). No conversion is applied, and
          `coordinate_units` is ignored.

        This keeps the full geometric and temporal information available in raw
        files without any loss, while simplified files drop straight onto your
        PsychoPy stimuli for typical AOI / scanpath / pupil analyses.

        ### Coordinate System

        The origin of the 2D gaze coordinates is fixed by the format:

        - **Simplified**: always PsychoPy-convention (origin at screen center,
          y up). `coordinate_units` selects only the unit/scale (`'pix'` by
          default; also `'height'`, `'norm'`, `'cm'`, `'deg'`). Because the file
          stores `screen_size` and `window_units` as metadata, any other
          coordinate system can be re-derived at read time.
        - **Raw**: always Tobii ADCS (0-1, top-left origin, y down), untouched.

        The 3D spatial columns present only in raw format (`*_in_user_coordinate_system_*`
        for gaze points and gaze origins) retain their SDK User Coordinate System
        units and are never converted under any setting.

        ### Timestamps

        Time representation is also fixed by the format:

        - **Simplified**: standardised — milliseconds, measured from 0 at the
          first gaze sample. The same convention is applied to both gaze samples
          and event markers so they stay aligned.
        - **Raw**: original — both `system_time_stamp` and `device_time_stamp`
          are kept in the SDK's native microseconds, untouched, for
          synchronisation with other recording systems.

        Simulation generates timestamps from the experiment clock, which resets
        at recording start; raw output does not subtract another timestamp origin.
        
        Examples
        --------
        ### Basic Usage
        
        #### Standard simplified format (PsychoPy pixels, relative timestamps)
        ```python
        ET_controller.start_recording('data.h5')
        # Creates: Left_X/Left_Y in pixels (center origin), TimeStamp from 0ms
        ```
        
        #### Auto-generated timestamped filename
        ```python
        ET_controller.start_recording()  # Creates YYYY-MM-DD_HH-MM-SS.h5
        ```
        
        ### Format Options

        #### Simplified format with defaults (PsychoPy pixels, relative ms)
        ```python
        ET_controller.start_recording('data.h5')
        # Simplified columns, 2D coords in pixels (center origin), TimeStamp from 0ms
        ```

        #### Simplified format with height units
        ```python
        ET_controller.start_recording('data.h5', coordinate_units='height')
        # Simplified columns, 2D coords in height units (still center origin)
        ```

        #### Raw format (full Tobii SDK, untouched)
        ```python
        ET_controller.start_recording('data.h5', raw_format=True)
        # All Tobii columns, 2D coords in ADCS (0-1, top-left), timestamps in microseconds
        # Note: coordinate_units is ignored (and warned) under raw_format=True
        ```

        ### Complete Workflows

        #### Standard experiment workflow
        ```python
        # Setup and calibration
        ET_controller.show_status()
        ET_controller.calibrate(5)

        # Start recording with defaults
        ET_controller.start_recording('participant_01.h5')

        # Run experiment with event markers
        ET_controller.record_event('trial_1_start')
        # ... present stimuli ...
        ET_controller.record_event('trial_1_end')

        # Stop recording
        ET_controller.stop_recording()
        ```

        #### Multi-file raw collection (synchronisation across sessions)
        ```python
        # Raw files keep the SDK's original microsecond timestamps, so multiple
        # files can be aligned temporally after the fact.
        ET_controller.start_recording('session1_gaze.h5', raw_format=True)
        # ... run session 1 ...
        ET_controller.stop_recording()

        ET_controller.start_recording('session2_gaze.h5', raw_format=True)
        # ... run session 2 ...
        ET_controller.stop_recording()
        ```
        """
        # --- State validation ---
        if self.recording:
            warnings.warn(
                "Recording is already in progress – start_recording() call ignored",
                UserWarning
            )
            return
        
        # --- Format flag ---
        self.raw_format = raw_format

        # --- Coordinate units ---
        # Raw format always keeps Tobii ADCS coordinates untouched: coordinate_units
        # is ignored (and warned) under raw_format=True. Simplified format always
        # uses the PsychoPy convention (center origin, y up) in the chosen unit.
        valid_units = ['pix', 'height', 'norm', 'cm', 'deg']
        if raw_format:
            self.coordinate_units = 'tobii'
            if coordinate_units != 'default':
                warnings.warn(
                    "coordinate_units is ignored under raw_format=True: raw files "
                    "always keep the original Tobii ADCS 0-1 coordinates.",
                    UserWarning,
                    stacklevel=2
                )
        else:
            if coordinate_units == 'default':
                self.coordinate_units = 'pix'
            else:
                if coordinate_units not in valid_units:
                    raise ValueError(
                        f"Invalid coordinate_units: '{coordinate_units}'. "
                        f"Must be one of {valid_units} (simplified format only)."
                    )
                self.coordinate_units = coordinate_units

        # --- Buffer initialization ---
        if self.gaze_data and not self.recording:
            self.gaze_data.clear()
        
        # --- Timing setup ---
        self.first_timestamp = None  # Each recording gets its own time origin.
        self.experiment_clock.reset()
        
        # --- File preparation ---
        self._prepare_recording(filename)
        
        # --- Recording info display ---
        self._get_info(moment='recording')

        # --- Data collection startup ---
        if self.simulate:
            # Simulation mode setup
            self._stop_simulation = threading.Event()
            self._simulation_thread = threading.Thread(
                target=self._simulate_data_loop,
                args=('gaze',),
                daemon=True
            )
            self.recording = True
            self._simulation_thread.start()
        else:
            # Real eye tracker setup

            # Subscribe to warnings
            self._subscribe_warnings()

            # Subscribe to gaze data stream
            self.eyetracker.subscribe_to(
                tr.EYETRACKER_GAZE_DATA, 
                self._on_gaze_data, 
                as_dictionary=True
            )

            # Small delay to ensure subscription is active before data starts arriving
            core.wait(1)
            self.recording = True

        
    def stop_recording(self, data_check=True):
        """
        Stop gaze data recording and finalize session.
        
        Stops data collection, attempts to save the remaining buffers, and
        optionally estimates dropped samples from saved timestamp gaps.
        
        Parameters
        ----------
        data_check : bool, optional
            If True (default), performs data quality check by reading the saved file
            and analyzing timestamp gaps to detect dropped samples. If False, skips
            the quality check and completes faster. Default True.
        
        Warns
        -----
        UserWarning
            If recording is not currently active; the method then returns.
            
        Details
        -------
        - Calls save_data() after stopping data collection; its limitations apply
        - Recording duration is measured from start_recording() call
        - Quality check reads all saved timestamps to analyze gaps between samples,
        including potential gaps between save_data() calls
        - At 120Hz, samples should be about 8333µs apart in raw format or 8.33ms
        in simplified format; larger gaps are used to estimate dropped samples
        
        Examples
        --------
        ### Basic Usage
        
        #### Standard stop with quality check
        ```python
        ET_controller.start_recording('data.h5')
        # ... run experiment ...
        ET_controller.stop_recording()  # Shows quality report
        ```
        
        #### Skip quality check for faster shutdown
        ```python
        ET_controller.stop_recording(data_check=False)
        ```
        
        ### Understanding Output
        
        #### Expected quality check report
        ```python
        # When data_check=True, you'll see output like:
        # ╔════════════════════════════╗
        # ║  Recording Complete        ║
        # ╠════════════════════════════╣
        # ║ Data collection lasted     ║
        # ║ approximately 120.45 sec   ║
        # ║ Data has been saved to     ║
        # ║ experiment.h5              ║
        # ║                            ║
        # ║ Data Quality Report:       ║
        # ║   - Total samples: 14454   ║
        # ║   - Dropped samples: 0     ║
        # ╚════════════════════════════╝
        ```
        """
        # --- State validation ---
        # Ensure recording is actually active before attempting to stop
        if not self.recording:
            warnings.warn(
                "Recording is not currently active - stop_recording() call ignored",
                UserWarning
            )
            return
        
        # Wait while acquisition can still deliver samples.
        self._check_gaze_samples()

        # --- Stop data collection ---
        self.recording = False
        
        # --- Mode-specific cleanup ---
        # Clean up resources based on recording mode
        if self.simulate:
            # --- Simulation cleanup ---
            # Signal simulation thread to stop
            if self._stop_simulation is not None:
                self._stop_simulation.set()
            
            # Wait for simulation thread to finish (with timeout)
            if self._simulation_thread is not None:
                self._simulation_thread.join(timeout=1.0)
                
        else:
            # --- Real eye tracker cleanup ---

            # Unsubscribe from warnings
            self._unsubscribe_warnings()

            # Unsubscribe from Tobii SDK data stream
            self.eyetracker.unsubscribe_from(tr.EYETRACKER_GAZE_DATA, self._on_gaze_data)

        # --- Live monitor cleanup ---
        self.stop_live_monitor()
        
        # --- Save final batch ---
        self.save_data()
        
        # --- Quality check ---
        if data_check:
            quality = self._check_continuity()
        else:
            quality = None
        
        # --- Session summary ---
        duration_seconds = self.experiment_clock.getTime()
        
        summary = (
            f'Data collection lasted approximately {duration_seconds:.2f} seconds\n'
            f'Data has been saved to {self.filename}\n'
        )
        
        if quality:
            summary += (
                f'\nData check Report:\n'
                f'  - Total samples: {quality["total_samples"]}\n'
                f'  - Dropped samples: {quality["dropped_samples"]}\n'
            )
        
        NicePrint(summary, title="Recording Complete", verbose=self.verbose)


    def record_event(self, label):
        """
        Record timestamped experimental event during data collection.
        
        Events are merged with gaze data based on timestamp proximity
        during save operations. Uses appropriate timing source for
        simulation vs. real eye tracker modes.
        
        Parameters
        ----------
        label : str
            Descriptive label for the event (e.g., 'trial_start', 'stimulus_onset').
            
        Raises
        ------
        RuntimeWarning
            If called when recording is not active.
        
        Details
        -------
        ### Event-Gaze Synchronization
        
        Events are stored separately and merged with gaze data when `save_data()` is
        called. Each event is aligned to the next closest gaze sample (at or after the
        event timestamp) using binary search. When multiple events occur within the same
        sampling interval, they are concatenated with pipe delimiters in the Events
        column (e.g., 'fixation_offset | stimulus_onset'). HDF5 also stores each
        marker separately in the events table. The fixed-width text limit still
        applies to both individual and combined labels when writing HDF5.
            
        Examples
        --------
        ### Basic Usage
        
        #### Recording single events
        ```python
        ET_controller.record_event('trial_1_start')
        # ... present stimulus ...
        ET_controller.record_event('stimulus_offset')
        ```
        
        ### Common Patterns
        
        #### Complete trial structure
        ```python
        ET_controller.record_event('trial_1_start')
        ET_controller.record_event('fixation_onset')
        core.wait(1.0)
        ET_controller.record_event('fixation_offset')
        
        ET_controller.record_event('stimulus_onset')
        # ... show stimulus ...
        ET_controller.record_event('stimulus_offset')
        
        ET_controller.record_event('response_prompt')
        # ... wait for response ...
        ET_controller.record_event('response_recorded')
        ET_controller.record_event('trial_1_end')
        ```
        
        #### Multi-trial experiment
        ```python
        ET_controller.start_recording('experiment.h5')
        
        for trial_num in range(10):
            ET_controller.record_event(f'trial_{trial_num}_start')
            # ... run trial ...
            ET_controller.record_event(f'trial_{trial_num}_end')
        
        ET_controller.stop_recording()
        ```
        
        #### Rapid successive events
        ```python
        # Events occurring within same sampling interval
        ET_controller.record_event('fixation_offset')
        ET_controller.record_event('stimulus_onset')
        
        # In saved data, may appear as: "fixation_offset; stimulus_onset"
        ```
        """
        # --- State validation ---
        if not self.recording:
            raise RuntimeWarning(
                "Cannot record event: recording session is not active. "
                "Call start_recording() first to begin data collection."
            )
        
        # --- Timestamp generation ---
        if self.simulate:
            # Simulation timing
            timestamp = self.experiment_clock.getTime() * 1_000_000  # Convert to microseconds
        else:
            # Real eye tracker timing
            timestamp = tr.get_system_time_stamp()  # Already in microseconds
        
        # --- Event storage ---
        self.event_data.append({
            'system_time_stamp': timestamp,
            'Events': label
        })


    def save_data(self):
        """
        Save buffered gaze and event data to file with optimized processing.
        
        Uses thread-safe buffer swapping to minimize lock time, then processes
        and saves data in CSV or HDF5 format. Events are merged with gaze data
        based on timestamp proximity.
        
        This method is typically called automatically by `stop_recording()`, but
        can be called manually during recording to periodically save data and
        clear buffers. This is useful for long experiments to avoid memory buildup
        and reduce the amount of data still in memory if the program crashes.
        
        Details
        -------
        - Automatically called by `stop_recording()`
        - Can be called during active recording; simultaneous saves are not serialized
        - Detaches buffers before processing; new gaze callbacks use fresh buffers
        - Events are attached to the first gaze sample at or after their timestamp
        - In HDF5 format, events are saved in two places:
          1. Merged into the main gaze table's 'Events' column
          2. As a separate 'events' table for independent event analysis
        - In CSV format, events only appear in the 'Events' column

        Notes
        -----
        If the detached gaze buffer is empty, the method returns without writing
        its events. A processing or write failure does not restore the detached
        buffers. HDF5 table appends are not transactional. Long merged labels
        remain subject to the fixed-width event-text limit.
        
        Examples
        --------
        ### Automatic Usage
        
        #### Default behavior (most common)
        ```python
        ET_controller.start_recording('data.h5')
        # ... run experiment ...
        ET_controller.stop_recording()  # Automatically calls save_data()
        ```
        
        ### Manual Periodic Saves
        
        #### Save every N trials for long experiments
        ```python
        ET_controller.start_recording('long_experiment.h5')
        
        for trial in range(100):
            ET_controller.record_event(f'trial_{trial}_start')
            # ... present stimuli ...
            ET_controller.record_event(f'trial_{trial}_end')
            
            # Save data every 10 trials to prevent memory buildup
            if (trial + 1) % 10 == 0:
                ET_controller.save_data()  # Saves and clears buffers
        
        ET_controller.stop_recording()
        ```
        
        ### Strategic Save Points
        
        #### Save at natural break points between blocks
        ```python
        ET_controller.start_recording('session.h5')
        
        # Block 1
        for trial in range(20):
            # ... run trial ...
            pass
        ET_controller.save_data()  # Save after block 1
        
        # Short break
        core.wait(30)
        
        # Block 2
        for trial in range(20):
            # ... run trial ...
            pass
        ET_controller.save_data()  # Save after block 2
        
        ET_controller.stop_recording()
        ```
        """
        # --- Performance monitoring ---
        start_saving = core.getTime()
        
        # --- Ensure event-gaze synchronization ---
        if self.recording:
            self._check_gaze_samples()
        
        # --- Thread-safe buffer swap (O(1) operation) ---
        # Swap buffers under lock to minimize thread blocking time
        with self._buf_lock:
            save_gaze,     self.gaze_data  = self.gaze_data,  deque()
            save_events,   self.event_data = self.event_data, deque()
        
        # --- Data validation ---
        # Log buffer sizes for monitoring and check if processing is needed
        gaze_count = len(save_gaze)
        event_count = len(save_events)
        
        if gaze_count == 0:
            print("|-- No new gaze data to save --|")
            return
        
        # --- Gaze data processing ---
        # Convert buffered data to DataFrame and prepare Events column
        gaze_df = pd.DataFrame(save_gaze)
        gaze_df['Events'] = pd.array([''] * len(gaze_df), dtype='string')
        
        # --- Event data processing and merging ---
        if event_count > 0:
            # Convert events to DataFrame
            events_df = pd.DataFrame(save_events)

            # --- Timestamp-based event merging ---
            # Map each event to the first gaze sample at or after its timestamp.
            idx = np.searchsorted(gaze_df['system_time_stamp'].values,
                                events_df['system_time_stamp'].values,
                                side='left')

            # Events past the last gaze sample have no sample to merge into.
            # They stay in the events table with their real timestamp; only the
            # inline gaze column skips them.
            covered = idx < len(gaze_df)
            covered_idx = idx[covered]
            covered_labels = events_df['Events'].values[covered]

            if len(covered_idx) > 0:
                # Check if multiple events map to the same sample
                if len(covered_idx) != len(np.unique(covered_idx)):
                    # Duplicates detected - join them onto one sample
                    grouped = pd.Series(covered_labels).groupby(covered_idx).agg(
                        lambda x: ' | '.join(x))
                    gaze_df.loc[grouped.index, 'Events'] = grouped.values
                else:
                    # No duplicates - fast vectorized assignment
                    gaze_df.iloc[covered_idx, gaze_df.columns.get_loc('Events')] = covered_labels
        else:
            print("|-- No new events to save --|")
            events_df = None
        
        # --- Data format adaptation ---
        # Convert coordinates, normalize timestamps, optimize data types
        gaze_df, events_df = self._adapt_gaze_data(gaze_df, events_df)
        
        # --- File output ---
        # Save using appropriate format handler
        if self.file_format == 'csv':
            self._save_csv_data(gaze_df)
        elif self.file_format == 'hdf5':  
            self._save_hdf5_data(gaze_df, events_df)
        
        # --- Performance reporting ---
        save_duration = round(core.getTime() - start_saving, 3)
        print(f"|-- Data saved in {save_duration} seconds --|")


    # --- Real-time Methods ---


    def gaze_contingent(self, N=0.5, units='seconds'):
        """
        Initialize real-time gaze buffer for contingent applications.
        
        Creates a rolling buffer that stores recent gaze samples over a specified
        time window or number of samples, enabling real-time gaze-contingent paradigms.
        Must be called before using `get_gaze_position()` for real-time gaze tracking.
        
        The buffer automatically maintains the most recent samples, discarding
        older data. This provides a stable estimate of current gaze position by
        aggregating across multiple samples.
        
        Parameters
        ----------
        N : float or int, optional
            Buffer size specification. Interpretation depends on `units` parameter:

             - If units='seconds': Duration in seconds (e.g., 0.5 = 500ms window)
             - If units='samples': Number of gaze samples (e.g., 5 = last 5 samples)
            Default is 0.5 (seconds).
        units : str, optional
            Unit for buffer size specification: 'seconds' or 'samples'.

             - 'seconds' (default): N specifies time duration, automatically calculates
            required samples based on eye tracker frequency
             - 'samples': N specifies exact number of samples
        
        Raises
        ------
        TypeError
            If N is not numeric or units is not a string.
        ValueError
            If units is not 'seconds' or 'samples', or if calculated buffer size < 1.
        
        Details
        -------
        - Call this method ONCE before your experimental loop
        - Buffer size trades off stability vs. latency:
        * Shorter duration/fewer samples: Lower latency, more noise
        * Longer duration/more samples: Smoother tracking, higher latency
        - For 120 Hz tracker with N=0.5 seconds: 60 samples, ~500ms latency
        - For 60 Hz tracker with N=0.5 seconds: 30 samples, ~500ms latency
        - For any tracker with N=5 samples: 5 samples, variable latency by fps
        
        Examples
        --------
        ### Basic Usage
        
        #### Default time-based buffer (0.5 seconds)
        ```python
            # Initialize with 500ms window (adapts to tracker frequency)
            ET_controller.gaze_contingent()  # Uses default N=0.5, units='seconds'
            ET_controller.start_recording('data.h5')
            
            # Create gaze-contingent stimulus
            circle = visual.Circle(win, radius=0.05, fillColor='red')
            
            for frame in range(600):  # 10 seconds at 60 fps
                gaze_pos = ET_controller.get_gaze_position()
                circle.pos = gaze_pos
                circle.draw()
                win.flip()
            
            ET_controller.stop_recording()
        ```
        
        #### Custom time window
        ```python
            # 250ms window for lower latency
            ET_controller.gaze_contingent(N=0.25, units='seconds')
            
            # 1 second window for very smooth tracking
            ET_controller.gaze_contingent(N=1.0, units='seconds')
        ```
        
        ### Sample-Based Configuration
        
        #### Explicit sample count
        ```python
            # Exactly 5 most recent samples
            ET_controller.gaze_contingent(N=5, units='samples')
            
            # Exactly 10 samples for smoother tracking
            ET_controller.gaze_contingent(N=10, units='samples')
        ```
        
        ### Complete Applications
        
        #### Gaze-contingent window paradigm
        ```python
            # 300ms time window (auto-adjusts to tracker frequency)
            ET_controller.gaze_contingent(N=0.3, units='seconds')
            ET_controller.start_recording('gaze_window.h5')
            
            stimulus = visual.ImageStim(win, 'image.png')
            window = visual.Circle(win, radius=0.1, fillColor=None, lineColor='white')
            
            for trial in range(20):
                stimulus.draw()
                
                for frame in range(120):  # 2 seconds
                    gaze_pos = ET_controller.get_gaze_position()
                    window.pos = gaze_pos
                    window.draw()
                    win.flip()
                
                ET_controller.record_event(f'trial_{trial}_end')
            
            ET_controller.stop_recording()
        ```
        """
        # --- Input validation ---
        if not isinstance(N, (int, float)):
            raise TypeError(
                f"Invalid buffer size for gaze_contingent(): expected int or float, "
                f"got {type(N).__name__}. Received value: {N}"
            )
        
        if not isinstance(units, str):
            raise TypeError(
                f"Invalid units for gaze_contingent(): expected str, "
                f"got {type(units).__name__}. Received value: {units}"
            )
        
        # --- Units validation and conversion ---
        valid_units = ['seconds', 'samples']
        if units not in valid_units:
            raise ValueError(
                f"Invalid units: '{units}'. Must be one of {valid_units}."
            )
        
        # --- Calculate buffer size in samples ---
        if units == 'seconds':
            # Convert time duration to number of samples based on tracker frequency
            buffer_size = int(N * self.fps)
            if buffer_size < 1:
                raise ValueError(
                    f"Buffer size too small: {N} seconds at {self.fps} Hz = {buffer_size} samples. "
                    f"Minimum is 1 sample. Use larger N or switch to units='samples'."
                )
        else:  # units == 'samples'
            # Use N directly as number of samples
            buffer_size = int(N)
            if buffer_size < 1:
                raise ValueError(
                    f"Buffer size must be at least 1 sample. Received: {buffer_size}"
                )

        # --- Check if buffer already exists ---
        if self.gaze_contingent_buffer is not None:
            warnings.warn(
                "gaze_contingent_buffer already exists — initialization skipped.",
                UserWarning,
                stacklevel=2
            )
            return  # <-- exit without overwriting the existing buffer

        # --- Buffer initialization (only if not already present) ---
        self.gaze_contingent_buffer = deque(maxlen=buffer_size)


    def get_gaze_position(self, fallback_offscreen=True, method="median", coordinate_units='default'):
        """
        Get current gaze position from rolling buffer.
        
        Aggregates recent gaze samples from both eyes to provide a stable,
        real-time gaze estimate. Handles missing or invalid data gracefully.
        
        Parameters
        ----------
        fallback_offscreen : bool, optional
            If True (default), returns an offscreen position (3x screen dimensions)
            when no valid gaze data is available. If False, returns None.
        method : str, optional
            Aggregation method for combining samples and eyes:

              - **"median"** (default): Robust to outliers, good for noisy data
              - **"mean"**: Smoother but sensitive to outliers
              - **"last"**: Lowest latency, uses only most recent sample
        coordinate_units : str, optional
            Target coordinate system for returned gaze position:
            
              - **'default'**: Use window's current coordinate system
              - **'tobii'**: Tobii ADCS coordinates (0-1 range, top-left origin)
              - **PsychoPy units**: 'pix', 'height', 'norm', 'cm', 'deg'
            See [Coordinate System Conversion](#coordinate-system-conversion) for more information.
            
        Returns
        -------
        tuple or None
            Gaze position (x, y) in specified coordinates, or None if no valid
            data and fallback_offscreen=False.
            
        Raises
        ------
        RuntimeError
            If gaze_contingent() was not called to initialize the buffer.
        
        Details
        -------
        ### Coordinate System Conversion
        
        The `coordinate_units` parameter determines how gaze positions are returned from
        the real-time buffer. By default, positions are returned in your window's current
        coordinate system, making it seamless to assign gaze positions directly to stimulus
        objects. For example, if your window uses height units, the returned gaze position
        will automatically be in height units, ready to use with `stimulus.pos = gaze_pos`.
        You can override this behavior to request gaze positions in any coordinate system
        regardless of your window settings. Setting `coordinate_units='tobii'` returns the 
        raw normalized coordinates from the eye tracker, where values range from 0 to 1 with
        the origin at the top-left corner.

        Examples
        --------
        ### Basic Usage
        
        #### Default behavior (window's coordinate system)
        ```python
        pos = ET_controller.get_gaze_position()
        if pos is not None:
            circle.pos = pos
        ```
        
        ### Coordinate Systems
        
        #### Get position in Tobii coordinates (0-1 range)
        ```python
        pos = ET_controller.get_gaze_position(coordinate_units='tobii')
        # Returns: (0.5, 0.3) for gaze at center-left
        ```
        
        #### Get position in pixels (center origin)
        ```python
        pos = ET_controller.get_gaze_position(coordinate_units='pix')
        # Returns: (120, -50) for gaze slightly right and below center
        ```
        
        #### Get position in height units
        ```python
        pos = ET_controller.get_gaze_position(coordinate_units='height')
        # Returns: (0.15, -0.08) in height units
        ```
        
        ### Aggregation Methods
        
        #### Mean for smoother tracking
        ```python
        pos = ET_controller.get_gaze_position(method="mean")
        ```
        
        #### Last sample for lowest latency
        ```python
        pos = ET_controller.get_gaze_position(method="last")
        ```
        
        ### Handling Missing Data
        
        #### Return None instead of offscreen position
        ```python
        pos = ET_controller.get_gaze_position(fallback_offscreen=False)
        if pos is None:
            print("No valid gaze data")
        ```
        
        #### Check for offscreen gaze
        ```python
        pos = ET_controller.get_gaze_position(fallback_offscreen=True)
        # Offscreen positions will be far outside window bounds
        if abs(pos[0]) > 2.0 or abs(pos[1]) > 2.0:
            print("Participant looking away")
        ```
        
        ### Complete Application
        
        #### Gaze-contingent stimulus in normalized coordinates
        ```python
        ET_controller.gaze_contingent(N=0.3, units='seconds')
        ET_controller.start_recording('data.h5')
        
        circle = visual.Circle(win, radius=0.05, fillColor='red', units='norm')
        
        for frame in range(600):
            # Get gaze in normalized coordinates to match circle
            gaze_pos = ET_controller.get_gaze_position(coordinate_units='norm')
            circle.pos = gaze_pos
            circle.draw()
            win.flip()
        
        ET_controller.stop_recording()
        ```
        """
        # --- Buffer validation ---
        if self.gaze_contingent_buffer is None:
            raise RuntimeError(
                "Gaze buffer not initialized. Call gaze_contingent(N) first "
                "to set up the rolling buffer for real-time gaze processing."
            )
        
        # --- Check if buffer is empty ---
        if len(self.gaze_contingent_buffer) == 0:
            if fallback_offscreen:
                tobii_offscreen = (3.0, 3.0)
                # Convert offscreen position to target units
                if coordinate_units == 'tobii':
                    return tobii_offscreen
                elif coordinate_units == 'default':
                    return Coords.get_psychopy_pos(self.win, tobii_offscreen)
                else:
                    return Coords.get_psychopy_pos(self.win, tobii_offscreen, units=coordinate_units)
            else:
                return None
        
        # --- Convert buffer to numpy array ---
        data = np.array(list(self.gaze_contingent_buffer))  # Shape: (n_samples, 2_eyes, 2_coords)
        
        # --- Check if all data is NaN (eye tracker lost tracking) ---
        if np.all(np.isnan(data)):
            if fallback_offscreen:
                tobii_offscreen = (3.0, 3.0)
                # Convert offscreen position to target units
                if coordinate_units == 'tobii':
                    return tobii_offscreen
                elif coordinate_units == 'default':
                    return Coords.get_psychopy_pos(self.win, tobii_offscreen)
                else:
                    return Coords.get_psychopy_pos(self.win, tobii_offscreen, units=coordinate_units)
            else:
                return None
        
        # --- Validate and apply aggregation method ---
        valid_methods = {"mean", "median", "last"}
        if method not in valid_methods:
            warnings.warn(
                f"Invalid method '{method}' — defaulting to 'median'.",
                UserWarning,
                stacklevel=2
            )
            method = "median"
        
        # --- Aggregate positions ---
        if method == "mean":
            # Average across all samples and both eyes
            mean_tobii = np.nanmean(data, axis=(0, 1))
        elif method == "median":
            # Median across all samples and both eyes (robust to outliers)
            mean_tobii = np.nanmedian(data, axis=(0, 1))
        elif method == "last":
            # Use last sample only, averaged across both eyes
            mean_tobii = np.nanmean(data[-1], axis=0)
        
        # --- Convert to target coordinate system ---
        if coordinate_units == 'tobii':
            # Return raw Tobii ADCS coordinates (0-1 range)
            return tuple(mean_tobii)
        elif coordinate_units == 'default':
            # Use window's default coordinate system
            return Coords.get_psychopy_pos(self.win, mean_tobii)
        else:
            # Convert to specified coordinate system
            return Coords.get_psychopy_pos(self.win, mean_tobii, units=coordinate_units)


    def stop_live_monitor(self):
        """
        Stop the experimental live eye-position monitor if it is running.

        Safe to call when no monitor is active. This method is called
        automatically by stop_recording() and during ETracker shutdown.
        """
        if self.live_monitor is not None:
            self.live_monitor.stop()
            self.live_monitor = None


    def enable_live_monitor(self, scale=1.0, screen=None, update_rate=20):
        """
        Enable an experimental detached real-time eye-position monitor window.

        This feature is experimental: its API and platform behavior may change.
        Validate it on the intended tracker/display setup before use.

        Parameters
        ----------
        scale : float, optional
            Scale factor for the monitor window. Default is 1.0.
        screen : int, optional
            Zero-based monitor index where the monitor should appear. It uses
            the same numbering as PsychoPy's ``Window(screen=...)``. If None,
            the monitor is placed on the other screen when possible. Default
            is None.
        update_rate : int or float, optional
            Maximum monitor update rate in Hz. Default is 20.

        Warns
        -----
        UserWarning
            If the screen geometry cannot be found, the monitor window may open
            on the default screen.

        Notes
        -----
        The monitor receives data only while gaze samples are being generated,
        usually after start_recording() is called. It is automatically stopped by
        stop_recording().

        The monitor is a Tkinter window, which has no concept of monitor indices,
        so the requested ``screen`` is resolved to pixel coordinates via pyglet
        (already a PsychoPy dependency) before the window is placed. This keeps
        screen numbering consistent with the PsychoPy window.

        Examples
        --------
        ### Default monitor
        ```python
        ET_controller.enable_live_monitor()
        ET_controller.start_recording('data.h5')
        ```

        ### Custom monitor screen and refresh rate
        ```python
        ET_controller.enable_live_monitor(screen=1, update_rate=10)
        ```
        """
        if self.live_monitor is None:
            from .gui.LiveMonitor import LiveMonitor
            from .gui.TrackBox import get_screen_rect

            if screen is None:
                psychopy_screen = getattr(self.win, 'screen', 0) or 0
                screen = 1 if psychopy_screen == 0 else 0

            screen_rect = get_screen_rect(screen)
            if screen_rect is None:
                warnings.warn(
                    f"Could not determine geometry for screen {screen}; "
                    f"the live monitor may open on the default screen.",
                    UserWarning
                )

            self.live_monitor = LiveMonitor(
                scale=scale,
                screen=screen,
                screen_rect=screen_rect,
                update_rate=update_rate,
                left_eye_color=cfg.colors.left_eye,
                right_eye_color=cfg.colors.right_eye
            )

            NicePrint(
                "Live monitor enabled - gaze will appear once recording starts.",
                title="Live Monitor",
                verbose=self.verbose
            )

    # --- Internal functions ---


    def _close(self):
        """
        Clean shutdown of ETracker instance.
        
        Automatically stops any active recording session and performs
        necessary cleanup. Called automatically on program exit via atexit.
        """
        # --- Graceful shutdown ---
        # Stop recording if active (includes data saving and cleanup)
        if self.recording:
            self.stop_recording()

        self.stop_live_monitor()


    def _check_continuity(self):
        """
        Analyze complete saved data file for dropped samples.
        
        Reads all timestamps from the saved file and calculates the intervals
        between consecutive samples. Detects gaps larger than expected based
        on the eye tracker's sampling frequency, accounting for natural timing
        jitter with 50% tolerance.
        
        This method is called automatically by stop_recording() if data_check=True.
        It performs a comprehensive analysis of the ENTIRE recording session,
        including potential gaps between save_data() calls that wouldn't be
        detected by incremental checking.
        
        Uses system_time_stamp in SDK microseconds for raw format and TimeStamp
        in relative milliseconds for simplified format.
        
        Returns
        -------
        dict or None
            Dictionary containing quality metrics:
            - 'total_samples' (int): Number of saved gaze rows
            - 'dropped_samples' (int): Estimated number of dropped samples based
            on timestamp gaps
            Returns None if the file has fewer than two samples. File-reading
            errors propagate to the caller.
        
        Details
        -------
        - Only reads the timestamp column for efficiency
        - Works with both HDF5 and CSV formats
        - Handles both raw format ('system_time_stamp') and simplified format ('TimeStamp')
        - Handles both absolute timestamps (microseconds) and relative timestamps (milliseconds)
        - Dropped sample estimation: gaps larger than expected_interval * 1.5
        are flagged, and the number of missing samples is calculated based
        on gap duration
        - At 120Hz: expected interval = 8333µs (or 8.33ms), tolerance = ±50%
        - At 60Hz: expected interval = 16667µs (or 16.67ms), tolerance = ±50%
        """
        # --- Determine correct timestamp column name based on format ---
        if self.raw_format:
            timestamp_col = 'system_time_stamp'
        else:
            timestamp_col = 'TimeStamp'
        
        # --- Read timestamps from saved file ---
        if self.file_format == 'hdf5':
            with tables.open_file(self.filename, mode='r') as f:
                timestamps = f.root.gaze.col(timestamp_col)
        elif self.file_format == 'csv':
            df = pd.read_csv(self.filename, usecols=[timestamp_col])
            timestamps = df[timestamp_col].values
        
        if len(timestamps) < 2:
            return None
        
        # --- Calculate expected interval in the timestamp unit of this format ---
        # Raw keeps the SDK's native microseconds; simplified stores relative ms.
        if self.raw_format:
            expected_interval = 1_000_000.0 / self.fps  # microseconds per sample
        else:
            expected_interval = 1000.0 / self.fps       # milliseconds per sample
        
        # --- Calculate intervals between consecutive samples ---
        intervals = np.diff(timestamps)
        
        # --- Detect gaps (allow 50% tolerance for jitter) ---
        tolerance = expected_interval * 0.5
        gaps = intervals > (expected_interval + tolerance)
        
        # --- Estimate dropped samples ---
        dropped_samples = int(np.sum((intervals[gaps] / expected_interval) - 1))
        total_samples = len(timestamps)
        
        return {
            'total_samples': total_samples,
            'dropped_samples': dropped_samples
        }


    def _check_gaze_samples(self):
        """
        Wait until gaze data has caught up to the last recorded event.
        
        Ensures that the last event has corresponding gaze samples by checking
        if the most recent gaze timestamp is at or past the most recent event
        timestamp. Releases GIL during waits to allow callbacks to continue
        collecting data.
        
        Details
        -------
        - Returns immediately if no events are buffered
        - Each wait releases GIL, allowing callback thread to run
        - Typically completes in 1-2 sample intervals

        Notes
        -----
        Waits at most 100 ms, including when no gaze samples are buffered.
        Returns True when there are no pending events or gaze has caught up;
        otherwise warns and returns False when the deadline expires.
        """
        deadline = time.monotonic() + 0.100

        while True:
            with self._buf_lock:
                if not self.event_data:
                    return True

                if self.gaze_data:
                    last_event_time = self.event_data[-1]['system_time_stamp']
                    last_gaze_time = self.gaze_data[-1]['system_time_stamp']

                    if last_gaze_time >= last_event_time:
                        return True

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                warnings.warn(
                    "Gaze did not catch up with the last event within 100 ms. "
                    "Continuing shutdown/save; some events may have no "
                    "following gaze sample.",
                    UserWarning,
                )
                return False

            # Allow acquisition callbacks to run between checks.
            core.wait(min(0.001, remaining))


    def _get_info(self, moment='connection'):
        """
        Displays information about the connected eye tracker or simulation settings.

        This method prints a formatted summary of the hardware or simulation
        configuration. It can be called at different moments (e.g., at connection
        or before recording) to show relevant information. The information is
        retrieved from the eye tracker or simulation settings and cached on the
        first call to avoid repeated hardware queries.

        Parameters
        ----------
        moment : str, optional
            Specifies the context of the information display.
            - 'connection': Shows detailed information, including all available
                options (e.g., frequencies, illumination modes). This is typically
                used right after initialization.
            - 'recording': Shows a concise summary of the settings being used
                for the current recording session.
            Default is 'connection'.
        """
        if self.verbose:
            # --- Handle Simulation Mode ---
            if self.simulate:
                # Set the simulated frames per second (fps) if not already set.
                if self.fps is None:
                    self.fps = cfg.simulation_framerate

                # Display information specific to the simulation context.
                if moment == 'connection':
                    text = (
                        "Simulating eyetracker:\n"
                        f" - Simulated frequency: {self.fps} Hz"
                    )
                    title = "Simulated Eyetracker Info"
                else:  # Assumes 'recording' context
                    text = (
                        "Recording mouse position:\n"
                        f" - frequency: {self.fps} Hz"
                    )
                    title = "Recording Info"

            # --- Handle Real Eyetracker Mode ---
            else:
                # On the first call, query the eyetracker for its properties and cache them.
                # This avoids redundant SDK calls on subsequent `get_info` invocations.
                if self.fps is None:
                    self.fps = self.eyetracker.get_gaze_output_frequency()
                    self.freqs = self.eyetracker.get_all_gaze_output_frequencies()

                if self.illum_mode is None:
                    self.illum_mode = self.eyetracker.get_eye_tracking_mode()
                    self.illum_modes = self.eyetracker.get_all_eye_tracking_modes()

                # Display detailed information upon initial connection.
                if moment == 'connection':
                    text = (
                        "Connected to the eyetracker:\n"
                        f"    - Model: {self.eyetracker.model}\n"
                        f"    - Current frequency: {self.fps} Hz\n"
                        f"    - Current illumination mode: {self.illum_mode}"
                        "\nOther options:\n"
                        f"    - Possible frequencies: {self.freqs}\n"
                        f"    - Possible illumination modes: {self.illum_modes}"
                    )
                    title = "Eyetracker Info"
                else:  # Assumes 'recording' context, shows a concise summary.
                    text = (
                        "Starting recording with:\n"
                        f"    - Model: {self.eyetracker.model}\n"
                        f"    - With frequency: {self.fps} Hz\n"
                        f"    - With illumination mode: {self.illum_mode}"
                    )
                    title = "Recording Info"

            # Use the custom NicePrint utility to display the formatted information.
            NicePrint(text, title, verbose=self.verbose)


    def _subscribe_warnings(self):
        """Subscribe to critical warnings during recording."""
        self._cb_conn_lost = lambda x: self._on_warning('CONNECTION_LOST', x)
        self._cb_conn_rest = lambda x: self._on_warning('CONNECTION_RESTORED', x)
        self._cb_stream_err = lambda x: self._on_warning(
            'STREAM_BUFFER_OVERFLOW' if self.sdk_major >= 2 else 'STREAM_ERRORS', x
        )

        self.eyetracker.subscribe_to(
            tr.EYETRACKER_NOTIFICATION_CONNECTION_LOST,
            self._cb_conn_lost
        )
        self.eyetracker.subscribe_to(
            tr.EYETRACKER_NOTIFICATION_CONNECTION_RESTORED,
            self._cb_conn_rest
        )
        
        if self.sdk_major >= 2:
            self.eyetracker.subscribe_to(
                tr.EYETRACKER_NOTIFICATION_STREAM_BUFFER_OVERFLOW,
                self._cb_stream_err
            )
        else:
            self.eyetracker.subscribe_to(
                tr.EYETRACKER_STREAM_ERRORS,
                self._cb_stream_err
            )
    
    
    def _unsubscribe_warnings(self):
        """Unsubscribe from warning notifications."""
        if hasattr(self, '_cb_conn_lost'):
            self.eyetracker.unsubscribe_from(
                tr.EYETRACKER_NOTIFICATION_CONNECTION_LOST,
                self._cb_conn_lost
            )

        if hasattr(self, '_cb_conn_rest'):
            self.eyetracker.unsubscribe_from(
                tr.EYETRACKER_NOTIFICATION_CONNECTION_RESTORED,
                self._cb_conn_rest
            )

        if hasattr(self, '_cb_stream_err'):
            if self.sdk_major >= 2:
                self.eyetracker.unsubscribe_from(
                    tr.EYETRACKER_NOTIFICATION_STREAM_BUFFER_OVERFLOW,
                    self._cb_stream_err
                )
            else:
                self.eyetracker.unsubscribe_from(
                    tr.EYETRACKER_STREAM_ERRORS,
                    self._cb_stream_err
                )
    
    
    def _on_warning(self, notification_type, data):
        """
        Unified callback for all eye tracker warnings.
        
        Prints Python warning to console when eye tracker issues occur:
          - CONNECTION_LOST: Hardware disconnected during recording
          - CONNECTION_RESTORED: Hardware reconnected (data was lost)
          - STREAM_BUFFER_OVERFLOW: Data loss from processing bottleneck
        
        Parameters
        ----------
        notification_type : str
            Type of warning (CONNECTION_LOST, CONNECTION_RESTORED, etc.)
        data : dict
            Warning data including system_time_stamp
        """
        warnings.warn(
            f"Eye tracker {notification_type} at timestamp {data.system_time_stamp}",
            RuntimeWarning
        )


    def _prepare_recording(self, filename=None):
        """
        Initialize file structure with metadata and validate recording setup.
        
        Creates the output file immediately with metadata and empty typed tables
        for HDF5, or column headers for CSV. No dummy samples are inserted.
        
        For HDF5, creates file with root-level metadata including session info,
        hardware configuration, recording settings, and display configuration.
        Empty gaze and events tables are created immediately with explicit types.
        Subsequent save_data() calls append rows to these tables.
        
        For CSV, writes header row with column names based on raw_format setting.
        Data is appended in subsequent save_data() calls.
        
        Parameters
        ----------
        filename : str, optional
            Output filename with optional extension (.csv, .h5, .hdf5).
            If None, generates timestamp-based name (YYYY-MM-DD_HH-MM-SS.h5).
            Missing extensions default to .h5 format.
            
        Raises
        ------
        ValueError
            If file extension is not supported (.csv, .h5, .hdf5 only).
            
        Notes
        -----
        If the requested filename exists, a datetime suffix is appended. The
        resulting suffixed path is not checked again for a collision.
        
        HDF5 metadata is stored at root level (_v_attrs) and persists even if
        no gaze data is collected. This enables post-hoc quality assessment and
        troubleshooting of failed recording sessions.

        Raw events contain system_time_stamp and Events. Simplified events contain
        TimeStamp and Events. timestamp_units records 'us' or 'ms'; the derived
        relative_timestamps metadata is not a configurable recording argument.
        """
        # --- Filename and format determination ---
        if filename is None:
            self.filename = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.h5"
            self.file_format = 'hdf5'
        else:
            base, ext = os.path.splitext(filename)
            if not ext:
                ext = '.h5'
                filename = base + ext
                
            if ext.lower() in ('.h5', '.hdf5'):
                self.file_format = 'hdf5'
                self.filename = filename
            elif ext.lower() == '.csv':
                self.file_format = 'csv'
                self.filename = filename
            else:
                raise ValueError(
                    f"Unsupported file extension '{ext}'. "
                    f"Supported formats: .csv, .h5, .hdf5"
                )
        
        # --- File conflict prevention ---
        if os.path.exists(self.filename):
            base, ext = os.path.splitext(self.filename)
            warnings.warn(
                f"File '{self.filename}' already exists. "
                f"Attaching datetime suffix to avoid overwriting.",
                UserWarning
            )
            self.filename = f"{base}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}{ext}"
        
        # --- Create file structure with metadata ---
        if self.file_format == 'hdf5':
            # --- HDF5: Create file with root-level metadata ---
            with tables.open_file(self.filename, mode='w') as f:
                
                # === FILE-LEVEL METADATA (Session Info) ===
                f.root._v_attrs.filename = self.filename                                    # Output filename
                f.root._v_attrs.collection_date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')  # Recording timestamp
                
                # === HARDWARE INFORMATION ===
                if not self.simulate:
                    f.root._v_attrs.eyetracker_model = self.eyetracker.model              # Tobii model name
                    f.root._v_attrs.eyetracker_serial = self.eyetracker.serial_number     # Unique device ID
                    f.root._v_attrs.illumination_mode = self.illum_mode                   # Eye tracking mode
                else:
                    f.root._v_attrs.eyetracker_model = "Simulation"                       # Simulation mode indicator
                
                # === RECORDING SETTINGS ===
                f.root._v_attrs.framerate = int(self.fps)                                 # Sampling frequency (Hz)
                
                # === DISPLAY CONFIGURATION ===
                f.root._v_attrs.screen_size = str(self.win.size)                          # Screen resolution (pixels)
                f.root._v_attrs.window_units = str(self.win.units)                        # PsychoPy coordinate system
                
                # === DATA FORMAT SETTINGS ===
                f.root._v_attrs.raw_format = str(self.raw_format)                         # Full vs simplified columns
                f.root._v_attrs.coordinate_units = str(self.coordinate_units)             # Coordinate unit (simplified only; raw always ADCS)
                f.root._v_attrs.relative_timestamps = str(not self.raw_format)            # Timestamp format
                f.root._v_attrs.timestamp_units = 'us' if self.raw_format else 'ms'

                # Create typed tables without inserting dummy samples.
                if self.raw_format:
                    gaze_dtype = cfg.RawDataColumns.get_hdf5_dtype()
                else:
                    gaze_dtype = cfg.SimplifiedDataColumns.get_hdf5_dtype()

                events_dtype = cfg.EventDataColumns.get_hdf5_dtype(self.raw_format)
                f.create_table('/', 'gaze', description=gaze_dtype, title='Gaze data samples')
                f.create_table('/', 'events', description=events_dtype, title='Event markers')
        
        elif self.file_format == 'csv':
            # --- CSV: Write column headers ---
            # Determine column names based on format
            if self.raw_format:
                columns = cfg.RawDataColumns.ORDER
            else:
                columns = cfg.SimplifiedDataColumns.ORDER
            
            # Write header row
            import csv
            with open(self.filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(columns)


    def _adapt_gaze_data(self, df, df_ev):
        """
        Transform raw gaze data based on format, coordinate, and timestamp settings.
        
        Handles three main transformations:
        1. Format conversion (raw vs simplified columns)
        2. Coordinate conversion (Tobii ADCS to target units)
        3. Preserve raw SDK microseconds or produce simplified relative milliseconds
        
        In raw format, expands tuple columns into separate x, y, z components
        for tabular storage without converting display coordinates.
        Raw system, device, and event timestamps are left unchanged.
        
        In simplified format, extracts essential columns and converts coordinates
        and timestamps to specified formats.
        
        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing raw gaze data from Tobii SDK or simulation.
        df_ev : pandas.DataFrame or None
            DataFrame containing event data, or None if no events.
            
        Returns
        -------
        tuple
            (adapted_gaze_df, adapted_events_df)
            
            Raw format returns all Tobii columns with tuples expanded to x, y, z
            components. Nothing is converted: 2D coordinates stay in Tobii ADCS
            (0-1), 3D spatial coordinates retain their SDK User Coordinate
            System units, and timestamps stay in the SDK's original microseconds.

            Simplified format returns essential columns (TimeStamp, Left_X, Left_Y,
            Right_X, Right_Y, pupil diameters, validity flags, Events) with 2D
            coordinates converted to the PsychoPy convention (center origin, y up)
            in the unit chosen with coordinate_units, and timestamps standardised
            to relative milliseconds measured from the first sample.

            The event DataFrame is None if no events were supplied. In simplified
            mode it retains system_time_stamp internally and gains TimeStamp;
            the HDF5 writer stores only TimeStamp and Events.
            
        Notes
        -----
        Raw format performs no conversion at all: it only splits Tobii tuple columns
        into x, y, z components for tabular storage. 2D coordinates stay in Tobii
        ADCS (0-1), 3D spatial coordinates retain their SDK User Coordinate
        System units, and timestamps stay in the SDK's native microseconds. `coordinate_units`
        is ignored (and warned) under raw format.

        Simplified format standardises everything for analysis: 2D coordinates are
        converted to the PsychoPy convention (origin at screen center, y up) in the
        unit selected by coordinate_units, and both gaze and event timestamps are
        rewritten as integer relative milliseconds measured from the first gaze
        sample across all saves, discarding sub-millisecond precision.
        """

        if self.raw_format:
            # =====================================================================
            # RAW FORMAT: keep the data exactly as the Tobii SDK delivered it.
            # No timestamp conversion (stays in microseconds); no coordinate
            # conversion (2D stays in Tobii ADCS 0-1; 3D retains SDK UCS units).
            # Only tuple columns are split into x, y, z for tabular storage.
            # =====================================================================

            display_columns = [
                'left_gaze_point_on_display_area',
                'right_gaze_point_on_display_area'
            ]

            spatial_columns = [
                'left_gaze_point_in_user_coordinate_system',
                'left_gaze_origin_in_user_coordinate_system',
                'right_gaze_point_in_user_coordinate_system',
                'right_gaze_origin_in_user_coordinate_system',
            ]

            # 2D display coordinates: keep original Tobii ADCS (0-1) untouched
            for col in display_columns:
                coords = np.array(df[col].tolist())
                df[f'{col}_x'] = coords[:, 0]
                df[f'{col}_y'] = coords[:, 1]

            # Preserve SDK units for 3D spatial coordinates.
            for col in spatial_columns:
                coords_3d = np.array(df[col].tolist())
                df[f'{col}_x'] = coords_3d[:, 0]
                df[f'{col}_y'] = coords_3d[:, 1]
                df[f'{col}_z'] = coords_3d[:, 2]

            # Remove original tuple columns
            df = df.drop(columns=display_columns + spatial_columns)

            # Return with enforced column order
            return (df[cfg.RawDataColumns.ORDER], df_ev)

        else:
            # =====================================================================
            # SIMPLIFIED FORMAT: standardise to the PsychoPy-native, analysis-ready
            # view. 2D coordinates are centered (origin at screen center, y up) in
            # the chosen unit; timestamps become relative milliseconds from 0.
            # =====================================================================

            # --- Timestamp standardisation: relative milliseconds from 0 ---
            if self.first_timestamp is None:
                self.first_timestamp = df.iloc[0]['system_time_stamp']

            df['TimeStamp'] = ((df['system_time_stamp'] - self.first_timestamp) / 1000.0).astype(int)
            if df_ev is not None:
                df_ev['TimeStamp'] = ((df_ev['system_time_stamp'] - self.first_timestamp) / 1000.0).astype(int)

            # --- Gaze coordinates: always PsychoPy convention (center, y up) ---
            left_tobii = np.array(df['left_gaze_point_on_display_area'].tolist())
            right_tobii = np.array(df['right_gaze_point_on_display_area'].tolist())

            left_coords = Coords.get_psychopy_pos(self.win, left_tobii, units=self.coordinate_units)
            right_coords = Coords.get_psychopy_pos(self.win, right_tobii, units=self.coordinate_units)

            df['Left_X'] = left_coords[:, 0]
            df['Left_Y'] = left_coords[:, 1]
            df['Right_X'] = right_coords[:, 0]
            df['Right_Y'] = right_coords[:, 1]

            # Rename remaining columns to simplified format
            df = df.rename(columns={
                'left_gaze_point_validity': 'Left_Validity',
                'left_pupil_diameter': 'Left_Pupil',
                'left_pupil_validity': 'Left_Pupil_Validity',
                'right_gaze_point_validity': 'Right_Validity',
                'right_pupil_diameter': 'Right_Pupil',
                'right_pupil_validity': 'Right_Pupil_Validity'
            })

            # --- Data type optimization ---
            validity_dtypes = cfg.SimplifiedDataColumns.get_validity_dtypes()
            df = df.astype(validity_dtypes)

            # Return with enforced column order
            return (df[cfg.SimplifiedDataColumns.ORDER], df_ev)


    def _save_csv_data(self, gaze_df):
        """
        Save data in CSV format with append mode.
        
        Appends data to existing file. Header was already written in
        _prepare_recording().
        
        Parameters
        ----------
        gaze_df : pandas.DataFrame
            DataFrame containing gaze data with events merged in Events column.
        """
        # Always append without header (header already written)
        gaze_df.to_csv(self.filename, index=False, mode='a', header=False)


    def _save_hdf5_data(self, gaze_df, events_df):
        """
        Append gaze and event rows to the existing typed HDF5 tables.

        Parameters
        ----------
        gaze_df : pandas.DataFrame
            Adapted gaze samples, including the merged Events column.
        events_df : pandas.DataFrame or None
            Adapted event markers. None skips the event-table append.

        Notes
        -----
        Tables must have been created by _prepare_recording(). Column selection
        excludes temporary fields and orders each batch to match the stored schema.
        Raw events store system_time_stamp and Events; simplified events store
        TimeStamp and Events. Input DataFrames are not modified.

        Both batches are converted before either is appended, but the two appends
        are not transactional. Event labels are stored as UTF-8 bytes in a
        fixed-width column (HDF5_EVENT_BYTES at table creation); labels that do
        not fit are truncated with a warning. Non-ASCII text no longer raises an
        encoding error. pandas.read_hdf() reads these tables as DataFrames and
        decodes their text.
        """
        with tables.open_file(self.filename, mode='r+') as f:
            batches = [(f.root.gaze, gaze_df), (f.root.events, events_df)]
            prepared_batches = []

            # Prepare both batches before writing either; this is not transactional.
            for table, df in batches:
                if df is None:
                    continue

                # Exclude temporary columns and use the table's column order.
                df = df.loc[:, table.colnames].copy()

                # Text labels are stored as bytes. Encode UTF-8 first so any
                # character (é, 😀) is stored without an encoding error, then
                # warn if the fixed width will cut the label before storing it.
                df['Events'] = df['Events'].astype(str).str.encode('utf-8')  # text -> UTF-8 bytes (safe for é, 😀)
                if df['Events'].str.len().gt(table.dtype['Events'].itemsize).any():
                    warnings.warn(
                        f"Event label exceeds {table.dtype['Events'].itemsize} bytes "
                        f"and will be truncated.",
                        UserWarning,
                    )
                df['Events'] = df['Events'].astype(table.dtype['Events'])    # bytes -> S-column (just stores them)
                column_types = {name: table.dtype[name] for name in table.colnames}
                records = df.to_records(index=False, column_dtypes=column_types)
                prepared_batches.append((table, records))

            for table, records in prepared_batches:
                table.append(records)


    def _on_gaze_data(self, gaze_data):
        """
        Thread-safe callback for incoming eye tracker data.
        
        This method is called internally by the Tobii SDK whenever new gaze data
        is available. Stores raw gaze data in the main buffer and updates the 
        real-time gaze-contingent buffer if enabled.
        
        Parameters
        ----------
        gaze_data : dict
            Gaze sample from Tobii SDK containing timestamps, coordinates,
            validity flags, and pupil data.
        """
        live_monitor = None
        live_monitor_payload = None

        # --- Thread-safe data storage ---
        # Use lock since this is called from Tobii SDK thread
        with self._buf_lock:
            
            # --- Main recording buffer ---
            # Store complete sample for later processing and file saving
            self.gaze_data.append(gaze_data)
            
            # --- Real-time gaze-contingent buffer ---
            # Update rolling buffer for immediate gaze-contingent applications
            if self.gaze_contingent_buffer is not None:
                self.gaze_contingent_buffer.append([
                    gaze_data.get('left_gaze_point_on_display_area'),
                    gaze_data.get('right_gaze_point_on_display_area')
                ])

            if self.live_monitor is not None:
                live_monitor = self.live_monitor
                left = gaze_data.get('left_gaze_origin_in_trackbox_coordinate_system')
                right = gaze_data.get('right_gaze_origin_in_trackbox_coordinate_system')
                left_valid = gaze_data.get('left_gaze_origin_validity', 0)
                right_valid = gaze_data.get('right_gaze_origin_validity', 0)
                live_monitor_payload = (
                    left if left_valid else None,
                    right if right_valid else None
                )

        if live_monitor_payload is not None:
            live_monitor.push(*live_monitor_payload)


    def _on_user_position(self, position_data):
        """
        Store the latest user-position guide sample for participant positioning.

        Parameters
        ----------
        position_data : dict
            Tobii user-position guide sample, received with
            ``as_dictionary=True``, or its simulated equivalent. Contains
            ``left_user_position`` and ``right_user_position`` as (x, y, z)
            tuples, plus their corresponding ``*_validity`` flags.

        Notes
        -----
        Called on the Tobii SDK callback thread for real hardware or the
        background positioning thread in simulation. The shared buffer lock
        protects writes and the positioning display's reads.

        The dedicated buffer has ``maxlen=1`` because the display needs only
        the most recent position, not an accumulating sample history.
        Positioning packets must not enter recording, gaze-contingent, or
        live-monitor processing, which expects gaze samples.
        """
        with self._buf_lock:
            self._position_data.append(position_data)


    # --- Simulation Methods ---


    def _simulate_data_loop(self, data_type='gaze'):
        """
        Flexible simulation loop for different data types.
        
        Runs continuously in separate thread, generating either gaze data
        or user position data at fixed framerate. Stops when recording
        flag is cleared or stop event is set.
        
        Parameters
        ----------
        data_type : str
            Type of data to simulate: 'gaze' (for recording) or 
            'user_position' (for show_status).
        """
        # --- Timing setup ---
        interval = 1.0 / cfg.simulation_framerate
        
        try:
            # --- Main simulation loop ---
            while self.recording and not self._stop_simulation.is_set():
                # --- Data generation dispatch ---
                if data_type == 'gaze':
                    self._simulate_gaze_data()
                elif data_type == 'user_position':
                    self._simulate_user_position_guide()
                else:
                    raise ValueError(f"Unknown data_type: {data_type}")
                
                # --- Frame rate control ---
                time.sleep(interval)
                
        except Exception as e:
            # --- Error handling ---
            print(f"Simulation error: {e}")
            self._stop_simulation.set()


    def _simulate_gaze_data(self):
        """Generate single gaze sample from current mouse position."""
        try:
            scroll = self.mouse.getWheelRel()
            if scroll[1] != 0:
                current_z = getattr(self, 'sim_z_position', 0.6)
                self.sim_z_position = current_z + scroll[1] * 0.05
                self.sim_z_position = max(0.2, min(1.0, self.sim_z_position))

            pos = self.mouse.getPos()
            tobii_pos = Coords.get_tobii_pos(self.win, pos)
            tbcs_z = getattr(self, 'sim_z_position', 0.6)
            trackbox_x = 1.0 - tobii_pos[0]
            trackbox_y = tobii_pos[1]
            eye_offset = 0.035
            left_trackbox = (trackbox_x + eye_offset, trackbox_y, tbcs_z)
            right_trackbox = (trackbox_x - eye_offset, trackbox_y, tbcs_z)
            
            timestamp = int(self.experiment_clock.getTime() * 1_000_000) 
            
            # Create full Tobii-compatible structure
            gaze_data = {
                'device_time_stamp': timestamp,
                'system_time_stamp': timestamp,
                'left_gaze_point_on_display_area': tobii_pos,
                'left_gaze_point_in_user_coordinate_system': (tobii_pos[0], tobii_pos[1], tbcs_z),
                'left_gaze_point_validity': 1,
                'left_pupil_diameter': 3.0,
                'left_pupil_validity': 1,
                'left_gaze_origin_in_user_coordinate_system': (tobii_pos[0], tobii_pos[1], tbcs_z),
                'left_gaze_origin_in_trackbox_coordinate_system': left_trackbox,
                'left_gaze_origin_validity': 1,
                'right_gaze_point_on_display_area': tobii_pos,
                'right_gaze_point_in_user_coordinate_system': (tobii_pos[0], tobii_pos[1], tbcs_z),
                'right_gaze_point_validity': 1,
                'right_pupil_diameter': 3.0,
                'right_pupil_validity': 1,
                'right_gaze_origin_in_user_coordinate_system': (tobii_pos[0], tobii_pos[1], tbcs_z),
                'right_gaze_origin_in_trackbox_coordinate_system': right_trackbox,
                'right_gaze_origin_validity': 1,
                # These aren't needed for raw format but keep for show_status compatibility:
                'left_user_position': left_trackbox,
                'right_user_position': right_trackbox,
                'left_user_position_validity': 1,
                'right_user_position_validity': 1,
            }
            
            self.gaze_data.append(gaze_data)

            # --- Real-time gaze-contingent buffer ---
            # Update rolling buffer for immediate gaze-contingent applications
            if self.gaze_contingent_buffer is not None:
                self.gaze_contingent_buffer.append([
                    gaze_data.get('left_gaze_point_on_display_area'),
                    gaze_data.get('right_gaze_point_on_display_area')
                ])

            if self.live_monitor is not None:
                self.live_monitor.push(left_trackbox, right_trackbox)

        except Exception as e:
            print(f"Simulated gaze error: {e}")


    def _simulate_user_position_guide(self):
            """
            Generate user position data for track box visualization.
            
            Creates position data mimicking Tobii's user position guide,
            with realistic eye separation and interactive Z-position control
            via scroll wheel. Used specifically for show_status() display.

            Notes
            -----
            Routes samples through ``_on_user_position()`` so simulation uses
            the same isolated, latest-sample buffer as real positioning data.
            Does not append positioning samples to the gaze recording buffer.
            """
            try:
                # --- Interactive Z-position control ---
                scroll = self.mouse.getWheelRel()
                if scroll[1] != 0:  # Vertical scroll detected
                    current_z = getattr(self, 'sim_z_position', 0.6)
                    self.sim_z_position = current_z + scroll[1] * 0.05
                    self.sim_z_position = max(0.2, min(1.0, self.sim_z_position))  # Clamp range
                
                # --- Position calculation ---
                pos = self.mouse.getPos()
                
                # Get ADCS coordinates (0=Left, 1=Right)
                center_adcs_pos = Coords.get_tobii_pos(self.win, pos)
                
                # FIX: Invert X because User Position coordinates are 0=Right, 1=Left
                center_user_x = 1.0 - center_adcs_pos[0]
                center_user_y = center_adcs_pos[1]  # Y axis (0=Top) matches
                
                # --- Realistic eye separation ---
                # Simulate typical interpupillary distance (~6-7cm at 65cm distance)
                eye_offset = 0.035  # Horizontal offset in TBCS coordinates
                
                # Apply offset to the INVERTED x position
                # Note: Since 0 is Right and 1 is Left, "Left Eye" is actually +offset (closer to 1)
                # and "Right Eye" is -offset (closer to 0) from the user's perspective.
                left_user_pos = (center_user_x + eye_offset, center_user_y)
                right_user_pos = (center_user_x - eye_offset, center_user_y)
                
                # --- Data structure creation ---
                timestamp = time.time() * 1_000_000
                tbcs_z = getattr(self, 'sim_z_position', 0.6)
                
                gaze_data = {
                    'system_time_stamp': timestamp,
                    'left_user_position': (left_user_pos[0], left_user_pos[1], tbcs_z),
                    'right_user_position': (right_user_pos[0], right_user_pos[1], tbcs_z),
                    'left_user_position_validity': 1,
                    'right_user_position_validity': 1
                }
                
                # --- Position-only storage (same callback as real hardware) ---
                self._on_user_position(gaze_data)
                
            except Exception as e:
                print(f"Simulated user position error: {e}")
