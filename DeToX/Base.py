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
            required for coordinate conversions.
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
        RuntimeError
            If `simulate` is False and no Tobii eye trackers can be found.
        """
        # --- Core Attributes ---
        # Store essential configuration parameters provided at initialization.
        self.win = win
        self.simulate = simulate
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
        NicePrint(summary, title="Applied Changes")


    # --- Calibration Methods ---

    def show_status(self, decision_key="space", video_help=True):
        """
        Real-time visualization of participant's eye position in track box.
        
        Creates interactive display showing left/right eye positions and distance
        from screen. Useful for positioning participants before data collection.
        Updates continuously until exit key is pressed.
        
        Optionally displays an instructional video in the background to help guide
        participant positioning. You can use the built-in video, disable the video,
        or provide your own custom MovieStim object.
        
        Parameters
        ----------
        decision_key : str, optional
            Key to press to exit visualization. Default 'space'.
        video_help : bool or visual.MovieStim, optional
            Controls background video display:
            - True: Uses built-in instructional video (default)
            - False: No video displayed
            - visual.MovieStim: Uses your pre-loaded custom video. You are
            responsible for scaling (size) and positioning (pos) the MovieStim
            to fit your desired layout.
            Default True.
            
        Details
        -------
        In simulation mode, use scroll wheel to adjust simulated distance.
        Eye positions shown as green (left) and red (right) circles.
        
        The built-in video (when video_help=True) is sized at (1.06, 0.6) in 
        height units and positioned at (0, -0.08) to avoid covering the track box.
        
        Examples
        --------
        ### Basic Usage
        
        #### Default with built-in video
        ```python
            ET_controller.show_status()
        ```
        
        #### Without background video
        ```python
            ET_controller.show_status(video_help=False)
        ```
        
        ### Customization Options
        
        #### Custom exit key
        ```python
            ET_controller.show_status(decision_key='return')
        ```
        
        #### Custom video with specific size and position
        ```python
            from psychopy import visual
            my_video = visual.MovieStim(
                win, 
                'instructions.mp4', 
                size=(0.8, 0.6), 
                pos=(0, -0.1)
            )
            ET_controller.show_status(video_help=my_video)
        ```
        
        ### Complete Workflows
        
        #### Position participant before calibration
        ```python
            # Position participant
            ET_controller.show_status()
            
            # Run calibration
            success = ET_controller.calibrate(5)
            
            # Start recording if calibration successful
            if success:
                ET_controller.start_recording('data.h5')
        ```
        """
        # --- 1. Instruction Display ---
        instructions_text = f"""Showing participant position:

    - The track box is shown as a white rectangle.
    - Both eyes are shown as colored circles.
        (try to center them within the box)
    - The green bar  on the bottom indicates distance from screen. 
        (try to have the black marker in the center of
        the green bar, around 60 cm from the screen)

    Press '{decision_key}' to finish positioning.
        """

        NicePrint(instructions_text, title="Participant Positioning", verbose=self.verbose)

        # --- Video setup (if enabled) ---
        status_movie = None

        if isinstance(video_help, visual.MovieStim):
            # video_help is already a loaded MovieStim, just use it
            status_movie = video_help
            status_movie.play()  # Start playing
            
        elif video_help:
            # video_help is True, create new MovieStim from file
            video_path = os.path.join(os.path.dirname(__file__), 'stimuli', 'ShowStatus.mp4')
            status_movie = visual.MovieStim(
                self.win, 
                video_path,
                size=(1.06, 0.6),  # Width x Height maintaining 16:9 ratio
                pos=(0, -0.08),  # Offset down so it doesn't cover track box
                loop=True,
                units='height'
            )
            status_movie.play()  # Start playing


        # --- Visual element creation ---
        # Create display components for track box visualization
        bgrect = visual.Rect(self.win, pos=(0, 0.4), width=0.25, height=0.2,
                            lineColor="white", fillColor="black", units="height")
        
        leye = visual.Circle(self.win, size=0.02, units="height",
                            lineColor=None, fillColor=cfg.colors.left_eye, colorSpace='rgb255')  # Left eye indicator
        
        reye = visual.Circle(self.win, size=0.02, units="height", 
                            lineColor=None, fillColor=cfg.colors.right_eye, colorSpace='rgb255')    # Right eye indicator
        
        # Z-position visualization elements
        zbar = visual.Rect(self.win, pos=(0, 0.28), width=0.25, height=0.03,
                          lineColor="green", fillColor="green", units="height")
        zc = visual.Rect(self.win, pos=(0, 0.28), width=0.01, height=0.03,
                        lineColor="white", fillColor="white", units="height")
        zpos = visual.Rect(self.win, pos=(0, 0.28), width=0.005, height=0.03,
                          lineColor="black", fillColor="black", units="height")
        
        # --- Hardware validation ---
        if not self.simulate and self.eyetracker is None:
            raise ValueError("Eye tracker not found and not in simulation mode")
        
        # --- Mode-specific setup ---
        if self.simulate:
            # --- Simulation initialization ---
            self.sim_z_position = 0.6  # Start at optimal distance
            print("Simulation mode: Use scroll wheel to adjust Z-position (distance from screen)")
            
            # Start position data simulation thread
            self._stop_simulation = threading.Event()
            self._simulation_thread = threading.Thread(
                target=self._simulate_data_loop, 
                args=('user_position',),
                daemon=True
            )
            self.recording = True  # Required for simulation loop
            self._simulation_thread.start()
            
        else:
            # --- Real eye tracker setup ---
            # Subscribe to user position guide data stream
            self.eyetracker.subscribe_to(tr.EYETRACKER_USER_POSITION_GUIDE,
                                        self._on_gaze_data,
                                        as_dictionary=True)
        
        # --- System stabilization ---
        core.wait(1)  # Allow data stream to stabilize
        
        # --- Main visualization loop ---
        b_show_status = True
        while b_show_status:

            # --- Draw video first ---
            if status_movie:
                status_movie.draw()

            # --- Draw static elements ---
            bgrect.draw()
            zbar.draw()
            zc.draw()
            
            # --- Get latest position data ---
            gaze_data = self.gaze_data[-1] if self.gaze_data else None
            
            if gaze_data:
                # --- Extract eye position data ---
                lv = gaze_data["left_user_position_validity"]
                rv = gaze_data["right_user_position_validity"]
                lx, ly, lz = gaze_data["left_user_position"]
                rx, ry, rz = gaze_data["right_user_position"]
                
                # --- Draw left eye position ---
                if lv:
                    lx_conv, ly_conv = Coords.get_psychopy_pos_from_user_position(self.win, [lx, ly], "height")
                    leye.setPos((round(lx_conv * 0.25, 4), round(ly_conv * 0.2 + 0.4, 4)))
                    leye.draw()
                
                # --- Draw right eye position ---
                if rv:
                    rx_conv, ry_conv = Coords.get_psychopy_pos_from_user_position(self.win, [rx, ry], "height")
                    reye.setPos((round(rx_conv * 0.25, 4), round(ry_conv * 0.2 + 0.4, 4)))
                    reye.draw()
                
                # --- Draw distance indicator ---
                if lv or rv:
                    # Calculate weighted average z-position
                    avg_z = (lz * int(lv) + rz * int(rv)) / (int(lv) + int(rv))
                    zpos.setPos((round((avg_z - 0.5) * 0.125, 4), 0.28))
                    zpos.draw()
            
            # --- Check for exit input ---
            for key in event.getKeys():
                if key == decision_key:
                    b_show_status = False
                    break
            
            self.win.flip()
        
        # --- Cleanup ---
        if status_movie:
            status_movie.stop()  # Stop video playback

        self.win.flip()  # Clear display
        
        if self.simulate:
            # --- Simulation cleanup ---
            self.recording = False
            self._stop_simulation.set()
            if self._simulation_thread.is_alive():
                self._simulation_thread.join(timeout=1.0)
        else:
            # --- Real eye tracker cleanup ---
            self.eyetracker.unsubscribe_from(tr.EYETRACKER_USER_POSITION_GUIDE,
                                            self._on_gaze_data)
        
        core.wait(0.5)  # Brief pause before return


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
            if calibration_points == 5:
                norm_points = cfg.calibration.points_5
            elif calibration_points == 9:
                norm_points = cfg.calibration.points_9
            else:
                raise ValueError(
                    f"calibration_points must be 5, 9, or a list of tuples. Got: {calibration_points}"
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
            
            norm_points = calibration_points
        else:
            raise ValueError(
                f"calibration_points must be int (5 or 9) or list of tuples. "
                f"Got: {type(calibration_points).__name__}"
            )
        
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

    def start_recording(self, filename=None, raw_format=False, coordinate_units='default', relative_timestamps='default'):
        """
        Begin gaze data recording session.
        
        Initializes file structure, clears any existing buffers, and starts
        data collection from either the eye tracker or simulation mode.
        Creates HDF5 or CSV files based on filename extension.
        
        Parameters
        ----------
        filename : str, optional
            Output filename for gaze data. If None, generates timestamp-based
            name. File extension determines format (.h5/.hdf5 for HDF5,
            .csv for CSV, defaults to .h5).
        raw_format : bool, optional
            If True, preserves all original Tobii SDK column names and data
            (including 3D eye positions, gaze origins, etc.).
            If False (default), uses simplified column names and subset of columns
            (gaze positions, pupil diameters, validity flags only).
            See [Data Format Options](#data-format-options) for more information.
        coordinate_units : str, optional
            Target coordinate system for 2D screen positions. Default is 'default':
            
            - **'default'**: Smart defaults based on format
              - raw_format=True → 'tobii' (keeps 0-1 range)
              - raw_format=False → 'pix' (PsychoPy pixels, center origin)
            
            Converts to PsychoPy coordinate systems where (0,0) is at screen center.
            Other options: 'tobii', 'height', 'norm', 'cm', 'deg'.
            See [Coordinate System Conversion](#coordinate-system-conversion) for more information.
        relative_timestamps : str or bool, optional
            Controls timestamp format. Default is 'default':
            
            - **'default'**: Smart defaults based on format
              - raw_format=True → False (absolute microseconds)
              - raw_format=False → True (relative milliseconds from 0)
            
            Other options: True (always relative), False (always absolute).
            See [Timestamp Format](#timestamp-format) for more information.
        
        Details
        -------
        ### Data Format Options
        
        The `raw_format` parameter controls which columns are included in the saved
        data file. Raw format preserves the complete data structure from the Tobii
        Pro SDK, which includes both 2D screen coordinates and 3D spatial information
        about eye position and gaze origin in the user coordinate system. This format
        is useful for advanced analyses that require the full geometric relationship
        between the eyes and the screen, or when you need to preserve all metadata
        provided by the eye tracker. The simplified format extracts only the essential
        data needed for most eye tracking analyses: gaze positions on screen, pupil
        diameters, and validity flags. This results in smaller files and easier data
        analysis for typical gaze visualization and AOI (Area of Interest) tasks.
        
        ### Coordinate System Conversion
        
        The `coordinate_units` parameter determines how 2D screen coordinates are
        represented in the output file. By default, raw format keeps coordinates in
        Tobii's Active Display Coordinate System (ADCS) where values range from 0 to 1
        with the origin at the top-left corner. This normalized system is
        screen-independent and useful for comparing data across different display
        setups. The simplified format defaults to PsychoPy pixel coordinates, which
        place the origin at the screen center with the y-axis pointing upward. This
        matches PsychoPy's coordinate system and makes it seamless to overlay gaze
        data on your experimental stimuli. You can override these defaults for either
        format. For example, you might want pixel coordinates in raw format for easier
        visualization, or keep Tobii coordinates in simplified format for cross-screen
        comparisons. When converting to other PsychoPy units like 'height' or 'norm',
        the coordinates will match your PsychoPy window's unit system. Note that in
        raw format, only the 2D display coordinates are converted; the 3D spatial
        coordinates (eye positions and gaze origins) always remain in meters as
        provided by the Tobii SDK.
        
        ### Timestamp Format
        
        The `relative_timestamps` parameter controls how time is represented in your
        data. Absolute timestamps preserve the exact microsecond values from the Tobii
        system clock, which are useful when you need to synchronize eye tracking data
        with other recording systems or when collecting multiple data files that need
        to be aligned temporally. Relative timestamps convert the first sample to time
        zero and express all subsequent times in milliseconds relative to that starting
        point, making it much easier to analyze individual trials or sessions. For
        example, with relative timestamps you can immediately see that an event occurred
        at 1500ms into your recording, whereas with absolute timestamps you would need
        to subtract the session start time first. The default behavior chooses relative
        timestamps for simplified format (since most analyses focus on within-session
        timing) and absolute timestamps for raw format (to preserve the complete
        temporal information). Both gaze samples and event markers use the same
        timestamp format to ensure proper synchronization when analyzing your data.
        
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
        
        #### Raw format with defaults (Tobii coords, absolute timestamps)
        ```python
        ET_controller.start_recording('data.h5', raw_format=True)
        # All Tobii columns, coords in 0-1 range, timestamps in microseconds
        ```
        
        #### Raw format with pixel coordinates
        ```python
        ET_controller.start_recording('data.h5', raw_format=True, coordinate_units='pix')
        # All Tobii columns, display coords in PsychoPy pixels
        ```
        
        ### Timestamp Control
        
        #### Raw format with relative timestamps
        ```python
        ET_controller.start_recording('data.h5', raw_format=True, relative_timestamps=True)
        # Raw format but timestamps start at 0ms
        ```
        
        #### Simplified format with absolute timestamps
        ```python
        ET_controller.start_recording('data.h5', relative_timestamps=False)
        # Simplified format but keeps absolute microsecond timestamps
        ```
        
        ### Coordinate Units
        
        #### Simplified format with Tobii coordinates
        ```python
        ET_controller.start_recording('data.h5', coordinate_units='tobii')
        # Simplified columns, coordinates in 0-1 range
        ```
        
        #### Simplified format with height units
        ```python
        ET_controller.start_recording('data.h5', coordinate_units='height')
        # Simplified columns, coordinates in height units matching window
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
        
        #### Multi-file synchronization with absolute timestamps
        ```python
        # Recording multiple synchronized files
        ET_controller.start_recording('session1_gaze.h5', relative_timestamps=False)
        # ... run session 1 ...
        ET_controller.stop_recording()
        
        # Session 2 can be synchronized using absolute timestamps
        ET_controller.start_recording('session2_gaze.h5', relative_timestamps=False)
        # ... run session 2 ...
        ET_controller.stop_recording()
        ```
        
        #### Advanced raw data collection
        ```python
        # Collect all data with pixel coords and relative time
        ET_controller.start_recording(
            'advanced_analysis.h5',
            raw_format=True,
            coordinate_units='pix',
            relative_timestamps=True
        )
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
        
        # --- Smart coordinate unit defaults ---
        if coordinate_units == 'default':
            self.coordinate_units = 'tobii' if raw_format else 'pix'
        else:
            # Validate coordinate_units
            valid_units = ['tobii', 'pix', 'height', 'norm', 'cm', 'deg', 'degFlat', 'degFlatPos']
            if coordinate_units not in valid_units:
                raise ValueError(
                    f"Invalid coordinate_units: '{coordinate_units}'. "
                    f"Must be one of {valid_units}"
                )
            self.coordinate_units = coordinate_units
        
        # --- Smart timestamp format defaults ---
        if relative_timestamps == 'default':
            self.relative_timestamps = False if raw_format else True
        else:
            if not isinstance(relative_timestamps, bool):
                raise TypeError(
                    f"relative_timestamps must be 'default', True, or False. "
                    f"Got: {type(relative_timestamps).__name__}"
                )
            self.relative_timestamps = relative_timestamps

        # --- Buffer initialization ---
        if self.gaze_data and not self.recording:
            self.gaze_data.clear()
        
        # --- Timing setup ---
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
        
        Performs complete shutdown: stops data collection, cleans up resources,
        saves all buffered data, and optionally performs a comprehensive data
        quality check. Handles both simulation and real eye tracker modes appropriately.
        
        Parameters
        ----------
        data_check : bool, optional
            If True (default), performs data quality check by reading the saved file
            and analyzing timestamp gaps to detect dropped samples. If False, skips
            the quality check and completes faster. Default True.
        
        Raises
        ------
        UserWarning
            If recording is not currently active.
            
        Details
        -------
        - All pending data in buffers is automatically saved before completion
        - Recording duration is measured from start_recording() call
        - Quality check reads the complete saved file to analyze gaps between ALL samples,
        including potential gaps between save_data() calls
        - At 120Hz, each sample should be ~8333µs apart; gaps significantly larger
        indicate dropped samples
        
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
        
        # --- Stop data collection ---
        # Set flag to halt data collection immediately
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
        
        NicePrint(summary, title="Recording Complete")


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
        sampling interval, they are concatenated with semicolon delimiters in the Events
        column (e.g., 'fixation_offset; stimulus_onset'). This ensures no event data is
        lost even when events occur in rapid succession.
            
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
        and ensure data is saved even if the program crashes.
        
        Details
        -------
        - Automatically called by `stop_recording()`
        - Safe to call during active recording
        - Clears buffers after saving
        - Events are matched to nearest gaze sample by timestamp
        - In HDF5 format, events are saved in two places:
          1. Merged into the main gaze table's 'Events' column
          2. As a separate 'events' table for independent event analysis
        - In CSV format, events only appear in the 'Events' column
        
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
            # Find closest gaze sample for each event using binary search
            idx = np.searchsorted(gaze_df['system_time_stamp'].values,
                                events_df['system_time_stamp'].values,
                                side='left')

            # Check if multiple events map to the same sample
            if len(idx) != len(np.unique(idx)):
                # Duplicates detected - use pandas groupby for vectorized aggregation
                events_df['_target_idx'] = idx
                grouped = events_df.groupby('_target_idx')['Events'].agg(lambda x: '; '.join(x))
                
                # Vectorized assignment to gaze_df
                gaze_df.loc[grouped.index, 'Events'] = grouped.values
            else:
                # No duplicates - fast vectorized assignment
                gaze_df.iloc[idx, gaze_df.columns.get_loc('Events')] = events_df['Events'].values
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

            
    # --- Interanl fucntions ---


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
        
        Returns
        -------
        dict or None
            Dictionary containing quality metrics:
            - 'total_samples' (int): Number of samples successfully collected
            - 'dropped_samples' (int): Estimated number of dropped samples based
            on timestamp gaps
            Returns None if file has fewer than 2 samples or cannot be read.
        
        Details
        -------
        - Only reads the timestamp column for efficiency
        - Works with both HDF5 and CSV formats
        - Dropped sample estimation: gaps larger than expected_interval * 1.5
        are flagged, and the number of missing samples is calculated based
        on gap duration
        - At 120Hz: expected interval = 8333microS, tolerance = ±4167microS
        - At 60Hz: expected interval = 16667microS, tolerance = ±8333microS
        """
        # --- Read timestamps from saved file ---
        if self.file_format == 'hdf5':
            with tables.open_file(self.filename, mode='r') as f:
                timestamps = f.root.gaze.col('system_time_stamp')
        elif self.file_format == 'csv':
            # Read only timestamp column for efficiency
            df = pd.read_csv(self.filename, usecols=['system_time_stamp'])
            timestamps = df['system_time_stamp'].values
        
        if len(timestamps) < 2:
            return None
        
        # --- Calculate quality metrics ---
        expected_interval_us = 1_000_000 / self.fps
        intervals = np.diff(timestamps)
        
        # Detect gaps (allow 50% tolerance for jitter)
        tolerance = expected_interval_us * 0.5
        gaps = intervals > (expected_interval_us + tolerance)
        
        # Estimate dropped samples
        dropped_samples = int(np.sum((intervals[gaps] / expected_interval_us) - 1))
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
        """
        if len(self.event_data) == 0:
            return  # No events to sync

        while True:

            # Thread-safe peek at last timestamps
            with self._buf_lock:
                if len(self.gaze_data) == 0:
                    break  # No gaze data yet, can't sync
                
                last_event_time = self.event_data[-1]['system_time_stamp']
                last_gaze_time = self.gaze_data[-1]['system_time_stamp']
            
            # If gaze has caught up to events, we're done
            if last_gaze_time >= last_event_time:
                break
            
            # Wait for one sample interval (releases GIL)
            core.wait(1 / self.fps)


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
        """
        Subscribe to critical warnings during recording.
        
        Monitors three types of warnings:
          - Connection lost/restored (hardware disconnection)
          - Stream buffer overflow (data loss from processing bottleneck)
        
        Automatically called by start_recording() for real eye trackers.
        Uses SDK version detection to handle API changes between versions 1.x and 2.x.
        """
        # Connection warnings (all SDK versions)
        self.eyetracker.subscribe_to(
            tr.EYETRACKER_NOTIFICATION_CONNECTION_LOST,
            lambda x: self._on_warning('CONNECTION_LOST', x)
        )
        self.eyetracker.subscribe_to(
            tr.EYETRACKER_NOTIFICATION_CONNECTION_RESTORED,
            lambda x: self._on_warning('CONNECTION_RESTORED', x)
        )
        
        # Stream buffer warnings (version dependent)
        if self.sdk_major >= 2:
            # SDK 2.0+: use new name
            self.eyetracker.subscribe_to(
                tr.EYETRACKER_NOTIFICATION_STREAM_BUFFER_OVERFLOW,
                lambda x: self._on_warning('STREAM_BUFFER_OVERFLOW', x)
            )
        else:
            # SDK 1.x: use old name
            self.eyetracker.subscribe_to(
                tr.EYETRACKER_STREAM_ERRORS,
                lambda x: self._on_warning('STREAM_ERRORS', x)
            )
    
    
    def _unsubscribe_warnings(self):
        """
        Unsubscribe from warning notifications.
        
        Cleans up warning subscriptions after recording completes.
        Automatically called by stop_recording() for real eye trackers.
        Must match the subscription pattern from _subscribe_warnings().
        """
        # Connection warnings (all SDK versions)
        self.eyetracker.unsubscribe_from(tr.EYETRACKER_NOTIFICATION_CONNECTION_LOST)
        self.eyetracker.unsubscribe_from(tr.EYETRACKER_NOTIFICATION_CONNECTION_RESTORED)
        
        # Stream buffer warnings (version dependent)
        if self.sdk_major >= 2:
            self.eyetracker.unsubscribe_from(tr.EYETRACKER_NOTIFICATION_STREAM_BUFFER_OVERFLOW)
        else:
            self.eyetracker.unsubscribe_from(tr.EYETRACKER_STREAM_ERRORS)
    
    
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
        Initialize file structure and validate recording setup.
        
        Determines output filename and format, creates empty file structure
        with proper schema based on raw_format flag. Uses dummy-row technique 
        for HDF5 table creation to ensure pandas compatibility.
        
        Parameters
        ----------
        filename : str, optional
            Output filename with optional extension (.csv, .h5, .hdf5).
            If None, generates timestamp-based name. Missing extensions
            default to .h5 format.
            
        Raises
        ------
        ValueError
            If file extension is not supported (.csv, .h5, .hdf5 only).
        FileExistsError
            If target file already exists (prevents accidental overwriting).
            
        Details
        -------
        Raw format expands tuple columns into separate x, y, z components
        for HDF5 compatibility (e.g., left_gaze_point_on_display_area becomes
        left_gaze_point_on_display_area_x and left_gaze_point_on_display_area_y).
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
            # Extract base name and extension
            base, ext = os.path.splitext(self.filename)
            
            warnings.warn(
                f"File '{self.filename}' already exists. "
                f"Attaching datetime suffix to avoid overwriting.",
                UserWarning
            )
            self.filename = f"{base}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}{ext}"
        

    def _adapt_gaze_data(self, df, df_ev):
        """
        Transform raw gaze data based on format, coordinate, and timestamp settings.
        
        Handles three main transformations:
        1. Format conversion (raw vs simplified columns)
        2. Coordinate conversion (Tobii ADCS to target units)
        3. Timestamp conversion (absolute vs relative timing)
        
        In raw format, expands tuple columns into separate x, y, z components
        for HDF5 compatibility while optionally converting display coordinates
        and timestamps.
        
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
        tuple of pandas.DataFrame
            (adapted_gaze_df, adapted_events_df)
            
            Raw format returns all Tobii columns with tuples expanded to x, y, z
            components, display coordinates converted based on coordinate_units,
            3D spatial coordinates preserved in meters, and timestamps converted
            if relative_timestamps is True.
            
            Simplified format returns essential columns (TimeStamp, Left_X, Left_Y,
            Right_X, Right_Y, pupil diameters, validity flags, Events) with
            coordinates converted based on coordinate_units and timestamps converted
            if relative_timestamps is True.
            
        Details
        -------
        Timestamp conversion transforms absolute Tobii system timestamps (microseconds
        since system epoch) into relative timestamps starting from zero and expressed
        in milliseconds. The first sample in each recording session becomes time zero,
        making it straightforward to identify when events occurred relative to the
        start of data collection. This conversion is applied consistently to both gaze
        samples and event markers to maintain temporal synchronization.
        
        Coordinate conversion in raw format affects only the 2D display coordinates
        (left_gaze_point_on_display_area and right_gaze_point_on_display_area), while
        3D spatial coordinates (left/right_gaze_point_in_user_coordinate_system and
        left/right_gaze_origin_in_user_coordinate_system) always remain in meters as
        provided by the Tobii SDK. These spatial coordinates represent the 3D position
        of the eyes and gaze vectors in the user coordinate system and are not
        meaningful to convert to screen-based units.
        """

        # --- Timestamp conversion (if enabled) ---
        if self.relative_timestamps:
            # Set reference timestamp from first sample
            if self.first_timestamp is None:
                self.first_timestamp = df.iloc[0]['system_time_stamp']
            
            # Convert to milliseconds starting from 0
            df['system_time_stamp'] = ((df['system_time_stamp'] - self.first_timestamp) / 1000.0).astype(int)
            
            # Convert event timestamps if present
            if df_ev is not None:
                df_ev['system_time_stamp'] = ((df_ev['system_time_stamp'] - self.first_timestamp) / 1000.0).astype(int)

        if self.raw_format:
            # --- Raw format processing ---
            # Preserve all Tobii SDK columns with optional coordinate conversion
            
            # Define column categories for processing
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

            # Process 2D display coordinates with optional conversion
            for col in display_columns:
                if self.coordinate_units == 'tobii':
                    # Keep original Tobii ADCS coordinates (0-1 range)
                    coords = np.array(df[col].tolist())
                else:
                    # Extract and convert to target coordinate system
                    coords_tobii = np.array(df[col].tolist())
                    coords = Coords.get_psychopy_pos(
                        self.win, 
                        coords_tobii, 
                        units=self.coordinate_units
                    )
                
                # Split into separate x and y columns
                df[f'{col}_x'] = coords[:, 0]
                df[f'{col}_y'] = coords[:, 1]

            # Process 3D spatial coordinates (always in meters, never converted)
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
            # --- Simplified format processing ---
            # Extract essential columns with optional coordinate conversion

            # Create TimeStamp column (relative or absolute based on flag)
            if self.relative_timestamps:
                # Already converted above, just rename
                df['TimeStamp'] = df['system_time_stamp']
                if df_ev is not None:
                    df_ev['TimeStamp'] = df_ev['system_time_stamp']
            else:
                # Keep absolute timestamps in microseconds
                df['TimeStamp'] = df['system_time_stamp']
                if df_ev is not None:
                    df_ev['TimeStamp'] = df_ev['system_time_stamp']
            
            # Process gaze coordinates with optional conversion
            if self.coordinate_units == 'tobii':
                # Keep original Tobii ADCS coordinates (0-1 range)
                left_coords = np.array(df['left_gaze_point_on_display_area'].tolist())
                right_coords = np.array(df['right_gaze_point_on_display_area'].tolist())
            else:
                # Extract and convert to target coordinate system
                left_tobii = np.array(df['left_gaze_point_on_display_area'].tolist())
                right_tobii = np.array(df['right_gaze_point_on_display_area'].tolist())
                
                left_coords = Coords.get_psychopy_pos(self.win, left_tobii, units=self.coordinate_units)
                right_coords = Coords.get_psychopy_pos(self.win, right_tobii, units=self.coordinate_units)

            # Add converted coordinates to dataframe
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
        
        Parameters
        ----------
        gaze_df : pandas.DataFrame
            DataFrame containing gaze data with events merged in Events column.
        events_df : pandas.DataFrame or None
            DataFrame containing raw event data (not used for CSV).
        """
        # Check if file exists to determine if we should write header
        write_header = not os.path.exists(self.filename)
        
        # Always append to file, write header only if file doesn't exist
        gaze_df.to_csv(self.filename, index=False, mode='a', header=write_header)

    def _save_hdf5_data(self, gaze_df, events_df):
        """Save gaze and event data to HDF5 using PyTables."""
        
        # Convert string columns to fixed-width bytes
        gaze_df['Events'] = gaze_df['Events'].astype('S50')
        gaze_array = gaze_df.to_records(index=False)
        
        if events_df is not None:
            events_df['Events'] = events_df['Events'].astype('S50')
            events_array = events_df.to_records(index=False)
        
        with tables.open_file(self.filename, mode='a') as f:
            # Gaze table
            if hasattr(f.root, 'gaze'):
                f.root.gaze.append(gaze_array)
            else:
                gaze_table = f.create_table(f.root, 'gaze', obj=gaze_array)
                gaze_table.attrs.screen_size = tuple(self.win.size)
                gaze_table.attrs.framerate = self.fps or cfg.simulation_framerate
                gaze_table.attrs.raw_format = self.raw_format
            
            # Events table
            if events_df is not None:
                if hasattr(f.root, 'events'):
                    f.root.events.append(events_array)
                else:
                    f.create_table(f.root, 'events', obj=events_array)


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
            pos = self.mouse.getPos()
            tobii_pos = Coords.get_tobii_pos(self.win, pos)
            tbcs_z = getattr(self, 'sim_z_position', 0.6)
            
            timestamp = int(self.experiment_clock.getTime() * 1_000_000) 
            
            # Create full Tobii-compatible structure
            gaze_data = {
                'device_time_stamp': timestamp,      # ← DEVICE FIRST
                'system_time_stamp': timestamp,      # ← SYSTEM SECOND
                'left_gaze_point_on_display_area': tobii_pos,
                'left_gaze_point_in_user_coordinate_system': (tobii_pos[0], tobii_pos[1], tbcs_z),
                'left_gaze_point_validity': 1,
                'left_pupil_diameter': 3.0,
                'left_pupil_validity': 1,
                'left_gaze_origin_in_user_coordinate_system': (tobii_pos[0], tobii_pos[1], tbcs_z),
                'left_gaze_origin_validity': 1,
                'right_gaze_point_on_display_area': tobii_pos,
                'right_gaze_point_in_user_coordinate_system': (tobii_pos[0], tobii_pos[1], tbcs_z),
                'right_gaze_point_validity': 1,
                'right_pupil_diameter': 3.0,
                'right_pupil_validity': 1,
                'right_gaze_origin_in_user_coordinate_system': (tobii_pos[0], tobii_pos[1], tbcs_z),
                'right_gaze_origin_validity': 1,
                # These aren't needed for raw format but keep for show_status compatibility:
                'left_user_position': (tobii_pos[0], tobii_pos[1], tbcs_z),
                'right_user_position': (tobii_pos[0], tobii_pos[1], tbcs_z),
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

            
        except Exception as e:
            print(f"Simulated gaze error: {e}")


    def _simulate_user_position_guide(self):
        """
        Generate user position data for track box visualization.
        
        Creates position data mimicking Tobii's user position guide,
        with realistic eye separation and interactive Z-position control
        via scroll wheel. Used specifically for show_status() display.
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
            center_tobii_pos = Coords.get_tobii_pos(self.win, pos)
            
            # --- Realistic eye separation ---
            # Simulate typical interpupillary distance (~6-7cm at 65cm distance)
            eye_offset = 0.035  # Horizontal offset in TBCS coordinates
            left_tobii_pos = (center_tobii_pos[0] - eye_offset, center_tobii_pos[1])
            right_tobii_pos = (center_tobii_pos[0] + eye_offset, center_tobii_pos[1])
            
            # --- Data structure creation ---
            timestamp = time.time() * 1_000_000
            tbcs_z = getattr(self, 'sim_z_position', 0.6)
            
            gaze_data = {
                'system_time_stamp': timestamp,
                'left_user_position': (left_tobii_pos[0], left_tobii_pos[1], tbcs_z),
                'right_user_position': (right_tobii_pos[0], right_tobii_pos[1], tbcs_z),
                'left_user_position_validity': 1,
                'right_user_position_validity': 1
            }
            
            # --- Data storage ---
            self.gaze_data.append(gaze_data)
            
        except Exception as e:
            print(f"Simulated user position error: {e}")
