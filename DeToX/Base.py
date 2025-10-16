import os
import time
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

    # --- Core Lifecycle Methods ---

    def __init__(self, win, etracker_id=0, simulate=False):
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

        # --- Finalization ---
        # Display connection info and register the cleanup function to run on exit.
        self.get_info(moment='connection')
        atexit.register(self._close)

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


    def get_info(self, moment='connection'):
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
                self.illum = self.eyetracker.get_illumination_mode()
                self.illums = self.eyetracker.get_all_illumination_modes()

            # Display detailed information upon initial connection.
            if moment == 'connection':
                text = (
                    "Connected to the eyetracker:\n"
                    f" - Model: {self.eyetracker.model}\n"
                    f" - Current frequency: {self.fps} Hz\n"
                    f" - Illumination mode: {self.illum}\n"
                    "\nOther options:\n"
                    f" - Possible frequencies: {self.freqs}\n"
                    f" - Possible illumination modes: {self.illums}"
                )
                title = "Eyetracker Info"
            else:  # Assumes 'recording' context, shows a concise summary.
                text = (
                    "Starting recording with:\n"
                    f" - Model: {self.eyetracker.model}\n"
                    f" - Current frequency: {self.fps} Hz\n"
                    f" - Illumination mode: {self.illum}"
                )
                title = "Recording Info"

        # Use the custom NicePrint utility to display the formatted information.
        NicePrint(text, title)

    # --- Calibration Methods ---

    def show_status(self, decision_key="space"):
        """
        Real-time visualization of participant's eye position in track box.
        
        Creates interactive display showing left/right eye positions and distance
        from screen. Useful for positioning participants before data collection.
        Updates continuously until exit key is pressed.
        
        Parameters
        ----------
        decision_key : str, optional
            Key to press to exit visualization. Default 'space'.
            
        Notes
        -----
        In simulation mode, use scroll wheel to adjust simulated distance.
        Eye positions shown as green (left) and red (right) circles.
        """
        # --- Visual element creation ---
        # Create display components for track box visualization
        bgrect = visual.Rect(self.win, pos=(0, 0.4), width=0.25, height=0.2,
                            lineColor="white", fillColor="black", units="height")
        
        leye = visual.Circle(self.win, size=0.02, units="height",
                            lineColor=None, fillColor="green")  # Left eye indicator
        
        reye = visual.Circle(self.win, size=0.02, units="height", 
                            lineColor=None, fillColor="red")    # Right eye indicator
        
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
                    lx_conv, ly_conv = Coords.get_psychopy_pos_from_trackbox(self.win, [lx, ly], "height")
                    leye.setPos((round(lx_conv * 0.25, 4), round(ly_conv * 0.2 + 0.4, 4)))
                    leye.draw()
                
                # --- Draw right eye position ---
                if rv:
                    rx_conv, ry_conv = Coords.get_psychopy_pos_from_trackbox(self.win, [rx, ry], "height")
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
                calibration_points,
                infant_stims,
                shuffle=True,
                audio=None,
                anim_type='zoom',
                save_calib=False,
                num_samples=5):
        """
        Run the infant-friendly calibration procedure.

        Automatically selects the calibration method based on operating mode
        (real eye tracker vs. simulation). Uses animated stimuli and optional
        audio to engage infants during calibration.

        Parameters
        ----------
        calibration_points : list[tuple[float, float]]
            Target locations in PsychoPy coordinates (e.g., height units).
            Typically 5–9 points distributed across the screen.
        infant_stims : list[str]
            Paths to engaging image files for calibration targets
            (e.g., animated characters, colorful objects).
        shuffle : bool, optional
            Whether to randomize stimulus presentation order. Default True.
        audio : psychopy.sound.Sound | None, optional
            Attention-getting sound to play during calibration. Default None.
        anim_type : {'zoom', 'trill'}, optional
            Animation style for the stimuli. Default 'zoom'.
        save_calib : bool | str, optional
            Controls saving of calibration after a successful run:
            - False: do not save (default)
            - True: save using default naming (timestamped)
            - str: save to this filename; if it has no extension, '.dat' is added.
        num_samples : int, optional
            Samples per point in simulation mode. Default 5.

        Returns
        -------
        bool
            True if calibration completed successfully, False otherwise.

        Notes
        -----
        - Real mode uses Tobii's calibration with result visualization.
        - Simulation mode uses mouse position to approximate the process.
        - If in simulation mode, any save request is safely skipped with a warning.
        """
        # --- Mode-specific calibration setup ---
        if self.simulate:
            # Simulation calibration (mouse-based)
            session = MouseCalibrationSession(
                win=self.win,
                infant_stims=infant_stims,
                mouse=self.mouse,
                shuffle=shuffle,
                audio=audio,
                anim_type=anim_type
            )
            success = session.run(calibration_points, num_samples=num_samples)
        else:
            # Real eye tracker calibration (Tobii)
            session = TobiiCalibrationSession(
                win=self.win,
                calibration_api=self.calibration,
                infant_stims=infant_stims,
                shuffle=shuffle,
                audio=audio,
                anim_type=anim_type
            )
            success = session.run(calibration_points)

        # --- Save calibration data if requested and calibration succeeded ---
        if success and save_calib:
            if isinstance(save_calib, str):
                # Pass the provided filename (extension handled in save_calibration)
                self.save_calibration(filename=save_calib)
            else:
                # True -> no-arg save with default naming
                self.save_calibration()

        return success


    def save_calibration(self, filename=None, use_gui=False):
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

        Returns
        -------
        bool
            True if saved successfully; False if cancelled, no data available, in
            simulation mode, or on error.

        Notes
        -----
        - In simulation mode, saving is skipped and a warning is issued.
        - If `use_gui` is True and the dialog is cancelled, returns False.
        """
        # --- Simulation guard ---
        if self.simulate:
            warnings.warn(
                "Skipping calibration save: running in simulation mode. "
                "Saving requires a real Tobii eye tracker."
            )
            return False

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
                    allowed='*.dat'
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

            NicePrint(f"Calibration data saved to:\n{path}", title="Calibration Saved")
            return True

        except Exception as e:
            warnings.warn(f"Failed to save calibration data: {e}")
            return False

    def load_calibration(self, filename=None, use_gui=False):
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
        """
        # --- Pre-condition Check: Ensure not in simulation mode ---
        # Calibration can only be applied to a physical eye tracker.
        if self.simulate:
            raise RuntimeError(
                "Cannot load calibration in simulation mode. "
                "Calibration loading requires a real Tobii eye tracker."
            )
        
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
                tryFilePath=start_path
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
                      title="Calibration Loaded")
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

    def start_recording(self, filename=None):
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
            
        Raises
        -----
        UserWarning
            If recording is already in progress.
        """
        # --- State validation ---
        # Check current recording status and handle conflicts
        if self.recording:
            warnings.warn(
                "Recording is already in progress – start_recording() call ignored",
                UserWarning
            )
            return
        
        # --- Buffer initialization ---
        # Clear any residual data from previous sessions
        if self.gaze_data and not self.recording:
            self.gaze_data.clear()
        
        # --- Timing setup ---
        # Reset experiment clock for relative timestamp calculation
        self.experiment_clock.reset()
        
        # --- File preparation ---
        # Create output file structure and determine format
        self._prepare_recording(filename)
        
        # --- Data collection startup ---
        # Configure and start appropriate data collection method
        if self.simulate:
            # --- Simulation mode setup ---
            # Initialize threading controls for mouse-based simulation
            self._stop_simulation = threading.Event()
            
            # Create simulation thread for gaze data generation
            self._simulation_thread = threading.Thread(
                target=self._simulate_data_loop,
                args=('gaze',),  # Specify gaze data type for simulation
                daemon=True
            )
            
            # Activate recording and start simulation thread
            self.recording = True
            self._simulation_thread.start()
            
        else:
            # --- Real eye tracker setup ---
            # Subscribe to Tobii SDK gaze data stream
            self.eyetracker.subscribe_to(
                tr.EYETRACKER_GAZE_DATA, 
                self._on_gaze_data, 
                as_dictionary=True
            )
            
            # Allow eye tracker to stabilize before setting recording flag
            core.wait(1)
            self.recording = True

    def stop_recording(self):
        """
        Stop gaze data recording and finalize session.
        
        Performs complete shutdown: stops data collection, cleans up resources,
        saves all buffered data, and reports session summary. Handles both
        simulation and real eye tracker modes appropriately.
        
        Raises
        -----
        UserWarning
            If recording is not currently active.
            
        Notes
        -----
        All pending data in buffers is automatically saved before completion.
        Recording duration is measured from start_recording() call.
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
            # Unsubscribe from Tobii SDK data stream
            self.eyetracker.unsubscribe_from(tr.EYETRACKER_GAZE_DATA, self._on_gaze_data)
        
        # --- Data finalization ---
        # Save all remaining buffered data to file
        self.save_data()
        
        # --- Session summary ---
        # Calculate total recording duration and display results
        duration_seconds = self.experiment_clock.getTime()
        
        NicePrint(
            f'Data collection lasted approximately {duration_seconds:.2f} seconds\n'
            f'Data has been saved to {self.filename}',
            title="Recording Complete"
        )

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
            
        Examples
        --------
        tracker.record_event('trial_1_start')
        # ... present stimulus ...
        tracker.record_event('stimulus_offset')
        """
        # --- State validation ---
        # Ensure recording is active before logging events
        if not self.recording:
            raise RuntimeWarning(
                "Cannot record event: recording session is not active. "
                "Call start_recording() first to begin data collection."
            )
        
        # --- Timestamp generation ---
        # Use appropriate timing source based on recording mode
        if self.simulate:
            # --- Simulation timing ---
            # Use experiment clock for consistency with simulated gaze data
            timestamp = self.experiment_clock.getTime() * 1_000_000  # Convert to microseconds
        else:
            # --- Real eye tracker timing ---
            # Use Tobii SDK system timestamp for precise synchronization
            timestamp = tr.get_system_time_stamp()  # Already in microseconds
        
        # --- Event storage ---
        # Add timestamped event to buffer for later merging with gaze data
        self.event_data.append({
            'system_time_stamp': timestamp,
            'label': label
        })

    def save_data(self):
        """
        Save buffered gaze and event data to file with optimized processing.
        
        Uses thread-safe buffer swapping to minimize lock time, then processes
        and saves data in CSV or HDF5 format. Events are merged with gaze data
        based on timestamp proximity.
        """
        # --- Performance monitoring ---
        start_saving = core.getTime()
        
        # --- Ensure event-gaze synchronization ---
        # Wait for 2 samples to ensure events have corresponding gaze data
        core.wait(2/self.fps)
        
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
        gaze_df = pd.DataFrame(list(save_gaze))
        gaze_df['Events'] = pd.array([''] * len(gaze_df), dtype='string')
        
        # --- Event data processing and merging ---
        if event_count > 0:
            # Convert events to DataFrame
            events_df = pd.DataFrame(list(save_events))
            
            # --- Timestamp-based event merging ---
            # Find closest gaze sample for each event using binary search
            idx = np.searchsorted(gaze_df['system_time_stamp'].values,
                                events_df['system_time_stamp'].values,
                                side='left')
            
            # Merge events into gaze data at corresponding timestamps
            gaze_df.iloc[idx, gaze_df.columns.get_loc('Events')] = events_df['label'].values
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

    def gaze_contingent(self, N=5):
        """
        Initialize real-time gaze buffer for contingent applications.
        
        Sets up rolling buffer to store recent gaze coordinates for
        immediate processing during experiments. Enables smooth gaze
        estimation and real-time gaze-contingent paradigms.
        
        Parameters
        ----------
        N : int
            Number of recent gaze samples to buffer. Buffer holds
            coordinate pairs from both eyes.
            
        Raises
        ------
        TypeError
            If N is not an integer.
            
        Examples
        --------
        tracker.gaze_contingent(10)  # Buffer last 10 samples
        pos = tracker.get_average_gaze()  # Get smoothed position
        """
        # --- Input validation ---
        if not isinstance(N, int):
            raise TypeError(
                f"Invalid buffer size for gaze_contingent(): expected int, got {type(N).__name__}. "
                f"Received value: {N}"
            )
        
        # --- Buffer initialization ---
        # Store coordinate pairs [left_gaze, right_gaze] for averaging
        self.gaze_contingent_buffer = deque(maxlen=N)

    def get_average_gaze(self, fallback_offscreen=True):
        """
        Compute smoothed gaze position from recent samples.
        
        Averages valid gaze coordinates from rolling buffer to provide
        stable gaze estimates for real-time applications. Handles missing
        or invalid data gracefully.
        
        Parameters
        ----------
        fallback_offscreen : bool, optional
            Return offscreen position if no valid data available.
            Default True (returns position far outside screen bounds).
            
        Returns
        -------
        tuple or None
            Average gaze position (x, y) in Tobii ADCS coordinates,
            offscreen position, or None if no data and fallback disabled.
            
        Raises
        ------
        RuntimeError
            If gaze_contingent() was not called first to initialize buffer.
            
        Examples
        --------
        pos = tracker.get_average_gaze()
        if pos is not None:
            psychopy_pos = Coords.get_psychopy_pos(win, pos)
        """
        # --- Buffer validation ---
        if self.gaze_contingent_buffer is None:
            raise RuntimeError(
                "Gaze buffer not initialized. Call gaze_contingent(N) first "
                "to set up the rolling buffer for real-time gaze processing."
            )
        
        # --- Data extraction and validation ---
        # Filter out invalid or malformed coordinate pairs
        valid_points = [p for p in self.gaze_contingent_buffer if len(p) == 2]
        
        # --- Position calculation ---
        if not valid_points:
            # --- No valid data handling ---
            return self.win.size * 2 if fallback_offscreen else None
        else:
            # --- Average computation ---
            # Compute mean of valid coordinate pairs, handling NaN values
            return np.nanmean(valid_points, axis=0)

    # --- Private Data Processing Methods ---

    def _prepare_recording(self, filename=None):
        """
        Initialize file structure and validate recording setup.
        
        Determines output filename and format, creates empty file structure
        with proper schema, and validates no conflicts exist. Uses dummy-row
        technique for HDF5 table creation to ensure pandas compatibility.
        
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
        """
        # --- Filename and format determination ---
        # Set default timestamp-based filename or process provided name
        if filename is None:
            from datetime import datetime
            self.filename = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.h5"
            self.file_format = 'hdf5'
        else:
            # Parse filename and determine format from extension
            base, ext = os.path.splitext(filename)
            if not ext:
                ext = '.h5'  # Default to HDF5 if no extension provided
                filename = base + ext
                
            # Validate and set format based on extension
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
        # Ensure we don't accidentally overwrite existing data
        if os.path.exists(self.filename):
            raise FileExistsError(
                f"File '{self.filename}' already exists. "
                f"Choose a different filename or remove the existing file."
            )
        
        # --- File structure creation ---
        # Create empty file with proper schema based on format
        if self.file_format == 'hdf5':
            # --- HDF5 table creation using dummy-row technique ---
            # Create temporary data with correct structure and types
            dummy_gaze = pd.DataFrame({
                'TimeStamp': [-999999],
                'Left_X': [np.nan], 'Left_Y': [np.nan],
                'Left_Validity': [0], 'Left_Pupil': [np.nan],
                'Left_Pupil_Validity': [0],
                'Right_X': [np.nan], 'Right_Y': [np.nan],
                'Right_Validity': [0], 'Right_Pupil': [np.nan],
                'Right_Pupil_Validity': [0],
                'Events': ['__DUMMY__']
            })
            
            dummy_events = pd.DataFrame({
                'TimeStamp': [-999999],
                'Event': ['__DUMMY__']
            })
            
            # Create HDF5 file with proper table structure
            with pd.HDFStore(self.filename, mode='w', complevel=5, complib='blosc') as store:
                # --- Gaze table creation ---
                # Create table structure then remove dummy data
                store.append('gaze', dummy_gaze, format='table',
                            min_itemsize={'Events': 50},
                            data_columns=['TimeStamp'], index=False)
                store.remove('gaze', where='TimeStamp == -999999')
                
                # --- Events table creation ---
                # Create table structure then remove dummy data
                store.append('events', dummy_events, format='table',
                            min_itemsize={'Event': 50},
                            data_columns=['TimeStamp'], index=False)
                store.remove('events', where='TimeStamp == -999999')
                
                # --- Metadata attachment ---
                # Store experiment and system information
                gaze_attrs = store.get_storer('gaze').attrs
                gaze_attrs.subject_id = getattr(self, 'subject_id', 'unknown')
                gaze_attrs.screen_size = tuple(self.win.size)
                gaze_attrs.framerate = self.fps or cfg.simulation_framerate
                
        else:  # self.file_format == 'csv'
            # --- CSV header creation ---
            # Create empty CSV file with proper column structure
            cols = [
                'TimeStamp', 'Left_X', 'Left_Y', 'Left_Validity',
                'Left_Pupil', 'Left_Pupil_Validity',
                'Right_X', 'Right_Y', 'Right_Validity',
                'Right_Pupil', 'Right_Pupil_Validity', 'Events'
            ]
            pd.DataFrame(columns=cols).to_csv(self.filename, index=False)
        
        # --- Setup confirmation ---
        # Display recording configuration to user
        self.get_info(moment="recording")

    def _adapt_gaze_data(self, df, df_ev):
        """
        Transform raw gaze data into analysis-ready format.
        
        Converts Tobii coordinate system to PsychoPy coordinates, normalizes
        timestamps, optimizes data types, and merges event data.
        
        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing raw gaze data from Tobii SDK.
        df_ev : pandas.DataFrame or None
            DataFrame containing event data, or None if no events.
            
        Returns
        -------
        tuple of pandas.DataFrame
            (adapted_gaze_df, adapted_events_df) with converted coordinates,
            relative timestamps, and optimized data types.
        """
        # --- Coordinate conversion ---
        # Extract coordinate arrays for batch processing (faster than row-by-row)
        left_coords = np.array(df['left_gaze_point_on_display_area'].tolist())
        right_coords = np.array(df['right_gaze_point_on_display_area'].tolist())
        
        # Convert from Tobii ADCS to PsychoPy coordinate system
        left_psychopy = np.array([Coords.get_psychopy_pos(self.win, coord) for coord in left_coords])
        right_psychopy = np.array([Coords.get_psychopy_pos(self.win, coord) for coord in right_coords])
        
        # Add converted coordinates to DataFrame
        df['Left_X'] = left_psychopy[:, 0]
        df['Left_Y'] = left_psychopy[:, 1]
        df['Right_X'] = right_psychopy[:, 0]
        df['Right_Y'] = right_psychopy[:, 1]
        
        # --- Timestamp normalization ---
        # Set baseline timestamp from first sample for relative timing
        if self.first_timestamp is None:
            self.first_timestamp = df.iloc[0]['system_time_stamp']
        
        # Convert from absolute microseconds to relative milliseconds
        df['TimeStamp'] = ((df['system_time_stamp'] - self.first_timestamp) / 1000.0).astype(int)
        
        # --- Column renaming and data type optimization ---
        # Rename to standard format and convert validity flags to int8 for memory efficiency
        df = df.rename(columns={
            'left_gaze_point_validity': 'Left_Validity',
            'left_pupil_diameter': 'Left_Pupil',
            'left_pupil_validity': 'Left_Pupil_Validity',
            'right_gaze_point_validity': 'Right_Validity',
            'right_pupil_diameter': 'Right_Pupil',
            'right_pupil_validity': 'Right_Pupil_Validity'
        }).astype({
            'Left_Validity': 'int8',
            'Left_Pupil_Validity': 'int8',
            'Right_Validity': 'int8',
            'Right_Pupil_Validity': 'int8'
        })
        
        # --- Event data processing ---
        # Apply same timestamp conversion to events if they exist
        if df_ev is not None:
            df_ev['TimeStamp'] = ((df_ev['system_time_stamp'] - self.first_timestamp) / 1000.0).astype(int)
            df_ev = df_ev[['TimeStamp', 'label']].rename(columns={'label': 'Event'})
        
        # --- Return structured data ---
        # Select final columns in standardized order
        return (df[['TimeStamp', 'Left_X', 'Left_Y', 'Left_Validity',
                   'Left_Pupil', 'Left_Pupil_Validity',
                   'Right_X', 'Right_Y', 'Right_Validity',
                   'Right_Pupil', 'Right_Pupil_Validity', 'Events']],
                df_ev)

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
        """Optimized HDF5 saving without compression."""
        with pd.HDFStore(self.filename, mode="a") as store:
            
            # No compression for maximum speed
            store.append(
                "gaze", 
                gaze_df, 
                format="table", 
                append=True,
                min_itemsize={'Events': 50},  # Pre-allocate string size
                index=False  # Don't store row indices
                # No complib/complevel = no compression
            )
            
            if events_df is not None:
                store.append("events", events_df, format="table", append=True, index=False)
            
            # Add metadata only once
            if "gaze" in store and not hasattr(store.get_storer("gaze").attrs, "subject_id"):
                attrs = store.get_storer("gaze").attrs
                attrs.subject_id = getattr(self, "subject_id", "unknown")
                attrs.screen_size = tuple(self.win.size)
                attrs.framerate = self.fps

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

    # --- Private Simulation Methods ---

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

        # FIIIIIIIIIIIIIIIIIIIIIIIX TIME
        """
        Generate single gaze sample from current mouse position.
        
        Creates realistic gaze data structure matching Tobii SDK format,
        using mouse coordinates as gaze point and current experiment time
        as timestamp. Includes pupil and validity data.
        """
        try:
            # --- Position acquisition ---
            pos = self.mouse.getPos()
            tobii_pos = Coords.get_tobii_pos(self.win, pos)
            tbcs_z = getattr(self, 'sim_z_position', 0.6)
            
            # --- Timestamp generation ---
            # Use experiment clock for consistency with record_event()
            timestamp = self.experiment_clock.getTime() * 1_000_000  # Convert to microseconds
            
            # --- Gaze data structure creation ---
            gaze_data = {
                'system_time_stamp': timestamp,
                'left_gaze_point_on_display_area': tobii_pos,
                'right_gaze_point_on_display_area': tobii_pos,
                'left_gaze_point_validity': 1,
                'right_gaze_point_validity': 1,
                'left_pupil_diameter': 3.0,  # Realistic pupil size in mm
                'right_pupil_diameter': 3.0,
                'left_pupil_validity': 1,
                'right_pupil_validity': 1,
                'left_user_position': (tobii_pos[0], tobii_pos[1], tbcs_z),
                'right_user_position': (tobii_pos[0], tobii_pos[1], tbcs_z),
                'left_user_position_validity': 1,
                'right_user_position_validity': 1
            }
            
            # --- Data storage ---
            self.gaze_data.append(gaze_data)
            
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


# Example usage:
'''
from psychopy import visual, sound

# Create window
win = visual.Window(fullscr=True, units='height')

# Create controller
controller = ETracker(win)

# Define calibration points (in height units)
cal_points = [
(-0.4, 0.4), (0.0, 0.4), (0.4, 0.4),
(-0.4, 0.0), (0.0, 0.0), (0.4, 0.0),
(-0.4, -0.4), (0.0, -0.4), (0.4, -0.4)
]

# Define stimuli paths
stims = ['stims/stim1.png', 'stims/stim2.png', 'stims/stim3.png']

# Optional: add sound
audio = sound.Sound('stims/attention.wav')

# Run calibration and save data
success = controller.run_calibration(cal_points, stims, 
                                   audio=audio,
                                   save_calib=True, 
                                   calib_filename="subject1_calib.dat")

if success:
    print("Calibration successful!")
    
    # Start recording
    controller.start_recording('subject1_gaze.tsv')

    # Record events during experiment
    controller.record_event('trial_1_start')
    # Run trial 1...
    controller.record_event('trial_1_end')
    controller.save_data()  # Save and clear buffer after trial 1

    # Later, in a different session, you can load the calibration:
    # controller.load_calibration("subject1_calib.dat")

    controller.stop_recording()
else:
    print("Calibration failed!")

# Clean up
controller.close()
win.close()
'''