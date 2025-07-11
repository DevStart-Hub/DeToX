# Standard library imports
import os
import time
import atexit
import threading
from datetime import datetime
from collections import deque

# Third party imports
import numpy as np
import pandas as pd
import tobii_research as tr
from psychopy import core, event, visual

# Local imports
from . import Coords
from .Calibration import CalibrationSession
from .Utils import NicePrint


class TobiiController:
    """
    Tobii controller for infant research.

    The TobiiController class is a simple Python wrapper around the Tobii
    Pro SDK for use in infant research. It provides convenience methods for
    starting/stopping gaze data recording and saving the data to a file.

    The TobiiController class is designed to be used with the PsychoPy
    package, which is a popular Python library for creating psychology
    experiments. It is compatible with the Tobii Pro SDK version 3.0 or
    later.

    The TobiiController class provides the following features:

        - Starting and stopping recording of gaze data
        - Saving the recorded data to a file
        - Running a calibration procedure
        - Loading a calibration from a file
        - Running a recording procedure
        - Stopping the recording procedure

    The TobiiController class is designed to be easy to use and provides a
    minimal interface for the user to interact with the Tobii Pro SDK.
    """
    
    _numkey_dict = {
        "0": -1, "num_0": -1,
        "1": 0,  "num_1": 0,
        "2": 1,  "num_2": 1,
        "3": 2,  "num_3": 2,
        "4": 3,  "num_4": 3,
        "5": 4,  "num_5": 4,
        "6": 5,  "num_6": 5,
        "7": 6,  "num_7": 6,
        "8": 7,  "num_8": 7,
        "9": 8,  "num_9": 8,
    }
 
    # Animation defaults: speed multiplier and min zoom
    _animation_settings = {
        'animation_speed': 1.0,
        'target_min': 0.2,
    }

    _simulation_settings = {
        'framerate': 120,  # Default to Tobii Pro Spectrum rate
    }


    def __init__(self, win, id=0, simulate=False):
        self.eyetracker_id = id
        self.win = win
        self.simulate = simulate

        self._stop_simulation = None
        self._simulation_thread = None

        self.gaze_data = deque()
        self.event_data = deque()
        self.recording = False

        # Configure the environment based on simulation mode
        if self.simulate:
            self.mouse = event.Mouse(win=self.win)
        else:
            eyetrackers = tr.find_all_eyetrackers()
            if not eyetrackers:
                raise RuntimeError(
                    "No Tobii eyetrackers detected.\n"
                    "Verify the connection and make sure to power on the "
                    "eyetracker before starting your computer."
                )

            self.eyetracker = eyetrackers[self.eyetracker_id]
            self.calibration = tr.ScreenBasedCalibration(self.eyetracker)

        self.get_info(moment='connection')
        atexit.register(self.close)



    def get_info(self, moment='connection'):
        """
        Print information about the current eyetracker or simulation.
        """
        if self.simulate:
            if moment == 'connection':
                text = (
                    "Simulating eyetracker:\n"
                    f" - Simulated frequency: {self._simulation_settings['framerate']} Hz"
                )
                title = "Simulated Eyetracker Info"
            else:  # 'recording'
                text = (
                    "Recording mouse position:\n"
                    f" - frequency: {self._simulation_settings['framerate']} Hz"
                )
                title = "Recording Info"
        else:
            fps = self.eyetracker.get_gaze_output_frequency()
            freqs = self.eyetracker.get_all_gaze_output_frequencies()
            illum = self.eyetracker.get_illumination_mode()
            illums = self.eyetracker.get_all_illumination_modes()

            if moment == 'connection':
                text = (
                    "Connected to the eyetracker:\n"
                    f" - Model: {self.eyetracker.model}\n"
                    f" - Current frequency: {fps} Hz\n"
                    f" - Illumination mode: {illum}\n"
                    "\nOther options:\n"
                    f" - Possible frequencies: {freqs}\n"
                    f" - Possible illumination modes: {illums}"
                )
                title = "Eyetracker Info"
            else:  # 'recording'
                text = (
                    "Starting recording with:\n"
                    f" - Model: {self.eyetracker.model}\n"
                    f" - Current frequency: {fps} Hz\n"
                    f" - Illumination mode: {illum}"
                )
                title = "Recording Info"

        NicePrint(text, title)



    def save_calibration(self, filename=None):
        """
        Save calibration data to a file.

        This method saves the current calibration data of the eye tracker to
        the specified file. The calibration data is retrieved from the eye
        tracker using the retrieve_calibration_data() method and then written
        to the file in binary format.

        Parameters
        ----------
        filename : str, optional
            The name of the file to save the calibration data to. If not provided,
            a default name based on the basename will be used.

        Returns
        -------
        bool
            True if the calibration data was successfully saved, False otherwise.
        """
        # Determine the filename to use for saving calibration data
        calibration_name = filename or f"{self.basename}_calibration.dat"

        try:
            # Retrieve calibration data from the eyetracker
            calib_data = self.eyetracker.retrieve_calibration_data()
            
            # Check if calibration data is available
            if not calib_data:
                print("No calibration data available")
                return False

            # Open the file in binary write mode and save the calibration data
            with open(calibration_name, 'wb') as f:
                f.write(calib_data)
            
            # Inform the user that the data has been successfully saved
            print(f"Calibration data saved to {calibration_name}")
            return True

        except Exception as e:
            # Handle any exceptions that occur during the saving process
            print(f"Error saving calibration: {e}")
            return False


    def load_calibration(self, filename):
        """
        Load calibration data from a file.

        Parameters
        ----------
        filename : str
            The name of the file containing the calibration data.

        Returns
        -------
        bool
            True if the calibration data was successfully loaded, False otherwise.
        """
        try:
            # Open the file in binary read mode
            with open(filename, 'rb') as f:
                # Read the calibration data from the file
                calib_data = f.read()
            # Apply the calibration data to the eye tracker
            self.eyetracker.apply_calibration_data(calib_data)
            print(f"Calibration loaded from {filename}")
            return True
        except Exception as e:
            # Handle any exceptions that occur during the loading process
            print(f"Error loading calibration: {e}")
            return False


    def _on_gaze_data(self, gaze_data):
        """
        Callback to collect gaze data.

        This method is called whenever new gaze data is available.
        It appends the received gaze data to the internal gaze data list.

        Parameters
        ----------
        gaze_data : dict
            A dictionary containing the latest gaze data.
        """
        # Append the latest gaze data to the list for later processing
        self.gaze_data.append(gaze_data)

        if self.recent_gaze_positions:
            self.recent_gaze_positions.extend(
                [gaze_data.get('left_gaze_point_on_display_area'),
                gaze_data.get('right_gaze_point_on_display_area')]
            )



    def _adapt_gaze_data(self, df):
        """
        Adapt gaze data format and convert coordinates.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing raw gaze data.

        Returns
        -------
        pandas.DataFrame
            DataFrame with adapted gaze data, including converted coordinates
            and renamed columns.
        """
        # Convert left eye gaze coordinates from Tobii to PsychoPy
        df['Left_X'], df['Left_Y'] = zip(*[
            Coords.get_psychopy_pos(self.win, coord)
            for coord in df['left_gaze_point_on_display_area']
        ])

        # Convert right eye gaze coordinates from Tobii to PsychoPy
        df['Right_X'], df['Right_Y'] = zip(*[
            Coords.get_psychopy_pos(self.win, coord)
            for coord in df['right_gaze_point_on_display_area']
        ])
        
        # Process and convert system timestamps to a unified format
        df['TimeStamp'] = self._process_timestamps(df['system_time_stamp'])
        
        # Create Events column
        df['Events'] = ''

        # Rename columns to a more readable format and convert validity columns to integers
        df = df.rename(columns={
            'left_gaze_point_validity': 'Left_Validity',
            'left_pupil_diameter': 'Left_Pupil',
            'left_pupil_validity': 'Left_Pupil_Validity',
            'right_gaze_point_validity': 'Right_Validity',
            'right_pupil_diameter': 'Right_Pupil',
            'right_pupil_validity': 'Right_Pupil_Validity'
        }).astype({
            'Left_Validity': 'int',
            'Left_Pupil_Validity': 'int', 
            'Right_Validity': 'int',
            'Right_Pupil_Validity': 'int'
        })

        # Return a DataFrame with selected columns in a specific order
        return df[['TimeStamp', 'Left_X', 'Left_Y', 'Left_Validity',
                   'Left_Pupil', 'Left_Pupil_Validity', 
                   'Right_X', 'Right_Y', 'Right_Validity',
                   'Right_Pupil', 'Right_Pupil_Validity', 'Events']]



    def save_data(self):
        """Save gaze and event data to an HDF5 file with two datasets: 'gaze' and 'events'."""

        # Start timing the save process
        start_saving = time.perf_counter()

        # Check if there is gaze data to save
        if not self.gaze_data:
            print("No gaze data to save.")
            return

        # Copy and clear buffers to prevent data loss
        gaze_data_copy = list(self.gaze_data)
        event_data_copy = list(self.event_data)

        # Convert gaze data to a DataFrame and adapt its format
        gaze_df = pd.DataFrame(gaze_data_copy) 
        gaze_df = self._adapt_gaze_data(gaze_df)

        # Convert event data to a DataFrame if it exists
        if event_data_copy:
            events_df = pd.DataFrame(event_data_copy)

            # Sort DataFrames to prepare for merging ()
            gaze_df.sort_values("TimeStamp", inplace=True)
            events_df.sort_values("TimeStamp", inplace=True)

            #### Merge events with gaze data based on closest timestamp

            # Find where each event is in the dataframe
            idx = np.searchsorted(gaze_df['system_time_stamp'].values,
                events_df['system_time_stamp'].values,
                side='left')

            #Add events
            gaze_df.loc[idx, 'Events'] = events_df['label'].values


        #### Save data based on format
        if self.file_format == 'csv':
            self._save_csv_data(gaze_df)
        elif self.file_format == 'hdf5':   
            self._save_hdf5_data(gaze_df, events_df if event_data_copy else None)
                

        #### Pop out the samples 
        # Data
        for _ in gaze_data_copy:
            self.gaze_data.popleft()
        # Event
        for _ in event_data_copy:
            self.event_data.popleft()


        #### Print time taken to save data
        print(f"Data saved in {round(time.perf_counter() - start_saving, 3)} seconds.")


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
        """
        Save data in HDF5 format with append mode.
        
        Parameters
        ----------
        gaze_df : pandas.DataFrame
            DataFrame containing gaze data with events merged in Events column.
        events_df : pandas.DataFrame or None
            DataFrame containing raw event data (saved as separate dataset).
        """
        # Always append to HDF5 file (HDF5 handles headers automatically)
        with pd.HDFStore(self.filename, mode="a") as store:
            # Append gaze data
            store.append("gaze", gaze_df, format="table", append=True)

            # Always append raw events data if they exist
            if events_df is not None:
                store.append("events", events_df, format="table", append=True)

            # Add metadata attributes if not already present (only on first write)
            if "gaze" in store and not hasattr(store.get_storer("gaze").attrs, "subject_id"):
                attrs = store.get_storer("gaze").attrs
                attrs.subject_id = getattr(self, "subject_id", "unknown")
                attrs.screen_size = tuple(self.win.size)
                attrs.framerate = self._simulation_settings["framerate"]
                attrs.notes = "Auto metadata added"



    def start_recording(self, filename=None):
        """
        Start recording gaze data.
        
        Parameters
        ----------
        filename : str, optional
            The name of the file to save the gaze data to. If not provided, a 
            default name based on the current datetime will be used.
        """
        # Common setup for both real and simulation modes
        self._prepare_recording(filename)
        
        if self.simulate:
            # Simulation-specific setup
            if self._stop_simulation is None:
                self._stop_simulation = threading.Event()
            else:
                self._stop_simulation.clear()

            self._simulation_thread = threading.Thread(target=self._simulate_gaze_data_loop, daemon=True)
            self.recording = True
            self.t0 = time.perf_counter() * 1000
            self._simulation_thread.start()
        else:
            # Real eyetracker setup
            self.eyetracker.subscribe_to(tr.EYETRACKER_GAZE_DATA, self._on_gaze_data, as_dictionary=True)
            core.wait(1)
            self.recording = True



    def stop_recording(self):
        """
        Stop recording gaze data and save it to file.
        """
        if not self.recording:
            return
        
        self.recording = False
        
        if self.simulate:
            # Simulation-specific cleanup
            if self._stop_simulation is not None:
                self._stop_simulation.set()
            if self._simulation_thread is not None:
                self._simulation_thread.join(timeout=1.0)
        else:
            # Real eyetracker cleanup
            self.eyetracker.unsubscribe_from(tr.EYETRACKER_GAZE_DATA, self._on_gaze_data)
        
        # Common code for both modes
        self.save_data()



    def record_event(self, label):
        """
        Record an event with a timestamp.
        
        Parameters
        ----------
        label : str
            The label for the event to record
        
        Raises
        ------
        RuntimeWarning
            If recording is not active
        """
        if not self.recording:
            raise RuntimeWarning("Not recording now.")
        
        if self.simulate:
            # Use simulation timestamp
            self.event_data.append({'system_time_stamp':  time.perf_counter() * 1000 , 'label': label })
        else:
            # Use eyetracker timestamp
            self.event_data.append({'system_time_stamp':  tr.get_system_time_stamp(), 'label': label})



    def close(self):
        """
        Stop recording and perform necessary cleanup.
        """
        if self.recording:
            self.stop_recording()



    def _prepare_recording(self, filename):
        """
        Prepare recording by setting filename and format.

        Parameters
        ----------
        filename : str, optional
            The name of the file to save the gaze data. Can include extension (.csv, .hdf5, .h5).
            If not provided, a default name based on the current datetime will be used.

        Raises
        ------
        ValueError
            If the provided file extension is not supported.
        """
        if filename is None:
            # Default to HDF5 format
            self.filename = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.h5"
            self.basename = self.filename[:-3]
            self.file_format = 'hdf5'
        else:
            self.basename, ext = os.path.splitext(filename)
            
            if ext:
                # Check if extension is supported
                if ext.lower() in ['.csv']:
                    self.file_format = 'csv'
                    self.filename = filename
                elif ext.lower() in ['.h5', '.hdf5']:
                    self.file_format = 'hdf5'
                    self.filename = filename
                else:
                    raise ValueError(f"Unsupported file extension: {ext}. Use .csv, .h5, or .hdf5")
            else:
                # No extension provided, default to HDF5
                self.file_format = 'hdf5'
                self.filename = f"{self.basename}.h5"

        self.get_info(moment="recording")



    def calibrate(self,
                    calibration_points,
                    infant_stims,
                    shuffle=True,
                    audio=None,
                    focus_time=0.5,
                    anim_type='zoom',
                    save_calib=False):
        """
        Run an infant-friendly calibration procedure with point selection and
        animated stimuli. The calibration points are presented in a sequence
        (either in order or shuffled) and at each point, an animated stimulus
        is presented (either zooming or trilling). The procedure can optionally
        play an attention-getting audio during the calibration process. The
        calibration data can be saved to a file if desired.

        Parameters
        ----------
        calibration_points : list of (float, float)
            PsychoPy-normalized (x, y) coordinates for calibration targets.
        infant_stims : list of str
            List of image file paths for calibration stimuli.
        shuffle : bool, optional
            Whether to shuffle stimuli order. Default is True.
        audio : str, optional
            Path to audio file to play during calibration. Default is None.
        focus_time : float, optional
            Time to wait before collecting data. Default is 0.5s
        anim_type : str, optional
            Type of animation to use. Options are 'zoom' or 'trill'. Default is 'zoom'.
        save_calib : bool, optional
            Whether to save calibration data. Default is False

        Returns
        -------
        bool
            True if calibration successful, False otherwise
        """
        # Create session, injecting our two dicts from here:
        session = CalibrationSession(
            win=self.win,
            calibration_api=self.calibration,
            infant_stims=infant_stims,
            shuffle=shuffle,
            audio=audio,
            focus_time=focus_time,
            anim_type=anim_type,
            animation_settings=self._animation_settings,
            numkey_dict=self._numkey_dict
        )
        return session.run(calibration_points, save_calib=save_calib)



    def show_status(self, decision_key="space"):
        """
        Show participant's gaze position in track box.

        This function creates a visualization of the participant's gaze
        position in the track box. The visualization consists of a green
        bar representing the z-position of the user, and circles for the
        left and right eye positions. The visualization is updated in real
        time based on the latest gaze data received from the eye tracker.

        Parameters
        ----------
        decision_key : str, optional
            The key to press to exit the visualization. Default is 'space'.
        """
        # Create visual elements
        # Background rectangle
        bgrect = visual.Rect(self.win, pos=(0, 0.4), width=0.25, height=0.2,
                            lineColor="white", fillColor="black", units="height")

        # Left eye circle
        leye = visual.Circle(self.win, size=0.02, units="height",
                            lineColor=None, fillColor="green")

        # Right eye circle
        reye = visual.Circle(self.win, size=0.02, units="height", 
                            lineColor=None, fillColor="red")

        # Z-position bar
        zbar = visual.Rect(self.win, pos=(0, 0.28), width=0.25, height=0.03,
                        lineColor="green", fillColor="green", units="height")

        # Z-position center line
        zc = visual.Rect(self.win, pos=(0, 0.28), width=0.01, height=0.03,
                        lineColor="white", fillColor="white", units="height")

        # Z-position indicator
        zpos = visual.Rect(self.win, pos=(0, 0.28), width=0.005, height=0.03,
                        lineColor="black", fillColor="black", units="height")

        # Check that the eye tracker is present
        if self.eyetracker is None and not self.simulate:  # Add check for simulation mode
            raise ValueError("Eyetracker not found")

        # Subscribe to user position guide events
        if not self.simulate:
            self.eyetracker.subscribe_to(tr.EYETRACKER_USER_POSITION_GUIDE,
                                        self._on_gaze_data,
                                        as_dictionary=True)

        # Wait for 1 second to allow the eye tracker to settle
        core.wait(1)

        # Flag to indicate whether to show the status visualization
        b_show_status = True

        # Loop until the user presses the exit key
        while b_show_status:
            # Draw the background rectangle
            bgrect.draw()

            # Draw the z-position bar and center line
            zbar.draw()
            zc.draw()

            # Get the latest gaze data
            gaze_data = self.gaze_data[-1]
            lv = gaze_data["left_user_position_validity"]
            rv = gaze_data["right_user_position_validity"]
            lx, ly, lz = gaze_data["left_user_position"]
            rx, ry, rz = gaze_data["right_user_position"]

            # Update the left eye position
            if lv:
                # Convert TBCS coordinates to PsychoPy coordinates
                lx, ly = Coords.get_psychopy_pos_from_trackbox(self.win, [lx, ly], "height")
                leye.setPos((round(lx * 0.25, 4), round(ly * 0.2 + 0.4, 4)))
                leye.draw()

            # Update the right eye position
            if rv:
                # Convert TBCS coordinates to PsychoPy coordinates
                rx, ry = Coords.get_psychopy_pos_from_trackbox(self.win, [rx, ry], "height")
                reye.setPos((round(rx * 0.25, 4), round(ry * 0.2 + 0.4, 4)))
                reye.draw()

            # Update the z-position indicator
            if lv or rv:
                # Calculate the z-position as a weighted average of left and right eye z-positions
                zpos.setPos((
                    round((((lz * int(lv) + rz * int(rv)) /
                            (int(lv) + int(rv))) - 0.5) * 0.125, 4),
                    0.28,
                ))
                zpos.draw()

            # Check for the exit key
            for key in event.getKeys():
                if key == decision_key:
                    b_show_status = False
                    break

            # Update the display
            self.win.flip()

        # Unsubscribe from user position guide events
        self.eyetracker.unsubscribe_from(tr.EYETRACKER_USER_POSITION_GUIDE,
                                        self._on_gaze_data)



    def _simulate_gaze_data_loop(self):
        """
        Simulate gaze data using mouse position at a fixed framerate.

        This method is used for simulating gaze data using the mouse position at a
        fixed framerate. It uses time.sleep() to control the simulation rate and
        stops when either the recording is stopped or an exception occurs.
        """
        interval = 1.0 / self._simulation_settings['framerate']
        try:
            # Loop until the recording is stopped or an exception occurs
            while self.recording and not self._stop_simulation.is_set():
                # Simulate a single gaze data point
                self._simulate_gaze_data()
                # Sleep for the specified interval
                time.sleep(interval)
        except Exception as e:
            # Print the error if something goes wrong
            print(f"Simulation error: {e}")
            # Stop the simulation loop
            self._stop_simulation.set()



    def _simulate_gaze_data(self):
        """
        Simulate a single gaze data point using current mouse position.
        
        This method simulates a single gaze data point using the current mouse position.
        It uses the mouse position to calculate the gaze point coordinates in Tobii ADCS
        and generates a sample gaze data dictionary with the current timestamp, left and
        right eye positions, and the user position.
        """
        try:
            # Get the current mouse position in PsychoPy coordinates
            pos = self.mouse.getPos()
            
            # Convert the mouse position to Tobii ADCS coordinates
            tobii_pos = Coords.get_tobii_pos(self.win, pos)
            
            # Get the current timestamp in milliseconds since the Unix epoch
            timestamp = time.time() * 1000  
            
            # Create a sample gaze data dictionary
            gaze_data = {
                'system_time_stamp': timestamp,  # milliseconds since Unix epoch
                'left_gaze_point_on_display_area': tobii_pos,  # Tobii ADCS coordinates
                'right_gaze_point_on_display_area': tobii_pos,  # Tobii ADCS coordinates
                'left_gaze_point_validity': 1,  # 0 or 1 indicating validity
                'right_gaze_point_validity': 1,  # 0 or 1 indicating validity
                'left_pupil_diameter': 3.0,  # mm
                'right_pupil_diameter': 3.0,  # mm
                'left_pupil_validity': 1,  # 0 or 1 indicating validity
                'right_pupil_validity': 1,  # 0 or 1 indicating validity
                'left_user_position': (0.0, 0.0, 0.6),  # Tobii ADCS coordinates
                'right_user_position': (0.0, 0.0, 0.6),  # Tobii ADCS coordinates
                'left_user_position_validity': 1,  # 0 or 1 indicating validity
                'right_user_position_validity': 1  # 0 or 1 indicating validity
            }
            
            # Append the sample gaze data to the buffer
            self.gaze_data.append(gaze_data)

        except Exception as e:
            print(f"Simulated gaze error: {e}")



    def gaze_contingent(self, N=5):
        """
        Initialize a rolling buffer to store recent gaze positions.

        This method sets up a deque (double-ended queue) to hold the last N gaze samples
        from both eyes, meaning the buffer can hold up to 2*N samples total. This is useful
        for real-time gaze contingent logic where you want to compute smooth gaze estimates
        from recent samples.

        Parameters
        ----------
        N : int
            The number of recent gaze samples (pairs of left/right eye data) to buffer.

        Raises
        ------
        TypeError
            If N is not an integer.
        """
        if not isinstance(N, int):
            raise TypeError(
                "\n[ERROR] Invalid value for `N` in gaze_contingent().\n"
                "`N` must be an integer, representing the number of recent gaze samples to buffer.\n"
                f"Received type: {type(N).__name__} (value: {N})\n"
            )
        # Store up to N samples, each consisting of two [x, y] points (left and right eye)
        self.recent_gaze_positions = deque(maxlen=N * 2)



    def get_average_gaze(self, fallback_offscreen=True):
        """
        Compute the average gaze position from the most recent gaze samples.

        This method averages valid Tobii ADCS coordinates from the rolling gaze buffer
        initialized via `gaze_contingent()`. If no valid data is available, it can return
        a fallback offscreen value to help avoid crashes or unwanted triggers in the experiment.

        Parameters
        ----------
        fallback_offscreen : bool, optional
            Whether to return an offscreen position (win.size * 2) if no valid gaze
            data is found. Default is True.

        Returns
        -------
        avg_psychopy_pos : tuple or None
            The average gaze position as a 2D coordinate in Tobii ADCS units,
            or offscreen position (tuple) / None if no data is available.

        Raises
        ------
        Warning
            If `gaze_contingent()` was not run before this function.
        """
        if not self.recent_gaze_positions:
            raise RuntimeError(
                "\n[ERROR] Gaze buffer not initialized.\n"
                "You must call `gaze_contingent(N)` before using `get_average_gaze()`.\n"
                "This sets up the internal buffer for collecting recent gaze data.\n"
            )

        # Keep only [x, y] gaze points; skip empty or malformed entries
        valid_points = [p for p in self.recent_gaze_positions if len(p) == 2]

        if not valid_points:
            # Return a dummy position far outside the screen if nothing valid is available
            avg_psychopy_pos = self.win.size * 2 if fallback_offscreen else None
        else:
            # Compute average of valid [x, y] points
            avg_psychopy_pos = np.nanmean(valid_points, axis=0)

        return avg_psychopy_pos





# Example usage:
'''
from psychopy import visual, sound

# Create window
win = visual.Window(fullscr=True, units='height')

# Create controller
controller = TobiiController(win)

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