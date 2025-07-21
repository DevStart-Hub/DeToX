import os
import time
import atexit
import tables
import warnings
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
# Remove the old import and import both calibration classes
from .Calibration import TobiiCalibrationSession, MouseCalibrationSession
from .Utils import NicePrint

class ETracker:
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


    _simulation_settings = {
        'framerate': 120,  # Default to Tobii Pro Spectrum rate
    }


    def __init__(self, win, id=0, simulate=False):

        self.experiment_clock = core.Clock()

        self.eyetracker_id = id
        self.win = win
        self.simulate = simulate

        self._stop_simulation = None
        self._simulation_thread = None
        self.fps = None

        self.gaze_data = deque()
        self.event_data = deque()
        self.recording = False

        # Initial timestamp
        self.first_timestamp = None

        # Gaze contingent deque
        self.gaze_contingent_buffer = None

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
            if self.fps is None:
                self.fps = self._simulation_settings['framerate']

            if moment == 'connection':
                text = (
                    "Simulating eyetracker:\n"
                    f" - Simulated frequency: {self.fps} Hz"
                )
                title = "Simulated Eyetracker Info"
            else:  # 'recording'
                text = (
                    "Recording mouse position:\n"
                    f" - frequency: {self.fps} Hz"
                )
                title = "Recording Info"
        else:
            # Cache eyetracker info on first call
            if self.fps is None:
                self.fps = self.eyetracker.get_gaze_output_frequency()
                self.freqs = self.eyetracker.get_all_gaze_output_frequencies()
                self.illum = self.eyetracker.get_illumination_mode()
                self.illums = self.eyetracker.get_all_illumination_modes()

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
            else:  # 'recording'
                text = (
                    "Starting recording with:\n"
                    f" - Model: {self.eyetracker.model}\n"
                    f" - Current frequency: {self.fps} Hz\n"
                    f" - Illumination mode: {self.illum}"
                )
                title = "Recording Info"

        NicePrint(text, title)



    def save_calibration(self, filename=None, use_gui=False):
        """
        Save calibration data to a file.
        
        This method saves the current calibration data of the eye tracker to
        the specified file. The calibration data is retrieved from the eye
        tracker using the retrieve_calibration_data() method and then written
        to the file in binary format.
        
        Parameters
        ----------
        filename : str, optional
            The name of the file to save the calibration data to. If not provided
            and use_gui=False, a default name based on timestamp will be used.
            If use_gui=True, this serves as the default filename in the dialog.
        use_gui : bool, optional
            If True, opens a file save dialog to select the save location.
            Default is False.
            
        Returns
        -------
        bool
            True if the calibration data was successfully saved, False otherwise.
            
        Raises
        ------
        RuntimeError
            If called in simulation mode.
        """
        # Check if in simulation mode first
        if self.simulate:
            raise RuntimeError(
                "Cannot save calibration in simulation mode. "
                "Calibration saving requires a real Tobii eye tracker."
            )
        
        try:
            # Prepare name
            if filename is None:   
                calibration_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_calibration.dat"
            else:
                calibration_name = filename

            # Determine the filename to save to
            if use_gui:
                from psychopy import gui
                
                # Open save dialog for calibration files
                save_path = gui.fileSaveDlg(
                    prompt='Save calibration data as…',
                    allowed='*.dat',  # Common calibration file extensions
                    initFilePath=calibration_name
                )
                
                if not save_path:
                    print("Save dialog cancelled")
                    return False
                    
                calibration_name = save_path
                print(f"Saving calibration to: {calibration_name}")
            
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


    def load_calibration(self, filename=None, use_gui=False):
        """
        Load calibration data from a file.
        
        Parameters
        ----------
        filename : str, optional
            The name of the file containing the calibration data.
            If None and use_gui=True, will open a file dialog.
            If None and use_gui=False, will raise an error.
        use_gui : bool, optional
            If True, opens a file dialog to select the calibration file.
            Default is False.
            
        Returns
        -------
        bool
            True if the calibration data was successfully loaded, False otherwise.
            
        Raises
        ------
        RuntimeError
            If called in simulation mode.
        ValueError
            If no filename provided and use_gui=False.
        """
        # Check if in simulation mode first
        if self.simulate:
            raise RuntimeError(
                "Cannot load calibration in simulation mode. "
                "Calibration loading requires a real Tobii eye tracker."
            )
        
        try:
            # Determine the filename to load
            if use_gui or filename is None:
                from psychopy import gui
                
                # Open file dialog for calibration files
                file_list = gui.fileOpenDlg(
                    prompt='Select calibration file to load…',
                    allowed='*.dat',  # Common calibration file extensions
                    tryFilePath='.'  # Start in current directory
                )
                    
                # Take the first selected file
                filename = file_list[0]
                print(f"|-- Selected calibration file: {filename} --|")
            
            elif filename is None:
                raise ValueError("No filename provided and use_gui=False")
            
            # Load the calibration file
            with open(filename, 'rb') as f:
                calib_data = f.read()
                
            # Apply the calibration data to the eye tracker
            self.eyetracker.apply_calibration_data(calib_data)
            print(f"|-- Calibration loaded from {filename} --|")
            return True
                
        except Exception as e:
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

        if self.gaze_contingent_buffer is not None:
            self.gaze_contingent_buffer.append(
                [gaze_data.get('left_gaze_point_on_display_area'),
                gaze_data.get('right_gaze_point_on_display_area')]
            )



    def _adapt_gaze_data(self, df, df_ev):
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
        left_coords = np.array(df['left_gaze_point_on_display_area'].tolist())
        right_coords = np.array(df['right_gaze_point_on_display_area'].tolist())
        
        # Convert coordinates in batch (much faster than row-by-row)
        left_psychopy = np.array([Coords.get_psychopy_pos(self.win, coord) for coord in left_coords])
        right_psychopy = np.array([Coords.get_psychopy_pos(self.win, coord) for coord in right_coords])
        
        df['Left_X'] = left_psychopy[:, 0]
        df['Left_Y'] = left_psychopy[:, 1]
        df['Right_X'] = right_psychopy[:, 0]
        df['Right_Y'] = right_psychopy[:, 1]
        
        # Remove inital timestamp
        if self.first_timestamp is None:
            self.first_timestamp = df.iloc[0]['system_time_stamp']

        # Convert microseconds to milliseconds (absolute timestamps)
        df['TimeStamp'] = ((df['system_time_stamp'] - self.first_timestamp) / 1000.0).astype(int)

        # Rename columns and convert validity columns to int8
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

        # Convert validity columns in batch
        validity_cols = ['Left_Validity', 'Left_Pupil_Validity', 'Right_Validity', 'Right_Pupil_Validity']
        df[validity_cols] = df[validity_cols].astype(np.int8)  # int8 is smaller and faster
        
        if df_ev is not None:
            df_ev['TimeStamp'] = ((df_ev['system_time_stamp'] - self.first_timestamp) / 1000.0).astype(int)
            df_ev = df_ev[['TimeStamp', 'label']].rename(columns={'label': 'Event'})

        # Return DataFrame with selected columns
        return (df[['TimeStamp', 'Left_X', 'Left_Y', 'Left_Validity',
               'Left_Pupil', 'Left_Pupil_Validity',
               'Right_X', 'Right_Y', 'Right_Validity',
               'Right_Pupil', 'Right_Pupil_Validity', 'Events']],
            df_ev)


    def save_data(self):
        """Save gaze and event data to an HDF5 file with two datasets: 'gaze' and 'events'."""

        # Start timing the save process
        start_saving = core.getTime()

        # wait 2 sample to ensure the events have their correspective sample in data
        core.wait(2/self.fps)

        # Check if there is gaze data to save
        if not self.gaze_data:
            print("|-- No new gaze data to save --|")
            return

        # Convert gaze data to a DataFrame and adapt its format
        gaze_df = pd.DataFrame(list(self.gaze_data))
        gaze_df['Events'] = pd.array([''] * len(gaze_df), dtype='string')


        # Convert event data to a DataFrame if it exists
        if len(self.event_data)>0:
            events_df = pd.DataFrame(list(self.event_data))

            #### Merge events with gaze data based on closest timestamp

            # Find where each event is in the dataframe
            idx = np.searchsorted(gaze_df['system_time_stamp'].values,
                events_df['system_time_stamp'].values,
                side='left')

            # Add events
            gaze_df.iloc[idx, gaze_df.columns.get_loc('Events')] = events_df['label'].values
        else:
            events_df = None

        #### Adapt data formats
        gaze_df, events_df = self._adapt_gaze_data(gaze_df, events_df)

        #### Save data based on format
        if self.file_format == 'csv':
            self._save_csv_data(gaze_df)
        elif self.file_format == 'hdf5':   
            self._save_hdf5_data(gaze_df, events_df)
                

        #### Pop out the samples 
        # Data
        for _ in range(len(gaze_df)):
            self.gaze_data.popleft()
        # Event
        if events_df is not None:
            for _ in range(len(events_df)):
                self.event_data.popleft()


        #### Print time taken to save data
        print(f"|-- Data saved in {round(core.getTime() - start_saving, 3)} seconds --|")


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


    def start_recording(self, filename=None):
        """
        Start recording gaze data.
        
        Parameters
        ----------
        filename : str, optional
            The name of the file to save the gaze data to. If not provided, a 
            default name based on the current datetime will be used.
        event_mode : str, optional
            Mode for event recording. Options are 'samplebased' or 'precise'. 
            Default is 'precise'.
        """
        if self.gaze_data and not self.recording:
            self.gaze_data.clear()
        elif self.recording:
            warnings.warn(
                "Recording is already in progress – start_recording() call ignored",
                UserWarning
            )
            return

        self.experiment_clock.reset() 

        # Common setup for both real and simulation modes
        self._prepare_recording(filename)
        
        if self.simulate:
            self._stop_simulation = threading.Event()

            # Use the flexible simulation loop for gaze data
            self._simulation_thread = threading.Thread(
                target=self._simulate_data_loop, 
                args=('gaze',),  # Pass 'gaze' as data type
                daemon=True
            )
            self.recording = True
            self._simulation_thread.start()
        else:
            # Real eyetracker setup
            self.eyetracker.subscribe_to(tr.EYETRACKER_GAZE_DATA, self._on_gaze_data, as_dictionary=True)
            core.wait(1)
            self.recording = True



    def stop_recording(self):
        """
        Stop recording gaze data and save all collected data to file.
        
        This method performs a complete shutdown of the recording session:
        1. Sets recording flag to False to stop data collection
        2. Cleans up simulation threads (simulation mode) or unsubscribes from 
        Tobii data streams (real eyetracker mode)
        3. Saves all buffered gaze and event data to the specified file
        4. Displays a summary of the recording session including duration 
        and file location
        
        The method will issue a warning if called when recording is not active.
        
        Warns
        -----
        UserWarning
            If recording is not currently active when this method is called.
        
        Notes
        -----
        - In simulation mode: Stops the simulation thread and cleans up resources
        - In real mode: Unsubscribes from Tobii gaze data stream
        - All pending data in buffers (gaze_data and event_data) is automatically
        saved before the method completes
        - Recording duration is measured using the experiment clock from when
        start_recording() was called
        
        Examples
        --------
        tracker = ETracker(win)
        tracker.start_recording("experiment_data.h5")
        # ... run experiment ...
        tracker.stop_recording()
        # Output: Data collection lasted approximately 45.23 seconds
        #         Data has been saved to experiment_data.h5
        
        tracker.stop_recording()  # Called again
        # Warning: Recording is not currently active - stop_recording() call ignored
        
        See Also
        --------
        start_recording : Start recording gaze data
        save_data : Save current buffer contents without stopping recording
        record_event : Record timestamped events during recording
        """
        if not self.recording:
            warnings.warn(
                "Recording is not currently active - stop_recording() call ignored",
                UserWarning
            )
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
        
        # Get recording duration and format nicely
        duration_seconds = self.experiment_clock.getTime()
        
        NicePrint(
            f'Data collection lasted approximately {duration_seconds:.2f} seconds\n'
            f'Data has been saved to {self.filename}',
            title="Recording Complete"
        )


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
            # Use simulation timestamp (milliseconds, consistent with _simulate_gaze_data)
            self.event_data.append({'system_time_stamp': self.experiment_clock.getTime() * 1_000_000, 'label': label})
        else:
            # Use eyetracker timestamp (microseconds from Tobii SDK)
            self.event_data.append({'system_time_stamp': tr.get_system_time_stamp(), 'label': label})



    def close(self):
        """
        Stop recording and perform necessary cleanup.
        """
        if self.recording:
            self.stop_recording()



    def _prepare_recording(self, filename=None):
        """
        Prepare recording by setting filename and format, using PyTables to create
        empty HDF5 tables without dummy data.

        Parameters
        ----------
        filename : str, optional
            The name of the file to save the gaze data. Can include extension (.csv, .hdf5, .h5).
            If not provided, a default name based on the current datetime will be used.

        Raises
        ------
        ValueError
            If the provided file extension is not supported.
        FileExistsError
            If the specified file already exists.
        """
        # Determine filename and format
        if filename is None:
            from datetime import datetime

            self.filename   = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.h5"
            self.file_format = 'hdf5'
        else:
            base, ext = os.path.splitext(filename)
            if not ext:
                ext = '.h5'
                filename = base + ext
            if ext.lower() in ('.h5', '.hdf5'):
                self.file_format = 'hdf5'
                self.filename   = filename
            elif ext.lower() == '.csv':
                self.file_format = 'csv'
                self.filename   = filename
            else:
                raise ValueError(f"Unsupported extension {ext}. Use .csv, .h5, or .hdf5")

        # Prevent overwriting
        if os.path.exists(self.filename):
            raise FileExistsError(f"File '{self.filename}' already exists.")

        # Create file structure
        if self.file_format == 'hdf5':
                    # Define PyTables table descriptions inside the function
            class GazeDesc(tables.IsDescription):
                TimeStamp            = tables.Int64Col()
                Left_X               = tables.Float64Col()
                Left_Y               = tables.Float64Col()
                Left_Validity        = tables.Int8Col()
                Left_Pupil           = tables.Float64Col()
                Left_Pupil_Validity  = tables.Int8Col()
                Right_X              = tables.Float64Col()
                Right_Y              = tables.Float64Col()
                Right_Validity       = tables.Int8Col()
                Right_Pupil          = tables.Float64Col()
                Right_Pupil_Validity = tables.Int8Col()
                Events               = tables.StringCol(50)

            class EventDesc(tables.IsDescription):
                system_time_stamp    = tables.Int64Col()
                label                = tables.StringCol(50)     


            with tables.open_file(self.filename, mode='w') as h5file:
                grp = h5file.create_group('/', 'data', 'Recording data')
                gaze_tbl = h5file.create_table(
                    where=grp,
                    name='gaze',
                    description=GazeDesc,
                    title='Gaze Data',
                    filters=tables.Filters(complevel=5, complib='blosc')
                )
                events_tbl = h5file.create_table(
                    where=grp,
                    name='events',
                    description=EventDesc,
                    title='Events',
                    filters=tables.Filters(complevel=5, complib='blosc')
                )
                # Attach metadata to gaze table
                gaze_tbl.attrs.subject_id  = getattr(self, 'subject_id', 'unknown')
                gaze_tbl.attrs.screen_size = tuple(self.win.size)
                gaze_tbl.attrs.framerate   = self.fps or self._simulation_settings.get('framerate')
                gaze_tbl.attrs.notes       = "Pre-created structure for optimal performance"

        else:  # CSV fallback
            cols = [
                'TimeStamp', 'Left_X', 'Left_Y', 'Left_Validity',
                'Left_Pupil', 'Left_Pupil_Validity',
                'Right_X', 'Right_Y', 'Right_Validity',
                'Right_Pupil', 'Right_Pupil_Validity', 'Events'
            ]
            pd.DataFrame(columns=cols).to_csv(self.filename, index=False)

        # Notify readiness
        self.get_info(moment="recording")




    def calibrate(self,
                    calibration_points,
                    infant_stims,
                    shuffle=True,
                    audio=None,
                    anim_type='zoom',
                    save_calib=False,
                    num_samples=5):
        """
        Run calibration procedure.
        
        Automatically selects between Tobii calibration (real mode) and 
        mouse-based calibration (simulation mode).
        
        Parameters
        ----------
        calibration_points : list of (float, float)
            PsychoPy-normalized (x, y) coordinates for calibration targets.
        infant_stims : list of str
            List of image file paths for calibration stimuli.
        shuffle : bool, optional
            Whether to shuffle stimuli order. Default is True.
        audio : psychopy.sound.Sound, optional
            Audio object to play during calibration. Default is None.
        focus_time : float, optional
            Time to wait before collecting data. Default is 0.5s
        anim_type : str, optional
            Type of animation to use. Options are 'zoom' or 'trill'. Default is 'zoom'.
        save_calib : bool, optional
            Whether to save calibration data. Default is False
        num_samples : int, optional
            Number of samples to collect per point (simulation mode only). Default is 5.

        Returns
        -------
        bool
            True if calibration successful, False otherwise
        """
        if self.simulate:
            
            session = MouseCalibrationSession(
                win=self.win,
                infant_stims=infant_stims,
                mouse=self.mouse,
                shuffle=shuffle,
                audio=audio,
                anim_type=anim_type
            )
            
            success = session.run(calibration_points, num_samples=num_samples)

            
            return success
        
        else:
            
            session = TobiiCalibrationSession(
                win=self.win,
                calibration_api=self.calibration,
                infant_stims=infant_stims,
                shuffle=shuffle,
                audio=audio,
                anim_type=anim_type
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

        # Check that the eye tracker is present or we're in simulation mode
        if not self.simulate and self.eyetracker is None:
            raise ValueError("Eyetracker not found")

        # Initialize simulation Z-position if in simulation mode
        if self.simulate:
            self.sim_z_position = 0.6  # Start at optimal distance
            print("Simulation mode: Use scroll wheel to adjust Z-position (distance from screen)")
            
            # Start continuous simulation loop for user position data
            self._stop_simulation = threading.Event()
            self._simulation_thread = threading.Thread(
                target=self._simulate_data_loop, 
                args=('user_position',),
                daemon=True
            )
            self.recording = True  # Required for simulation loop to run
            self._simulation_thread.start()

        # Subscribe to user position guide events (real eyetracker only)
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

            if self.simulate:
                # Get the latest simulated data from buffer (generated by loop)
                if self.gaze_data:
                    gaze_data = self.gaze_data[-1]
                else:
                    gaze_data = None
            else:
                # Get the latest gaze data from real eyetracker
                if self.gaze_data:
                    gaze_data = self.gaze_data[-1]
                else:
                    gaze_data = None

            if gaze_data:
                lv = gaze_data["left_user_position_validity"]
                rv = gaze_data["right_user_position_validity"]
                lx, ly, lz = gaze_data["left_user_position"]
                rx, ry, rz = gaze_data["right_user_position"]

                # Update the left eye position
                if lv:
                    # Convert TBCS coordinates to PsychoPy coordinates
                    lx_conv, ly_conv = Coords.get_psychopy_pos_from_trackbox(self.win, [lx, ly], "height")
                    leye.setPos((round(lx_conv * 0.25, 4), round(ly_conv * 0.2 + 0.4, 4)))
                    leye.draw()

                # Update the right eye position
                if rv:
                    # Convert TBCS coordinates to PsychoPy coordinates
                    rx_conv, ry_conv = Coords.get_psychopy_pos_from_trackbox(self.win, [rx, ry], "height")
                    reye.setPos((round(rx_conv * 0.25, 4), round(ry_conv * 0.2 + 0.4, 4)))
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

        # Clean the display
        self.win.flip()

        # Stop simulation loop and unsubscribe from events
        if self.simulate:
            # Stop the simulation loop
            self.recording = False
            self._stop_simulation.set()
            if self._simulation_thread.is_alive():
                self._simulation_thread.join(timeout=1.0)
        else:
            # Unsubscribe from user position guide events
            self.eyetracker.unsubscribe_from(tr.EYETRACKER_USER_POSITION_GUIDE,
                                            self._on_gaze_data)

        core.wait(0.5)


    def _simulate_data_loop(self, data_type='gaze'):
        """
        Simulate eye tracking data using mouse position at a fixed framerate.

        This method simulates either gaze data or user position data based on the
        data_type parameter. It uses time.sleep() to control the simulation rate and
        stops when either the recording is stopped or an exception occurs.
        
        Parameters
        ----------
        data_type : str
            Type of data to simulate. Options are:
            - 'gaze': Simulate gaze data (for recording)
            - 'user_position': Simulate user position guide data (for show_status)
        """
        interval = 1.0 / self._simulation_settings['framerate']
        try:
            # Loop until the recording is stopped or an exception occurs
            while self.recording and not self._stop_simulation.is_set():
                # Call the appropriate simulation method based on data type
                if data_type == 'gaze':
                    self._simulate_gaze_data()
                elif data_type == 'user_position':
                    self._simulate_user_position_guide()
                else:
                    raise ValueError(f"Unknown data_type: {data_type}")
                
                # Sleep for the specified interval
                time.sleep(interval)
        except Exception as e:
            # Print the error if something goes wrong
            print(f"Simulation error: {e}")
            # Stop the simulation loop
            self._stop_simulation.set()


    def _simulate_gaze_data(self):

        # FIIIIIIIIIIIIIIIIIIIIIIIX TIME
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
            
            # Use the interactive Z position if available, otherwise default
            tbcs_z = getattr(self, 'sim_z_position', 0.6)
            
            # Get the current timestamp in milliseconds since the Unix epoch
            timestamp = time.time() * 1_000_000  
            
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
                'left_user_position': (tobii_pos[0], tobii_pos[1], tbcs_z),  # Use same pos as gaze
                'right_user_position': (tobii_pos[0], tobii_pos[1], tbcs_z),  # Use same pos as gaze
                'left_user_position_validity': 1,  # 0 or 1 indicating validity
                'right_user_position_validity': 1  # 0 or 1 indicating validity
            }
            
            # Append the sample gaze data to the buffer
            self.gaze_data.append(gaze_data)

        except Exception as e:
            print(f"Simulated gaze error: {e}")


    def _simulate_user_position_guide(self):
        """
        Simulate user position guide data using current mouse position.
    
        This method creates simulated user position data that mimics what the
        Tobii EYETRACKER_USER_POSITION_GUIDE would provide. It uses the mouse
        position to simulate where the user's eyes would be positioned in the
        track box coordinate system (TBCS). The Z-position can be controlled
        with the scroll wheel.
    
        The data is appended to the gaze_data buffer, similar to _simulate_gaze_data().
        """
        try:
            # Handle scroll wheel for Z-position control
            scroll = self.mouse.getWheelRel()
            if scroll[1] != 0:  # Vertical scroll
                # Adjust Z-position based on scroll direction
                current_z = getattr(self, 'sim_z_position', 0.6)
                self.sim_z_position = current_z + scroll[1] * 0.05  # 0.05 units per scroll step
                # Clamp Z-position to reasonable range (0.2 to 1.0)
                self.sim_z_position = max(0.2, min(1.0, self.sim_z_position))
        
            # Get the current mouse position in PsychoPy coordinates
            pos = self.mouse.getPos()
        
            # Convert the mouse position to Tobii coordinates (ADCS)
            # We'll use this as TBCS coordinates for simulation purposes
            center_tobii_pos = Coords.get_tobii_pos(self.win, pos)
        
            # Add realistic eye separation (typical interpupillary distance ~6-7cm)
            # At 65cm distance, this translates to roughly 0.03-0.04 in TBCS coordinates
            eye_offset = 0.035  # Horizontal offset between eyes
            
            left_tobii_pos = (center_tobii_pos[0] - eye_offset, center_tobii_pos[1])
            right_tobii_pos = (center_tobii_pos[0] + eye_offset, center_tobii_pos[1])
        
            # Use the interactive Z position controlled by scroll wheel
            tbcs_z = getattr(self, 'sim_z_position', 0.6)
        
            # Get current timestamp (consistent with _simulate_gaze_data)
            timestamp = time.time() * 1_000_000
        
            # Create simulated user position guide data with separated eyes
            gaze_data = {
                'system_time_stamp': timestamp,
                'left_user_position': (left_tobii_pos[0], left_tobii_pos[1], tbcs_z),
                'right_user_position': (right_tobii_pos[0], right_tobii_pos[1], tbcs_z),
                'left_user_position_validity': 1,
                'right_user_position_validity': 1
            }
        
            # Append the sample data to the buffer
            self.gaze_data.append(gaze_data)
        
        except Exception as e:
            print(f"Simulated user position error: {e}")


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
        self.gaze_contingent_buffer = deque(maxlen=N )



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
        if self.gaze_contingent_buffer is None:
            raise RuntimeError(
                "\n[ERROR] Gaze buffer not initialized.\n"
                "You must call `gaze_contingent(N)` before using `get_average_gaze()`.\n"
                "This sets up the internal buffer for collecting recent gaze data.\n"
            )

        # Keep only [x, y] gaze points; skip empty or malformed entries
        valid_points = [p for p in self.gaze_contingent_buffer if len(p) == 2]

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