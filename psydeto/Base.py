import atexit
import os
from datetime import datetime
import numpy as np
import pandas as pd
import tobii_research as tr
from psychopy import core, event, visual
import PIL.Image
import PIL.ImageDraw


class InfantStimuli:
    """Stimuli for infant-friendly calibration."""

    def __init__(self, win, infant_stims, shuffle=True, *kwargs):
        """
        Initialize the InfantStimuli class.

        Parameters
        ----------
        win : psychopy.visual.Window
            The PsychoPy window to render the stimuli in.
        infant_stims : list of str
            List of paths to the image files to use for the stimuli.
        shuffle : bool, optional
            Whether to shuffle the order of the stimuli. Default is True.
        *kwargs : dict
            Additional keyword arguments to be passed to the ImageStim constructor.
        """
        self.win = win
        self.stims = dict((i, visual.ImageStim(self.win, image=stim, *kwargs))
                          for i, stim in enumerate(infant_stims))
        self.stim_size = dict((i, image_stim.size) for i, image_stim in self.stims.items())
        self.present_order = [*self.stims]
        if shuffle:
            np.random.shuffle(self.present_order)

    def get_stim(self, idx):
        """
        Get the stimulus by presentation order.

        Parameters
        ----------
        idx : int
            The index of the stimulus in the presentation order.

        Returns
        -------
        psychopy.visual.ImageStim
            The stimulus corresponding to the given index.
        """
        # Calculate the index using modulo to ensure it wraps around
        stim_index = self.present_order[idx % len(self.present_order)]
        
        # Retrieve and return the stimulus by its calculated index
        return self.stims[stim_index]

    def get_stim_original_size(self, idx):
        """
        Get the original size of the stimulus by presentation order.

        Parameters
        ----------
        idx : int
            The index of the stimulus in the presentation order.

        Returns
        -------
        tuple
            The original size of the stimulus as (width, height).
        """
        # Calculate the index using modulo to ensure it wraps around
        stim_index = self.present_order[idx % len(self.present_order)]
        
        # Return the original size of the stimulus
        return self.stim_size[stim_index]


class TobiiController:
    """Simplified Tobii controller for infant research."""
    
    _default_numkey_dict = {
        "0": -1, "num_0": -1,
        "1": 0, "num_1": 0,
        "2": 1, "num_2": 1,
        "3": 2, "num_3": 2,
        "4": 3, "num_4": 3,
        "5": 4, "num_5": 4,
        "6": 5, "num_6": 5,
        "7": 6, "num_7": 6,
        "8": 7, "num_8": 7,
        "9": 8, "num_9": 8,
    }
    
    _default_calibration_dot_size = {
        "norm": 0.02,
        "height": 0.01,
        "pix": 10.0,
    }
    
    _default_calibration_disc_size = {
        "norm": 0.08,
        "height": 0.04,
        "pix": 40.0,
    }
    
    _shrink_speed = 1  # Slower for infants
    _shrink_sec = 3 / _shrink_speed
    calibration_dot_color = (0, 0, 0)
    calibration_disc_color = (-1, -1, 0)
    calibration_target_min = 0.2
    recording = False

    def __init__(self, win, id=0, filename=None, event_mode='samplebased'):
        """
        Initialize the TobiiController.

        Args:
            win: Psychopy window object
            id: ID of the Tobii eye tracker to use (default: 0)
            filename: Name of the file to save the gaze data to (default: None --> datetime)
            event_mode: How to save events ('samplebased' or 'precise'). Default is 'samplebased'
                       'samplebased': Events are matched to nearest gaze samples
                       'precise': Events are saved in a separate file with exact timestamps
        """
        self.eyetracker_id = id
        self.win = win
        self.event_mode = event_mode

        if filename:
            self.filename = filename
        else:
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            self.filename = f"{timestamp}.tsv"
            
        # For precise event mode, create events filename
        if self.event_mode == 'precise':
            base_name = os.path.splitext(self.filename)[0]
            self.events_filename = f"{base_name}_events.tsv"

        # Set the default key mappings
        self.numkey_dict = self._default_numkey_dict.copy()

        # Set the sizes of the calibration stimuli
        self.calibration_dot_size = self._default_calibration_dot_size[self.win.units]
        self.calibration_disc_size = self._default_calibration_disc_size[self.win.units]

        # Connect to the eye tracker
        eyetrackers = tr.find_all_eyetrackers()
        if len(eyetrackers) == 0:
            raise RuntimeError("No Tobii eyetrackers detected.")
        else:
            self.eyetracker = eyetrackers[self.eyetracker_id]

        # Initialize the calibration object
        self.calibration = tr.ScreenBasedCalibration(self.eyetracker)

        # Initialize data storage
        self.gaze_data = []
        self.event_data = []

        # Register the close method to be called when the program exits
        atexit.register(self.close)

    def save_calibration(self, filename):
        """
        Save calibration data to a file.

        Parameters
        ----------
        filename : str
            The name of the file to save the calibration data to.

        Returns
        -------
        bool
            True if the calibration data was successfully saved, False otherwise.
        """
        try:
            # Retrieve the calibration data from the eye tracker
            calib_data = self.eyetracker.retrieve_calibration_data()
            if calib_data:
                # Open the file in binary write mode
                with open(filename, 'wb') as f:
                    # Write the calibration data to the file
                    f.write(calib_data)
                print(f"Calibration data saved to {filename}")
                return True
            else:
                print("No calibration data available")
                return False
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

    def _show_calibration_result(self):
        """Show calibration results with lines indicating accuracy.

        This function is used to display the results of the calibration process.
        It takes the calibration result from the Tobii eye tracker and creates
        an image with green lines for the left eye and red lines for the right
        eye. Each line connects the point to the corresponding eye's position
        on the display area.

        Parameters
        ----------
        None

        Returns
        -------
        SimpleImageStim
            A Psychopy SimpleImageStim object containing the image.
        """
        img = PIL.Image.new("RGBA", tuple(self.win.size))
        img_draw = PIL.ImageDraw.Draw(img)
        result_img = visual.SimpleImageStim(self.win, img, autoLog=False)
        
        if self.calibration_result.status != tr.CALIBRATION_STATUS_FAILURE:
            for point in self.calibration_result.calibration_points:
                p = point.position_on_display_area
                
                # Draw a small circle for the calibration point
                img_draw.ellipse(
                    ((p[0] * self.win.size[0] - 3, p[1] * self.win.size[1] - 3),
                     (p[0] * self.win.size[0] + 3, p[1] * self.win.size[1] + 3)),
                    outline=(0, 0, 0, 255)
                )
                
                for sample in point.calibration_samples:
                    lp = sample.left_eye.position_on_display_area
                    rp = sample.right_eye.position_on_display_area
                    
                    # Draw a line for the left eye if the sample is valid
                    if sample.left_eye.validity == tr.VALIDITY_VALID_AND_USED:
                        img_draw.line(
                            ((p[0] * self.win.size[0], p[1] * self.win.size[1]),
                             (lp[0] * self.win.size[0], lp[1] * self.win.size[1])),
                            fill=(0, 255, 0, 255)  # Green line for left eye
                        )
                    # Draw a line for the right eye if the sample is valid
                    if sample.right_eye.validity == tr.VALIDITY_VALID_AND_USED:
                        img_draw.line(
                            ((p[0] * self.win.size[0], p[1] * self.win.size[1]),
                             (rp[0] * self.win.size[0], rp[1] * self.win.size[1])),
                            fill=(255, 0, 0, 255)  # Red line for right eye
                        )
                        
        result_img.setImage(img)
        return result_img

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

    def _process_timestamps(self, df):
        """Convert system timestamps to seconds from start.

        This function takes a DataFrame with gaze data and converts the
        system timestamps to seconds from the start of the recording.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing the gaze data.

        Returns
        -------
        pandas.DataFrame
            Modified DataFrame with the timestamps converted to seconds.
        """
        # Subtract the start time from all timestamps
        df.loc[:, 'TimeStamp'] = df['system_time_stamp'] - self.t0
        
        # Convert the result to seconds with one decimal place
        df.loc[:, 'TimeStamp'] = np.round(df['TimeStamp'] / 1000.0, 1)
        
        # Sort the DataFrame by timestamp
        return df.sort_values('TimeStamp')

    def _adapt_gaze_data(self, df):
        """Adapt gaze data format, rename columns, and process validity flags.

        This function extracts gaze coordinates, renames columns for clarity,
        and converts validity flags to integers for further processing.

        Parameters
        ----------
        df : pandas.DataFrame
            The input DataFrame containing raw gaze data.

        Returns
        -------
        pandas.DataFrame
            The adapted DataFrame with renamed columns and selected fields.
        """
        # Extract gaze coordinates from display area arrays
        df['Left_X'] = df['left_gaze_point_on_display_area'].str[0]
        df['Left_Y'] = df['left_gaze_point_on_display_area'].str[1]
        df['Right_X'] = df['right_gaze_point_on_display_area'].str[0]
        df['Right_Y'] = df['right_gaze_point_on_display_area'].str[1]
        
        # Define a mapping for column renaming to improve readability
        column_mapping = {
            'left_gaze_point_validity': 'Left_Validity',
            'left_pupil_diameter': 'Left_Pupil',
            'left_pupil_validity': 'Left_Pupil_Validity',
            'right_gaze_point_validity': 'Right_Validity',
            'right_pupil_diameter': 'Right_Pupil',
            'right_pupil_validity': 'Right_Pupil_Validity'
        }
        
        # Rename columns using the defined mapping
        df = df.rename(columns=column_mapping)
        
        # Convert validity flags to integer type for consistency
        validity_columns = ['Left_Validity', 'Left_Pupil_Validity', 
                        'Right_Validity', 'Right_Pupil_Validity']
        for col in validity_columns:
            df[col] = df[col].astype(int)
        
        # Specify and order the final set of columns for output
        final_columns = [
            'TimeStamp',
            'Left_X', 'Left_Y', 'Left_Validity',
            'Left_Pupil', 'Left_Pupil_Validity',
            'Right_X', 'Right_Y', 'Right_Validity',
            'Right_Pupil', 'Right_Pupil_Validity'
        ]
        
        return df[final_columns]

    def save_data(self):
        """Save current buffer data to file and clear the buffer.

        This method is called internally by :meth:`stop_recording` and
        :meth:`close` to ensure all data is saved.
        """
        if not self.gaze_data:
            return  # Nothing to save

        # Make a copy of the buffers and clear them
        gaze_data_copy = self.gaze_data[:]
        event_data_copy = self.event_data[:]
        self.gaze_data.clear()
        self.event_data.clear()

        # Process gaze data
        gaze_df = pd.DataFrame(gaze_data_copy)
        gaze_df = self._process_timestamps(gaze_df)
        gaze_df = self._adapt_gaze_data(gaze_df)

        # Process events if they exist
        if event_data_copy:
            events_df = pd.DataFrame(event_data_copy, columns=['system_time_stamp', 'Event'])
            events_df = self._process_timestamps(events_df)

            if self.event_mode == 'samplebased':
                # Match events to nearest gaze samples
                df = pd.merge_asof(gaze_df, events_df[['TimeStamp', 'Event']],
                                on='TimeStamp',
                                direction='nearest',
                                tolerance=0.1)
            else:
                df = gaze_df
        else:
            df = gaze_df
            if self.event_mode == 'samplebased':
                df['Event'] = ''

        # Save gaze data
        file_exists = os.path.isfile(self.filename)
        df.to_csv(self.filename, mode='a', index=False, header=not file_exists)

        # Save events separately in precise mode
        if self.event_mode == 'precise' and event_data_copy:
            file_exists = os.path.isfile(self.events_filename)
            events_df[['TimeStamp', 'Event']].to_csv(
                self.events_filename, 
                mode='a', 
                index=False, 
                header=not file_exists
            )
            


    def start_recording(self, filename=None):
        """Start recording gaze data.

        This method initializes the data collection process by subscribing to
        gaze data and preparing data structures for storing the gaze and event
        data. It can optionally set a specific filename for data storage.

        Args:
            filename: Optional filename for saving data. If not provided, the
                default filename will be used.
        """
        # Set the filename if provided
        if filename is not None:
            self.filename = filename
            # Update events filename if in precise mode
            if self.event_mode == 'precise':
                base_name = os.path.splitext(self.filename)[0]
                self.events_filename = f"{base_name}_events.tsv"

        # Clear previous data
        self.gaze_data = []
        self.event_data = []

        # Subscribe to gaze data updates
        self.eyetracker.subscribe_to(
            tr.EYETRACKER_GAZE_DATA,
            self._on_gaze_data,
            as_dictionary=True
        )

        # Wait briefly before starting the recording
        core.wait(1)

        # Set recording flag and timestamp
        self.recording = True
        self.t0 = tr.get_system_time_stamp()


    def stop_recording(self):
        """
        Stop recording and save final data.

        This method unsubscribes from gaze data updates and sets the recording
        flag to False. It also calls the save_data() method to save any
        remaining data to the file.
        """
        # Check if recording is active
        if not self.recording:
            raise RuntimeWarning("Not recording now.")

        # Unsubscribe from gaze data updates
        self.eyetracker.unsubscribe_from(tr.EYETRACKER_GAZE_DATA, self._on_gaze_data)

        # Set recording flag to False
        self.recording = False

        # Save any remaining data to file
        self.save_data()


    def record_event(self, event_label):
        """Record an event with timestamp.

        Args:
            event_label: Event label to record
        """
        if not self.recording:
            raise RuntimeWarning("Not recording now.")
        self.event_data.append([tr.get_system_time_stamp(), event_label])


    def close(self):
        """Clean up and ensure all data is saved.

        This method is called by atexit and is also available as a public method.
        It stops recording and saves all data that was not saved before.
        """
        if self.recording:
            self.stop_recording()


    def run_calibration(self, calibration_points, infant_stims, shuffle=True, 
                    audio=None, focus_time=0.5, save_calib=False,
                    calib_filename="calibration.dat"):
        """Run infant-friendly calibration with point selection.

        Args:
            calibration_points: List of calibration point coordinates
            infant_stims: List of image files for calibration stimuli
            shuffle: Whether to shuffle stimuli presentation order
            audio: Audio stimulus to play (optional)
            focus_time: Time to wait before collecting data at each point
            save_calib: Whether to save calibration data (default: False)
            calib_filename: Filename for saving calibration (default: "calibration.dat")

        Returns:
            Success flag indicating if calibration was successful.
        """
        if len(calibration_points) < 2 or len(calibration_points) > 9:
            raise ValueError("Calibration points must be between 2 and 9")

        self.targets = InfantStimuli(self.win, infant_stims, shuffle=shuffle)
        self._audio = audio

        self.original_calibration_points = calibration_points[:]
        cp_num = len(self.original_calibration_points)
        self.retry_points = list(range(cp_num))

        in_calibration_loop = True

        while in_calibration_loop:
            # Clear points to recalibrate
            self.calibration.enter_calibration_mode()

            # Collect calibration data for current points
            point_idx = -1
            collecting = True
            while collecting:
                for key in event.getKeys():
                    if key in self.numkey_dict:
                        point_idx = self.numkey_dict[key]
                        if self._audio:
                            self._audio.play()
                    elif key == 'space':
                        if point_idx in self.retry_points:
                            core.wait(focus_time)
                            self.calibration.collect_data(
                                calibration_points[point_idx][0],
                                calibration_points[point_idx][1]
                            )
                    elif key == 'return':
                        collecting = False
                        break

                if point_idx >= 0 and point_idx < len(calibration_points):
                    stim = self.targets.get_stim(point_idx)
                    stim.setPos(calibration_points[point_idx])
                    stim.draw()

                self.win.flip()

            # Compute and show results
            self.calibration_result = self.calibration.compute_and_apply()
            result_img = self._show_calibration_result()

            # Show instructions
            instructions = visual.TextStim(
                self.win,
                text="Press numbers to select points to recalibrate\n"
                    "Space to accept/recalibrate\n"
                    "Escape to abort",
                pos=(0, -self.win.size[1]/4),
                color='white',
                height=20
            )

            # Allow selection of points to recalibrate
            self.retry_points = []
            selecting = True
            while selecting:
                result_img.draw()
                instructions.draw()

                for key in event.getKeys():
                    if key in self.numkey_dict:
                        idx = self.numkey_dict[key]
                        if idx < cp_num:
                            if idx in self.retry_points:
                                self.retry_points.remove(idx)
                            else:
                                self.retry_points.append(idx)
                    elif key == 'space':
                        selecting = False
                        if len(self.retry_points) == 0:
                            in_calibration_loop = False
                    elif key == 'escape':
                        selecting = False
                        in_calibration_loop = False

                # Show selected points
                for retry_p in self.retry_points:
                    visual.Circle(
                        self.win,
                        radius=10,
                        pos=calibration_points[retry_p],
                        lineColor='yellow'
                    ).draw()

                self.win.flip()

            # If points were selected, discard their data for recalibration
            for point_index in self.retry_points:
                x, y = calibration_points[point_index]
                self.calibration.discard_data(x, y)

        self.calibration.leave_calibration_mode()
        success = self.calibration_result.status == tr.CALIBRATION_STATUS_SUCCESS

        # Save calibration if requested and successful
        if success and save_calib:
            self.save_calibration(calib_filename)

        return success

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