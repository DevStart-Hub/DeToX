# Standard library imports
import os
import time
import atexit
from datetime import datetime

# Third party imports
import numpy as np
import pandas as pd
import tobii_research as tr
from psychopy import core, event, visual
from PIL import Image, ImageDraw

# Local imports
import coord_utils


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
    # TODO: better format default settings

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
    
        # Default settings dictionary
    _animation_settings = {
        'shrink_speed': 1.0,  # Slower for infants
        'shrink_sec': 3.0,    # Will be divided by shrink_speed
        'target_min': 0.2,    # Minimum size for calibration target
    }

    _shrink_speed = 1  # Slower for infants
    calibration_dot_color = (0, 0, 0)
    calibration_disc_color = (-1, -1, 0)
    calibration_target_min = 0.2
    recording = False

    def __init__(self, win, id=0, filename=None, event_mode='samplebased'):
        """
        Initialize the TobiiController.

        The TobiiController class is a simplified wrapper around the Tobii
        eye tracker API. It provides methods for starting/stopping gaze data
        recording and saving the data to a file.

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
        
        # Set the filename
        if filename is None:
            self.filename = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
        else:
            # Check if filename has an extension
            self.basename, ext = os.path.splitext(filename)
            if ext:
                raise Warning(
                    f"Filename '{filename}' contains extension '{ext}'. "
                    "Please provide filename without extension. "
                    "The .csv extension will be added automatically."
                )
            self.filename = f"{self.basename}.csv"


        # For precise event mode, create events filename
        if self.event_mode == 'precise':
            self.events_filename = f"{self.basename}_events.csv"

        # Set the default key mappings
        self.numkey_dict = self._default_numkey_dict.copy()

        # Connect to the eye tracker
        eyetrackers = tr.find_all_eyetrackers()
        if len(eyetrackers) == 0:
            raise RuntimeError("No Tobii eyetrackers detected./nVerify the connection and make sure to power on the eyetracker before starting your computer.")
        else:
            self.eyetracker = eyetrackers[self.eyetracker_id]

        # Initialize the calibration object
        self.calibration = tr.ScreenBasedCalibration(self.eyetracker)

        # Initialize data storage
        self.gaze_data = []
        self.event_data = []

        # Register the close method to be called when the program exits
        atexit.register(self.close)

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

    def _show_calibration_result(self):
        """
        Show calibration results with lines indicating accuracy.

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
        # Create a new image with the same size as the PsychoPy window
        img = Image.new("RGBA", tuple(self.win.size))
        
        # Create an ImageDraw object to draw on the image
        img_draw = ImageDraw.Draw(img)
        
        # Create a PsychoPy SimpleImageStim object from the image
        result_img = visual.SimpleImageStim(self.win, img, autoLog=False)
        
        # Check if the calibration result is not a failure
        if self.calibration_result.status != tr.CALIBRATION_STATUS_FAILURE:
            # Iterate over each calibration point
            for point in self.calibration_result.calibration_points:
                p = point.position_on_display_area
                
                # Draw a small circle for the calibration point
                img_draw.ellipse(
                    ((p[0] * self.win.size[0] - 3, p[1] * self.win.size[1] - 3),
                     (p[0] * self.win.size[0] + 3, p[1] * self.win.size[1] + 3)),
                    outline=(0, 0, 0, 255)  # Black outline
                )
                
                # Iterate over each sample in the calibration point
                for sample in point.calibration_samples:
                    lp = sample.left_eye.position_on_display_area
                    rp = sample.right_eye.position_on_display_area
                    
                    # Check if the left eye sample is valid
                    if sample.left_eye.validity == tr.VALIDITY_VALID_AND_USED:
                        # Draw a line for the left eye if the sample is valid
                        img_draw.line(
                            ((p[0] * self.win.size[0], p[1] * self.win.size[1]),
                             (lp[0] * self.win.size[0], lp[1] * self.win.size[1])),
                            fill=(0, 255, 0, 255)  # Green line for left eye
                        )
                    # Check if the right eye sample is valid
                    if sample.right_eye.validity == tr.VALIDITY_VALID_AND_USED:
                        # Draw a line for the right eye if the sample is valid
                        img_draw.line(
                            ((p[0] * self.win.size[0], p[1] * self.win.size[1]),
                             (rp[0] * self.win.size[0], rp[1] * self.win.size[1])),
                            fill=(255, 0, 0, 255)  # Red line for right eye
                        )
                        
        # Set the image of the PsychoPy SimpleImageStim object to the
        # created image
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
        """Convert system timestamps to seconds from start."""
        # More efficient to operate directly on the series
        return np.round((df - self.t0) / 1000.0, 1)

    def _adapt_gaze_data(self, df):
        """Adapt gaze data format and convert coordinates."""
        # Convert coordinates more efficiently
        df['Left_X'], df['Left_Y'] = zip(*[coord_utils.get_psychopy_pos(self.win, coord) 
                                        for coord in df['left_gaze_point_on_display_area']])
        df['Right_X'], df['Right_Y'] = zip(*[coord_utils.get_psychopy_pos(self.win, coord) 
                                            for coord in df['right_gaze_point_on_display_area']])
        
        # Process timestamps
        df['TimeStamp'] = self._process_timestamps(df['system_time_stamp'])
        
        # Rename and convert in one step
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

        return df[['TimeStamp', 'Left_X', 'Left_Y', 'Left_Validity',
                'Left_Pupil', 'Left_Pupil_Validity', 
                'Right_X', 'Right_Y', 'Right_Validity',
                'Right_Pupil', 'Right_Pupil_Validity']]

    def save_data(self):
        """Save gaze and event data to files."""
        start_saving = time.time()

        if not self.gaze_data:
            print("No gaze data to save.")
            return

        # Make a copy of the buffers and clear them
        gaze_data_copy = self.gaze_data[:]
        event_data_copy = self.event_data[:]
        self.gaze_data.clear()
        self.event_data.clear()
       
        # Process gaze data
        gaze_df = pd.DataFrame(gaze_data_copy)
        gaze_df = self._adapt_gaze_data(gaze_df)
        
        # Process events if they exist
        if event_data_copy:
            events_df = pd.DataFrame(event_data_copy, columns=['system_time_stamp', 'Event'])
            events_df['TimeStamp'] = self._process_timestamps(events_df['system_time_stamp'])
            
            if self.event_mode == 'samplebased':
                gaze_df = pd.merge_asof(gaze_df, events_df[['TimeStamp', 'Event']],
                                    on='TimeStamp', direction='nearest')
            elif self.event_mode == 'precise':
                file_exists = os.path.isfile(self.events_filename)
                events_df[['TimeStamp', 'Event']].to_csv(
                    self.events_filename, mode='a', 
                    index=False, header=not file_exists
                )
        elif self.event_mode == 'samplebased':
            gaze_df['Event'] = ''

        # Save gaze data
        file_exists = os.path.isfile(self.filename)
        gaze_df.to_csv(self.filename, mode='a', index=False, header=not file_exists)
        
        print(f"Data saved in {round(time.time() - start_saving, 3)} seconds.")
        
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
        """
        Record an event with a timestamp.

        This method adds an event to the event data buffer. The event is
        represented as a list with two elements: the first element is the
        timestamp of the event in milliseconds relative to the start of the
        recording, and the second element is the event label provided as a
        string.

        The timestamp is obtained using tr.get_system_time_stamp(), which
        returns the current time in milliseconds since the system was started.

        If the recording is not active, a RuntimeWarning is raised.

        Parameters
        ----------
        event_label : str
            The label for the event to record

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

    def _animate(self, stim, point_idx, clock, anim_type='zoom', rotation_range=15):
        """
        Animate a stimulus with either zoom or rotation effect.

        This function takes a stimulus and animates it with either a zoom or
        trill (rotational) effect. The animation is updated based on the
        current time of the provided clock, and the animation style and
        parameters are determined by the other arguments.

        Parameters
        ----------
        stim : visual.ImageStim
            The PsychoPy stimulus to animate.
        point_idx : int
            Index of the calibration point.
        clock : psychopy.core.Clock
            Clock for timing the animation.
        anim_type : str, optional
            Type of animation ('zoom' or 'trill'). Default is 'zoom'.
        rotation_range : float, optional
            Range of rotation in degrees for trill animation. Default is 15.

        Returns
        -------
        None
        """
        # Get current time and adjust with a shrink speed factor
        time = clock.getTime() * self._shrink_speed

        if anim_type == 'zoom':
            # Calculate the scale factor for zoom animation
            orig_size = self.targets.get_stim_original_size(point_idx)
            scale_factor = np.sin(time)**2 + self.calibration_target_min
            newsize = [scale_factor * size for size in orig_size]
            # Set the size of the stimulus to the new size
            stim.setSize(newsize)

        elif anim_type == 'trill':
            # Calculate the new orientation angle for trill animation
            new_angle = np.sin(time) * rotation_range
            # Set the orientation of the stimulus to the new angle
            stim.setOri(new_angle)

        # Draw the stimulus with the updated properties
        stim.draw()

    def run_calibration(self, calibration_points, infant_stims, shuffle=True, 
                    audio=None, focus_time=0.5, anim_type='zoom', save_calib=False):
        """
        Run an infant-friendly calibration procedure with point selection and
        animated stimuli. The calibration points are presented in a sequence
        (either in order or shuffled) and at each point, an animated stimulus
        is presented (either zooming or trilling). The procedure can optionally
        play an attention-getting audio during the calibration process. The
        calibration data can be saved to a file if desired.

        Parameters
        ----------
        calibration_points : list of tuple
            List of (x, y) coordinates for calibration points
        infant_stims : list of str
            List of image file paths for calibration stimuli
        shuffle : bool, optional
            Whether to shuffle stimuli order. Default is True
        audio : psychopy.sound.Sound, optional
            Audio to play during calibration. Default is None
        focus_time : float, optional
            Time to wait before collecting data. Default is 0.5s
        save_calib : bool, optional
            Whether to save calibration data. Default is False

        Returns
        -------
        bool
            True if calibration successful, False otherwise
        """
        # Check if number of calibration points is valid
        if len(calibration_points) < 2 or len(calibration_points) > 9:
            raise ValueError("Calibration points must be between 2 and 9")

        # Initialize stimuli and settings
        self.targets = InfantStimuli(self.win, infant_stims, shuffle=shuffle)
        self._audio = audio

        # Setup calibration points
        self.original_calibration_points = calibration_points[:]
        cp_num = len(self.original_calibration_points)
        self.retry_points = list(range(cp_num))

        # Enter calibration mode once at start
        self.calibration.enter_calibration_mode()
        
        # Main calibration loop
        in_calibration_loop = True
        clock = core.Clock()  # For animation timing

        while in_calibration_loop:
            # Collection phase
            point_idx = -1
            collecting = True

            while collecting:
                # Handle key presses
                for key in event.getKeys():
                    if key in self.numkey_dict:
                        # User selected a point to collect data
                        point_idx = self.numkey_dict[key]
                        if self._audio:
                            self._audio.play()
                    elif key == 'space':
                        # User wants to accept this point and move on
                        if point_idx in self.retry_points:
                            core.wait(focus_time)
                            # Convert coordinates for Tobii
                            psychopy_point = calibration_points[point_idx]
                            tobii_x, tobii_y = coord_utils.get_tobii_pos(
                                self.win, 
                                psychopy_point
                            )
                            self.calibration.collect_data(tobii_x, tobii_y)
                            point_idx = -1  # Reset point index
                            if self._audio:
                                self._audio.pause()
                    elif key == 'return':
                        # User wants to move on to the next phase
                        collecting = False
                        break

                # Display animated stimulus
                if 0 <= point_idx < len(calibration_points):
                    stim = self.targets.get_stim(point_idx)
                    stim.setPos(calibration_points[point_idx])
                    self._animate(stim, point_idx, clock, anim_type=anim_type)

                self.win.flip()

            # Compute calibration
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

            # Handle point selection for recalibration
            self.retry_points = []
            selecting = True
            while selecting:
                result_img.draw()
                instructions.draw()

                for key in event.getKeys():
                    if key in self.numkey_dict:
                        # User selected a point to recalibrate
                        idx = self.numkey_dict[key]
                        if idx < cp_num:
                            if idx in self.retry_points:
                                self.retry_points.remove(idx)
                            else:
                                self.retry_points.append(idx)
                    elif key == 'space':
                        # User wants to accept/recalibrate
                        selecting = False
                        if len(self.retry_points) == 0:
                            in_calibration_loop = False
                    elif key == 'escape':
                        # User wants to abort
                        selecting = False
                        in_calibration_loop = False

                # Show selected points for recalibration
                for retry_p in self.retry_points:
                    visual.Circle(
                        self.win,
                        radius=10,
                        pos=calibration_points[retry_p],
                        lineColor='yellow'
                    ).draw()

                self.win.flip()

            # Handle recalibration points
            for point_index in self.retry_points:
                tobii_x, tobii_y = coord_utils.get_tobii_pos(
                    self.win, 
                    calibration_points[point_index]
                )
                self.calibration.discard_data(tobii_x, tobii_y)

        # Exit calibration mode at end
        self.calibration.leave_calibration_mode()
        
        success = self.calibration_result.status == tr.CALIBRATION_STATUS_SUCCESS

        # Save if requested and successful
        if success and save_calib:
            self.save_calibration()

        return success
    
    @property
    def numkey_dict(self):
        """
        Dictionary mapping number keys to point indices.

        Keys are strings, e.g. '1', '2', etc. Values are the indices of the
        calibration points, e.g. 0, 1, etc.

        Returns
        -------
        dict
            A dictionary mapping number keys to point indices
        """
        return self._numkey_dict

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
        if self.eyetracker is None:
            raise ValueError("Eyetracker not found")

        # Subscribe to user position guide events
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
                lx, ly = coord_utils.get_psychopy_pos_from_trackbox(self.win, [lx, ly], "height")
                leye.setPos((round(lx * 0.25, 4), round(ly * 0.2 + 0.4, 4)))
                leye.draw()

            # Update the right eye position
            if rv:
                # Convert TBCS coordinates to PsychoPy coordinates
                rx, ry = coord_utils.get_psychopy_pos_from_trackbox(self.win, [rx, ry], "height")
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