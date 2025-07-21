# Standard library imports
import time
import numpy as np
from datetime import datetime
from PIL import Image, ImageDraw

# Third party imports
import tobii_research as tr
from psychopy import core, event, visual

# Local imports
from . import calibration_config as cfg
from .Utils import InfantStimuli, NicePrint
from .Coords import get_tobii_pos, psychopy_to_pixels, convert_height_to_units


class BaseCalibrationSession:
    """
    Base class with common functionality for both calibration types.
    """
    def __init__(
        self,
        win,
        infant_stims,
        shuffle=True,
        audio=None,
        anim_type='zoom',
    ):
        """
        Common initialization for both calibration types.
        
        Parameters
        ----------
        win : psychopy.visual.Window
            PsychoPy window for rendering stimuli and instructions.
        infant_stims : list of str
            List of image file paths for attention-getting stimuli.
        shuffle : bool, optional
            Whether to randomize stimulus order each run. Default True.
        audio : psychopy.sound.Sound, optional
            Sound to play when user selects a point. Default None.
        anim_type : str, optional
            Animation style: 'zoom' or 'trill'. Default 'zoom'.
        """
        self.win = win
        self.infant_stims = infant_stims
        self.shuffle = shuffle
        self.audio = audio
        self.focus_time = cfg.ANIMATION_SETTINGS['focus_time']
        self.anim_type = anim_type
        
        self.targets = None
        self.remaining_points = []  # Track which points still need calibration
        
        # Create calibration border (red thin border)
        self._create_calibration_border()
    
    
    def _create_calibration_border(self):
        """Create a thin red border to indicate calibration mode."""
        # Get window dimensions
        win_width = self.win.size[0]
        win_height = self.win.size[1]
        
        # Convert border thickness from height units to current units
        border_thickness = convert_height_to_units(self.win, cfg.DEFAULT_BORDER_THICKNESS_HEIGHT)
        
        # Convert to appropriate units for consistent sizing
        if self.win.units == 'height':
            # In height units, width is adjusted by aspect ratio
            border_width = win_width / win_height  # Full width in height units
            border_height = 1.0  # Full height in height units
        elif self.win.units == 'norm':
            border_width = 2.0  # Full width in norm units (-1 to 1)
            border_height = 2.0  # Full height in norm units
        else:
            border_width = win_width
            border_height = win_height
        
        # Create four rectangles for the border
        self.border_top = visual.Rect(
            self.win,
            width=border_width,
            height=border_thickness,
            pos=(0, border_height/2 - border_thickness/2),
            fillColor='red',
            lineColor=None,
            units=self.win.units  # Use same units as window
        )
        
        self.border_bottom = visual.Rect(
            self.win,
            width=border_width,
            height=border_thickness,
            pos=(0, -border_height/2 + border_thickness/2),
            fillColor='red',
            lineColor=None,
            units=self.win.units
        )
        
        self.border_left = visual.Rect(
            self.win,
            width=border_thickness,
            height=border_height,
            pos=(-border_width/2 + border_thickness/2, 0),
            fillColor='red',
            lineColor=None,
            units=self.win.units
        )
        
        self.border_right = visual.Rect(
            self.win,
            width=border_thickness,
            height=border_height,
            pos=(border_width/2 - border_thickness/2, 0),
            fillColor='red',
            lineColor=None,
            units=self.win.units
        )
    
    
    def _draw_calibration_border(self):
        """Draw the red calibration border."""
        self.border_top.draw()
        self.border_bottom.draw()
        self.border_left.draw()
        self.border_right.draw()
    
    
    def _create_message_visual(self, formatted_text, pos=(0, -0.15), font_type="instruction_text"):
        """
        Create a visual text stimulus from pre-formatted text.
        
        Parameters
        ----------
        formatted_text : str
            Pre-formatted text (e.g., from NicePrint)
        pos : tuple, optional
            Position of the text box (default slightly below center)
        font_type : str, optional
            Type of font to use from _font_size_multipliers keys
            
        Returns
        -------
        visual.TextStim
            PsychoPy text stimulus
        """
        # Get font and size
        size_multiplier = cfg.FONT_SIZE_MULTIPLIERS[font_type]

        return visual.TextStim(
            self.win,
            text=formatted_text,
            pos=pos,
            color='white',
            height=self._get_text_height(size_multiplier),
            alignText='center',
            anchorHoriz='center',
            anchorVert='center',
            units=self.win.units,  # Use same units as window
            font='Consolas',  # Monospace font that supports Unicode box characters
            languageStyle='LTR'  # Left-to-right text
        )
    
    
    def _get_text_height(self, size_percentage=2.0):
        """
        Calculate text height based on height units and size multiplier.
        
        Parameters
        ----------
        size_percentage : float
            Multiplier for the base text height. Default is 2.0.
            
        Returns
        -------
        float
            Text height in window units.
        """
        # Get base text height from config and convert to current units
        base_height = convert_height_to_units(self.win, cfg.DEFAULT_TEXT_HEIGHT_HEIGHT)
        # Scale by the size percentage
        return base_height * (size_percentage / 2.0)
    
    
    def show_message_and_wait(self, body, title="", pos=(0, -0.15)):
        """
        Display a message on screen and in console, then wait for keypress.
        
        Parameters
        ----------
        body : str
            The main message text
        title : str, optional
            Title for the message box
        pos : tuple, optional
            Position of the message box on screen
            
        Returns
        -------
        None
        """
        # Use NicePrint to print to console AND get formatted text
        formatted_text = NicePrint(body, title)
        
        # Create on-screen message using the formatted text
        message_visual = self._create_message_visual(formatted_text, pos)
        
        # Show message on screen
        self.win.clearBuffer()
        message_visual.draw()
        self.win.flip()
        
        # Wait for any key press
        event.waitKeys()
    
    
    def check_points(self, calibration_points):
        """
        Ensure number of calibration points is within allowed range.
        """
        if not (2 <= len(calibration_points) <= 9):
            raise ValueError("Calibration points must be between 2 and 9")
    
    
    def _prepare_session(self, calibration_points):
        """
        Initialize stimuli sequence and remaining points list.
        """
        self.targets = InfantStimuli(
            self.win,
            self.infant_stims,
            shuffle=self.shuffle
        )
        # Initialize remaining points to all points
        self.remaining_points = list(range(len(calibration_points)))
    
    
    def _animate(self, stim, clock):
        """
        Animate a stimulus with zoom or rotation ('trill') effects.
        
        Uses height-based settings that are automatically converted to current window units
        for consistent visual appearance across different screen configurations.
        
        Parameters
        ----------
        stim : psychopy.visual stimulus
            The stimulus object to animate
        point_idx : int
            Index of the calibration point (for compatibility)
        clock : psychopy.core.Clock
            Clock object for timing animations
            
        Notes
        -----
        Animation timing is controlled by cfg.ANIMATION_SETTINGS.
        Size settings are defined in height units and automatically converted to window units.
        Supported animation types: 'zoom' (cosine oscillation), 'trill' (discrete rotation pulses).
        """
        
        if self.anim_type == 'zoom':
            # Calculate elapsed time with zoom-specific speed multiplier
            elapsed_time = clock.getTime() * cfg.ANIMATION_SETTINGS['zoom_speed']
            
            # Retrieve and convert size settings from height units to current window units
            min_size_height = cfg.ANIMATION_SETTINGS['min_zoom_size']
            max_size_height = cfg.ANIMATION_SETTINGS['max_zoom_size']
            
            min_size = convert_height_to_units(self.win, min_size_height)
            max_size = convert_height_to_units(self.win, max_size_height)
            
            # Calculate smooth oscillation between min and max sizes using cosine
            # Cosine provides smooth acceleration/deceleration at size extremes
            size_range = max_size - min_size
            normalized_oscillation = (np.cos(elapsed_time) + 1) / 2.0  # Normalize to 0-1 range
            current_size = min_size + (normalized_oscillation * size_range)
            
            # Apply calculated size to stimulus (square aspect ratio)
            stim.setSize([current_size, current_size])
            
        elif self.anim_type == 'trill':
            # Set fixed size for trill animation from configuration
            trill_size_height = cfg.ANIMATION_SETTINGS['trill_size']
            trill_size = convert_height_to_units(self.win, trill_size_height)
            stim.setSize([trill_size, trill_size])
            
            # Create rapid trill and stop pattern
            elapsed_time = clock.getTime()
            trill_cycle_duration = cfg.ANIMATION_SETTINGS['trill_cycle_duration']  # 1.5s total
            trill_active_duration = cfg.ANIMATION_SETTINGS['trill_active_duration']  # 1.0s active
            
            # Determine position in the cycle
            cycle_position = elapsed_time % trill_cycle_duration
            
            if cycle_position < trill_active_duration:
                # TRILL PHASE: Rapid back-and-forth oscillations
                
                # Create rapid oscillations using high-frequency sine wave
                trill_frequency = cfg.ANIMATION_SETTINGS['trill_frequency']  # Oscillations per second
                trill_time = cycle_position * trill_frequency * 2 * np.pi
                
                # Create sharp, rapid back-and-forth movement
                rotation_base = np.sin(trill_time)
                
                # Apply rotation range
                rotation_angle = rotation_base * cfg.ANIMATION_SETTINGS['trill_rotation_range']
                stim.setOri(rotation_angle)
                
            else:
                # STOP PHASE: No rotation
                stim.setOri(0)
        
        # Render the animated stimulus
        stim.draw()
    

    def _selection_phase(self, calibration_points, result_img):
        """
        Show results and allow user to select points for retry.
        Uses height-based sizing for highlight circles.
        
        Parameters
        ----------
        calibration_points : list of (float, float)
            List of calibration point coordinates
        result_img : visual.SimpleImageStim
            Image showing calibration results
        
        Returns
        -------
        list or None
            List of indices to retry, empty list to accept, None to restart all
        """
        retries = set()
        
        # Create instructions for results phase
        result_instructions = """Review calibration results above.

• Press SPACE to accept calibration
• Press numbers to select points for retry  
• Press ESCAPE to restart calibration

Make your choice now:"""
        
        formatted_instructions = NicePrint(result_instructions, "Calibration Results")
        result_instructions_visual = visual.TextStim(
            self.win,
            text=formatted_instructions,
            pos=(0, -0.25),
            color='white',
            height=self._get_text_height(1.2),
            alignText='center',
            anchorHoriz='center',
            anchorVert='center',
            units=self.win.units,
            font='Consolas',  # Monospace font for Unicode support
            languageStyle='LTR'
        )
        
        while True:
            # Draw result image, calibration border, and instructions
            result_img.draw()
            self._draw_calibration_border()
            result_instructions_visual.draw()
            
            # Highlight retry points with height-based sizing
            for retry_idx in retries:
                if retry_idx < len(calibration_points):
                    # Convert highlight size and line width from height units
                    highlight_radius = convert_height_to_units(self.win, cfg.DEFAULT_HIGHLIGHT_SIZE_HEIGHT)
                    line_width_height = cfg.DEFAULT_LINE_WIDTH_HEIGHT
                    # Convert line width to pixels for consistency (PsychoPy expects pixel values for lineWidth)
                    line_width_pixels = line_width_height * self.win.size[1]
                    
                    # Create highlight circle with proper scaling
                    highlight = visual.Circle(
                        self.win,
                        radius=highlight_radius,
                        pos=calibration_points[retry_idx],
                        lineColor=cfg.CALIBRATION_COLORS['highlight'],
                        fillColor=None,                       # No fill for consistency
                        lineWidth=max(1, int(line_width_pixels)),  # Ensure minimum 1 pixel width
                        edges=128,                            # smooth circle
                        units=self.win.units                  # Explicit units
                    )
                    highlight.draw()

            self.win.flip()
            
            for key in event.getKeys():
                if key in cfg.NUMKEY_DICT:
                    idx = cfg.NUMKEY_DICT[key]
                    if 0 <= idx < len(calibration_points):
                        if idx in retries:
                            retries.remove(idx)
                        else:
                            retries.add(idx)
                            
                elif key == 'space':
                    return list(retries)  # Accept calibration (with or without retries)
                    
                elif key == 'escape':
                    return None  # Restart all calibration
                    
    def _collection_phase(self, calibration_points, **kwargs):
        """
        Unified collection phase for both calibration types.
        Uses callback methods for type-specific data collection.
        Only allows interaction with points in remaining_points list.
        
        Parameters
        ----------
        calibration_points : list of (float, float)
            List of calibration point coordinates
        **kwargs : dict
            Additional arguments passed to collect method (e.g., num_samples for mouse)
        
        Returns
        -------
        bool
            True if collection completed, False if user pressed escape
        """
        clock = core.Clock()
        point_idx = -1
        
        while True:
            # Clear screen and draw calibration border
            self.win.clearBuffer()
            self._draw_calibration_border()
            
            # Handle keyboard input
            for key in event.getKeys():
                if key in cfg.NUMKEY_DICT:
                    # Select point; play audio if available
                    candidate_idx = cfg.NUMKEY_DICT[key]
                    # Only allow selection of points that are still remaining
                    if candidate_idx in self.remaining_points:
                        point_idx = candidate_idx
                        if self.audio:
                            self.audio.play()
                    else:
                        # Ignore key press for points not in remaining list
                        point_idx = -1
                        
                elif key == 'space' and point_idx in self.remaining_points:
                    # Collect data using subclass-specific method
                    success = self._collect_data_at_point(
                        calibration_points[point_idx], 
                        point_idx, 
                        **kwargs
                    )
                    if success:
                        if self.audio:
                            self.audio.pause()
                        # DON'T remove from remaining points - allow re-doing same point
                        point_idx = -1
                        
                elif key == 'return':
                    # Finish early - check if we have any data
                    if self._has_collected_data():
                        return True
                        
                elif key == 'escape':
                    # Exit calibration
                    self._clear_collected_data()
                    return False
            
            # Show stimulus at selected point (only if it's in remaining points)
            if point_idx in self.remaining_points:
                stim = self.targets.get_stim(point_idx)
                stim.setPos(calibration_points[point_idx])
                self._animate(stim, clock)
            
            self.win.flip()
    
    
    def _create_calibration_result_image(self, calibration_points, sample_data):
        """
        Common function to create calibration result visualization.
        """
        # Create blank RGBA image matching window size (transparent background like Tobii)
        img = Image.new("RGBA", tuple(self.win.size))
        img_draw = ImageDraw.Draw(img)
        
        # Convert plot line width from height units to pixels
        line_width_pixels = cfg.DEFAULT_PLOT_LINE_WIDTH_HEIGHT * self.win.size[1]
        
        # Convert target circle size and line width from height units to pixels
        target_circle_radius_pixels = cfg.DEFAULT_TARGET_CIRCLE_SIZE_HEIGHT * self.win.size[1]
        target_circle_width_pixels = cfg.DEFAULT_TARGET_CIRCLE_WIDTH_HEIGHT * self.win.size[1]
        
        # Draw calibration data
        for point_idx, samples in sample_data.items():
            if point_idx < len(calibration_points):
                target_pos = calibration_points[point_idx]
                target_pix = psychopy_to_pixels(self.win, target_pos)
                
                # Draw target circle with configurable size and line width
                img_draw.ellipse(
                    (target_pix[0] - target_circle_radius_pixels, 
                    target_pix[1] - target_circle_radius_pixels,
                    target_pix[0] + target_circle_radius_pixels, 
                    target_pix[1] + target_circle_radius_pixels),
                    outline=cfg.CALIBRATION_COLORS['target_outline'],
                    width=max(1, int(target_circle_width_pixels))  # Ensure minimum 1 pixel width
                )
                
                # Draw lines from target to samples
                for _, sample_pos, line_color in samples:
                    sample_pix = psychopy_to_pixels(self.win, sample_pos)
                    img_draw.line(
                        (target_pix[0], target_pix[1], sample_pix[0], sample_pix[1]),
                        fill=line_color,
                        width=max(1, int(line_width_pixels))  # Ensure minimum 1 pixel width
                    )
        
        # Wrap in SimpleImageStim and return
        return visual.SimpleImageStim(self.win, img, autoLog=False)


class TobiiCalibrationSession(BaseCalibrationSession):
    """
    Tobii-based calibration session for real eye tracking.
    
    This is the original calibration system that works with Tobii hardware.
    """

    def __init__(
        self,
        win,
        calibration_api,
        infant_stims,
        shuffle=True,
        audio=None,
        anim_type='zoom',
    ):
        """
        Initialize Tobii calibration session.
        
        Parameters
        ----------
        calibration_api : tobii_research.ScreenBasedCalibration
            Tobii's calibration interface.
        """
        super().__init__(
            win, infant_stims, shuffle, audio, anim_type
        )
        self.calibration = calibration_api

    def run(self, calibration_points, save_calib=False):
        """
        Main routine to run the full Tobii calibration workflow.

        This function presents each calibration target, collects gaze data
        via the eye tracker, shows the results, and allows the user to retry
        any subset of points until satisfied. Optionally saves the calibration.

        Parameters
        ----------
        calibration_points : list of (float, float)
            List of PsychoPy (x, y) positions for calibration targets.
        save_calib : bool
            If True, saves the resulting calibration data to disk upon success.

        Returns
        -------
        bool
            True if calibration was successful, False if aborted or failed.
        """

        # --- 1. Show instructions before anything happens ---
        instructions_text = """Tobii Eye Tracker Calibration Setup:

    • Press number keys (1-9) to select calibration points
    • Look at the animated stimulus when it appears
    • Press SPACE to collect eye tracking data
    • Press ENTER to finish collecting and see results
    • Press ESCAPE to exit calibration

    Any key will start calibration immediately!"""

        self.show_message_and_wait(instructions_text, "Eye Tracker Calibration")

        # --- 2. Initial verification and preparation ---
        self.check_points(calibration_points)
        self._prepare_session(calibration_points)

        # --- 3. Enter Tobii calibration mode ---
        self.calibration.enter_calibration_mode()

        # --- 4. Main calibration-retry loop ---
        while True:
            # --- 4a. Data collection phase ---
            success = self._collection_phase(calibration_points)
            if not success:
                self.calibration.leave_calibration_mode()
                return False

            # --- 4b. Compute and show calibration results ---
            self.calibration_result = self.calibration.compute_and_apply()
            result_img = self._show_calibration_result(calibration_points)

            # --- 4c. Let user select points to retry ---
            retries = self._selection_phase(calibration_points, result_img)
            if retries is None:
                # Restart all: reset remaining points and clear collected data
                self.remaining_points = list(range(len(calibration_points)))
                self._clear_collected_data()
                continue
            elif not retries:
                # Accept: finished!
                break
            else:
                # Retry specific points: update remaining points and discard data
                self.remaining_points = retries.copy()
                self._discard_phase(calibration_points, retries)

        # --- 5. Exit calibration mode ---
        self.calibration.leave_calibration_mode()

        # --- 6. Optionally save calibration data ---
        success = (self.calibration_result.status == tr.CALIBRATION_STATUS_SUCCESS)
        if success and save_calib:
            data = self.calibration.retrieve_calibration_data()
            fname = f"{datetime.now():%Y-%m-%d_%H-%M-%S}_calibration.dat"
            with open(fname, 'wb') as f:
                f.write(data)

        return success


    def _collect_data_at_point(self, target_pos, point_idx, **kwargs):
        """
        Collect Tobii eye tracking data at a calibration point.
        
        Parameters
        ----------
        target_pos : tuple
            Target position in PsychoPy coordinates
        point_idx : int
            Index of the calibration point
        **kwargs : dict
            Unused for Tobii calibration
            
        Returns
        -------
        bool
            True if data was collected successfully
        """
        x, y = get_tobii_pos(self.win, target_pos)
        
        # Clear any existing data at this point first
        self.calibration.discard_data(x, y)
        
        # Wait focus time then collect NEW data
        core.wait(self.focus_time)
        self.calibration.collect_data(x, y)
        return True

    
    
    def _has_collected_data(self):
        """
        Check if any Tobii calibration data has been collected.
        
        Returns
        -------
        bool
            True if any points have been collected
        """
        # If remaining points is smaller than total points, we've collected some data
        return len(self.remaining_points) < len(range(9))  # Assuming max 9 points
    
    
    def _clear_collected_data(self):
        """
        Clear Tobii calibration data.
        For Tobii, this is handled by the API when we restart calibration mode.
        """
        # Tobii handles clearing internally
        pass

    def _discard_phase(self, calibration_points, retries):
        """
        Remove collected data for each retry point.
        """
        for idx in retries:
            x, y = get_tobii_pos(self.win, calibration_points[idx])
            self.calibration.discard_data(x, y)

    def _show_calibration_result(self, calibration_points):
        """
        Show Tobii calibration results using the common plotting function.
        Only shows results for ALL collected points, not just remaining ones.

        Parameters
        ----------
        calibration_points : list of (float, float)
            List of calibration point coordinates

        Returns
        -------
        SimpleImageStim
            PsychoPy stimulus containing the result image.
        """
        # Prepare sample data in common format
        sample_data = {}
        
        # Only process if not a full failure
        if self.calibration_result.status != tr.CALIBRATION_STATUS_FAILURE:
            for point_idx, point in enumerate(self.calibration_result.calibration_points):
                samples = []
                
                # Process each calibration sample
                for sample in point.calibration_samples:
                    target_pos = point.position_on_display_area
                    
                    # Left eye sample (green)
                    if sample.left_eye.validity == tr.VALIDITY_VALID_AND_USED:
                        left_pos = sample.left_eye.position_on_display_area
                        samples.append((target_pos, left_pos, cfg.CALIBRATION_COLORS['left_eye']))
                    
                    # Right eye sample (red)
                    if sample.right_eye.validity == tr.VALIDITY_VALID_AND_USED:
                        right_pos = sample.right_eye.position_on_display_area
                        samples.append((target_pos, right_pos, cfg.CALIBRATION_COLORS['right_eye']))
                
                if samples:
                    sample_data[point_idx] = samples
        
        # Use common plotting function
        return self._create_calibration_result_image(calibration_points, sample_data)


class MouseCalibrationSession(BaseCalibrationSession):
    """
    Mouse-based calibration session for simulation mode.
    
    This runs independently of Tobii and collects mouse position
    samples at calibration targets.
    """
    
    def __init__(
        self,
        win,
        infant_stims,
        mouse,
        shuffle=True,
        audio=None,
        anim_type='zoom',
    ):
        """
        Initialize mouse-based calibration session.
        
        Parameters
        ----------
        mouse : psychopy.event.Mouse
            Mouse object for getting positions.
        """
        super().__init__(
            win, infant_stims, shuffle, audio, anim_type
        )
        self.mouse = mouse
        self.calibration_data = {}  # point_idx -> list of (target_pos, sample_pos, timestamp)
    
    
    def run(self, calibration_points, num_samples=5):
        """
        Main function to run the mouse-based calibration routine.

        Parameters
        ----------
        calibration_points : list of (float, float)
            List of target positions in PsychoPy coordinates.
        num_samples : int
            How many mouse position samples to collect at each calibration point.

        Returns
        -------
        bool
            True if calibration finished successfully, False if the user exits early.
        """

        # --- 1. Show the instructions screen ---
        instructions_text = """Mouse-Based Calibration Setup:

    • Press number keys (1-9) to select calibration points
    • Move your mouse to the animated stimulus
    • Press SPACE to collect samples at the selected point
    • Press ENTER to finish collecting and see results
    • Press ESCAPE to exit calibration

    Any key will start calibration immediately!"""
        
        self.show_message_and_wait(instructions_text, "Calibration Setup")
        
        # --- 2. Sanity check and prepare stimuli ---
        self.check_points(calibration_points)
        self._prepare_session(calibration_points)
        
        # --- 3. Main calibration loop ---
        while True:
            # --- 3a. Collect calibration data at each point ---
            success = self._collection_phase(calibration_points, num_samples=num_samples)
            if not success:
                return False
                
            # --- 3b. Show results of current calibration ---
            result_img = self._show_results(calibration_points)
            
            # --- 3c. Let user review and pick points to retry ---
            retries = self._selection_phase(calibration_points, result_img)
            
            if retries is None:
                # Restart all: reset remaining points and clear data
                self.remaining_points = list(range(len(calibration_points)))
                self.calibration_data.clear()
                continue
            elif not retries:
                # Accept: we're done!
                return True
            else:
                # Retry specific points: update remaining points and remove their data
                self.remaining_points = retries.copy()
                for idx in retries:
                    if idx in self.calibration_data:
                        del self.calibration_data[idx]

    
    
    def _collect_data_at_point(self, target_pos, point_idx, **kwargs):
        """
        Collect mouse samples at a single calibration target.

        Parameters
        ----------
        target_pos : tuple
            The (x, y) coordinates of the current calibration target (in PsychoPy units).
        point_idx : int
            The index of this calibration point in the full list.
        **kwargs : dict
            Must contain 'num_samples': how many mouse samples to take.

        Returns
        -------
        bool
            True if samples were collected.
        """

        # Clear existing data for this point first (Option 2: Replace)
        if point_idx in self.calibration_data:
            del self.calibration_data[point_idx]

        # How many mouse samples to take at this point? (Default: 5)
        num_samples = kwargs.get('num_samples', 5)

        # --- Wait a moment before sampling ---
        core.wait(self.focus_time)

        # --- Setup: collect all samples in this list ---
        samples = []
        sample_duration = 1.0  # Total time (seconds) over which to collect samples
        sample_interval = sample_duration / num_samples  # Time between samples

        # --- 1. Collect mouse samples over a brief period ---
        for i in range(num_samples):
            mouse_pos = self.mouse.getPos()      # Get current mouse position (x, y)
            timestamp = time.time()              # Record current time
            samples.append((target_pos, mouse_pos, timestamp))

            # Don't wait after the final sample
            if i < num_samples - 1:
                core.wait(sample_interval)

        # --- 2. Store the collected samples ---
        if point_idx not in self.calibration_data:
            self.calibration_data[point_idx] = []
        self.calibration_data[point_idx].extend(samples)

        # --- 3. Done! Return True to indicate success ---
        return True

        
        
    def _has_collected_data(self):
        """
        Check if any mouse calibration data has been collected yet.

        Returns
        -------
        bool
            True if there is any calibration data in the storage,
            False if none has been collected yet.
        """
        return bool(self.calibration_data)


    def _clear_collected_data(self):
        """
        Remove all previously collected mouse calibration data.
        """
        self.calibration_data.clear()


    def _show_results(self, calibration_points):
        """
        Visualize and return a summary image of the collected mouse calibration data.

        Parameters
        ----------
        calibration_points : list of (float, float)
            The (x, y) positions of all calibration targets.

        Returns
        -------
        visual.SimpleImageStim
            A PsychoPy image stimulus with the plotted calibration results.
        """
        # --- Prepare data for plotting ---
        sample_data = {}

        for point_idx, samples in self.calibration_data.items():
            formatted_samples = []
            for target_pos, sample_pos, _ in samples:
                # Draw a line from the target to each sample; use orange color for mouse samples
                formatted_samples.append((target_pos, sample_pos, cfg.CALIBRATION_COLORS['mouse']))
            if formatted_samples:
                sample_data[point_idx] = formatted_samples

        # --- Use the shared plotting function ---
        return self._create_calibration_result_image(calibration_points, sample_data)