# Standard library imports
import time
import numpy as np
from datetime import datetime
from PIL import Image, ImageDraw

# Third party imports
import tobii_research as tr
from psychopy import core, event, visual

# Local imports
from .Utils import InfantStimuli, NicePrint
from .Coords import get_tobii_pos, psychopy_to_pixels


class BaseCalibrationSession:
    """
    Base class with common functionality for both calibration types.
    """
    
    # Default size dictionaries for different PsychoPy units
    _default_highlight_size = {
        "norm": 0.08,        # Normalized units (-1 to 1)
        "height": 0.04,      # Height units (screen height = 1)
        "pix": 40.0,         # Pixel units
        "degFlatPos": 1.0,   # Degrees (flat screen)
        "deg": 1.0,          # Degrees
        "degFlat": 1.0,      # Degrees (flat)
        "cm": 1.0,           # Centimeters
    }
    
    _default_marker_size = {
        "norm": 0.02,        # Smaller markers for selection
        "height": 0.01,
        "pix": 10.0,
        "degFlatPos": 0.25,
        "deg": 0.25,
        "degFlat": 0.25,
        "cm": 0.25,
    }
    
    _default_line_width = {
        "norm": 4,
        "height": 3,
        "pix": 6,
        "degFlatPos": 2,
        "deg": 2,
        "degFlat": 2,
        "cm": 2,
    }
    
    _default_border_thickness = {
        "norm": 0.01,
        "height": 0.005,
        "pix": 3.0,
        "degFlatPos": 0.1,
        "deg": 0.1,
        "degFlat": 0.1,
        "cm": 0.1,
    }
    
    _default_text_height = {
        "norm": 0.05,        # 5% in normalized units
        "height": 0.025,     # 2.5% of screen height
        "pix": 20.0,         # 20 pixels
        "degFlatPos": 0.5,   # 0.5 degrees
        "deg": 0.5,
        "degFlat": 0.5,
        "cm": 0.5,           # 0.5 cm
    }
    
    def __init__(
        self,
        win,
        infant_stims,
        shuffle=True,
        audio=None,
        focus_time=0.5,
        anim_type='zoom',
        animation_settings=None,
        numkey_dict=None
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
        focus_time : float, optional
            Delay (seconds) before collecting data at a point. Default 0.5.
        anim_type : str, optional
            Animation style: 'zoom' or 'trill'. Default 'zoom'.
        animation_settings : dict
            {'animation_speed':…, 'target_min':…} passed from controller.
        numkey_dict : dict
            Mapping of key strings to point indices passed from controller.
        """
        self.win = win
        self.infant_stims = infant_stims
        self.shuffle = shuffle
        self.audio = audio
        self.focus_time = focus_time
        self.anim_type = anim_type

        # Default settings
        self._animation_settings = animation_settings or {
            'animation_speed': 1.0,
            'target_min': 0.2
        }
        
        self._numkey_dict = numkey_dict or {
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
        
        # Set sizes based on window units
        self.highlight_size = self._default_highlight_size.get(
            self.win.units, self._default_highlight_size["height"]
        )
        self.marker_size = self._default_marker_size.get(
            self.win.units, self._default_marker_size["height"]
        )
        self.line_width = self._default_line_width.get(
            self.win.units, self._default_line_width["height"]
        )
        self.border_thickness = self._default_border_thickness.get(
            self.win.units, self._default_border_thickness["height"]
        )
        self.text_height = self._default_text_height.get(
            self.win.units, self._default_text_height["height"]
        )
        
        self.targets = None
        
        # Create calibration border (red thin border)
        self._create_calibration_border()
    
    
    def _create_calibration_border(self):
        """Create a thin red border to indicate calibration mode."""
        # Get window dimensions
        win_width = self.win.size[0]
        win_height = self.win.size[1]
        
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
            height=self.border_thickness,
            pos=(0, border_height/2 - self.border_thickness/2),
            fillColor='red',
            lineColor=None,
            units=self.win.units  # Use same units as window
        )
        
        self.border_bottom = visual.Rect(
            self.win,
            width=border_width,
            height=self.border_thickness,
            pos=(0, -border_height/2 + self.border_thickness/2),
            fillColor='red',
            lineColor=None,
            units=self.win.units
        )
        
        self.border_left = visual.Rect(
            self.win,
            width=self.border_thickness,
            height=border_height,
            pos=(-border_width/2 + self.border_thickness/2, 0),
            fillColor='red',
            lineColor=None,
            units=self.win.units
        )
        
        self.border_right = visual.Rect(
            self.win,
            width=self.border_thickness,
            height=border_height,
            pos=(border_width/2 - self.border_thickness/2, 0),
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
    
    
    def _create_message_visual(self, formatted_text, pos=(0, -0.15)):
        """
        Create a visual text stimulus from pre-formatted text.
        
        Parameters
        ----------
        formatted_text : str
            Pre-formatted text (e.g., from NicePrint)
        pos : tuple, optional
            Position of the text box (default slightly below center)
            
        Returns
        -------
        visual.TextStim
            PsychoPy text stimulus
        """
        return visual.TextStim(
            self.win,
            text=formatted_text,
            pos=pos,
            color='white',
            height=self._get_text_height(1.2),  # Smaller text for the box
            font='Courier New',  # Monospace font for proper box alignment
            alignText='center',
            anchorHoriz='center',
            anchorVert='center',
            units=self.win.units  # Use same units as window
        )
    
    
    def _get_text_height(self, size_percentage=2.0):
        """
        Calculate text height as a percentage of the base text height.
        Unit-aware text sizing.
        
        Parameters
        ----------
        size_percentage : float
            Multiplier for the base text height. Default is 2.0.
            
        Returns
        -------
        float
            Text height in window units.
        """
        # Scale by the size percentage
        return self.text_height * (size_percentage / 2.0)
    
    
    def show_message_and_wait(self, body, title="", pos=(0, -0.15)):
        """
        Unified function to display a message on screen and in console, then wait for keypress.
        
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
        # Use NicePrint to both print to console AND get formatted text
        formatted_text = NicePrint(body, title)
        
        # Create on-screen message using the formatted text
        message_visual = self._create_message_visual(formatted_text, pos)
        
        # Show message on screen
        self.win.clearBuffer()
        self._draw_calibration_border()
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
        Initialize stimuli sequence.
        """
        self.targets = InfantStimuli(
            self.win,
            self.infant_stims,
            shuffle=self.shuffle
        )
    
    
    def _animate(self, stim, point_idx, clock, rotation_range=15):
        """
        Animate a stimulus with zoom or rotation ('trill').
        """
        elapsed_time = clock.getTime() * self._animation_settings['animation_speed']

        if self.anim_type == 'zoom':
            orig_size = self.targets.get_stim_original_size(point_idx)
            scale_factor = np.sin(elapsed_time)**2 + self._animation_settings['target_min']
            newsize = [scale_factor * s for s in orig_size]
            stim.setSize(newsize)
        elif self.anim_type == 'trill':
            angle = np.sin(elapsed_time) * rotation_range
            stim.setOri(angle)

        stim.draw()
    
    
    def _selection_phase(self, calibration_points, result_img):
        """
        Show results and allow user to select points for retry.
        Uses unit-aware sizing for highlight circles.
        
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
            font='Courier New',
            alignText='center',
            anchorHoriz='center',
            anchorVert='center',
            units=self.win.units
        )
        
        while True:
            # Draw result image, calibration border, and instructions
            result_img.draw()
            self._draw_calibration_border()
            result_instructions_visual.draw()
            
            # Highlight retry points with proper unit-aware sizing
            for retry_idx in retries:
                if retry_idx < len(calibration_points):
                    # Create highlight circle with proper scaling
                    highlight = visual.Circle(
                        self.win,
                        radius=self.highlight_size,           # Unit-aware size
                        pos=calibration_points[retry_idx],
                        lineColor='yellow',                   # Yellow for both types
                        fillColor=None,                       # No fill for consistency
                        lineWidth=self.line_width,            # Unit-aware line width
                        edges=128,                            # smooth circle
                        units=self.win.units                  # Explicit units
                    )
                    highlight.draw()

            self.win.flip()
            
            for key in event.getKeys():
                if key in self._numkey_dict:
                    idx = self._numkey_dict[key]
                    if 0 <= idx < len(calibration_points):
                        if idx in retries:
                            retries.remove(idx)
                        else:
                            retries.add(idx)
                            
                elif key == 'space':
                    return list(retries)  # Accept calibration (with or without retries)
                    
    def _collection_phase(self, calibration_points, **kwargs):
        """
        Unified collection phase for both calibration types.
        Uses callback methods for type-specific data collection.
        
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
                if key in self._numkey_dict:
                    # Select point; play audio if available
                    point_idx = self._numkey_dict[key]
                    if 0 <= point_idx < len(calibration_points):
                        if self.audio:
                            self.audio.play()
                    else:
                        point_idx = -1
                        
                elif key == 'space' and 0 <= point_idx < len(calibration_points):
                    # Collect data using subclass-specific method
                    success = self._collect_data_at_point(
                        calibration_points[point_idx], 
                        point_idx, 
                        **kwargs
                    )
                    if success:
                        if self.audio:
                            self.audio.pause()
                        point_idx = -1
                        
                elif key == 'return':
                    # Finish early - check if we have any data
                    if self._has_collected_data():
                        return True
                        
                elif key == 'escape':
                    # Exit calibration
                    self._clear_collected_data()
                    return False
            
            # Show stimulus at selected point
            if 0 <= point_idx < len(calibration_points):
                stim = self.targets.get_stim(point_idx)
                stim.setPos(calibration_points[point_idx])
                self._animate(stim, point_idx, clock)
            
            # Draw any additional UI elements (e.g., collected markers for mouse)
            self._draw_collection_ui(calibration_points, point_idx)
            
            self.win.flip()
    
    
    def _collect_data_at_point(self, target_pos, point_idx, **kwargs):
        """
        Abstract method for collecting data at a point.
        Must be implemented by subclasses.
        
        Parameters
        ----------
        target_pos : tuple
            Target position (x, y)
        point_idx : int
            Index of the calibration point
        **kwargs : dict
            Additional arguments for collection
            
        Returns
        -------
        bool
            True if data was collected successfully
        """
        raise NotImplementedError("Subclasses must implement _collect_data_at_point")
    
    
    def _has_collected_data(self):
        """
        Abstract method to check if any data has been collected.
        Must be implemented by subclasses.
        
        Returns
        -------
        bool
            True if any data has been collected
        """
        raise NotImplementedError("Subclasses must implement _has_collected_data")
    
    
    def _clear_collected_data(self):
        """
        Abstract method to clear collected data.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _clear_collected_data")
    
    
    def _draw_collection_ui(self, calibration_points, current_point_idx):
        """
        Draw additional UI elements during collection.
        Base implementation does nothing - subclasses can override.
        
        Parameters
        ----------
        calibration_points : list
            List of calibration points
        current_point_idx : int
            Currently selected point index (-1 if none)
        """
        pass  # Base implementation does nothing


class CalibrationSession(BaseCalibrationSession):
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
        focus_time=0.5,
        anim_type='zoom',
        animation_settings=None,
        numkey_dict=None
    ):
        """
        Initialize Tobii calibration session.
        
        Parameters
        ----------
        calibration_api : tobii_research.ScreenBasedCalibration
            Tobii's calibration interface.
        """
        super().__init__(
            win, infant_stims, shuffle, audio, focus_time, 
            anim_type, animation_settings, numkey_dict
        )
        self.calibration = calibration_api

    def run(self, calibration_points, save_calib=False):
        """
        Execute the complete Tobii calibration loop.

        Parameters
        ----------
        calibration_points : list of (float, float)
            PsychoPy-normalized (x, y) coordinates for calibration targets.
        save_calib : bool, optional
            If True and calibration succeeds, save binary data to disk.

        Returns
        -------
        bool
            True if calibration succeeded.
        """
        # Show initial instructions
        instructions_text = """Tobii Eye Tracker Calibration Setup:

• Press number keys (1-9) to select calibration points
• Look at the animated stimulus when it appears
• Press SPACE to collect eye tracking data
• Press ENTER to finish collecting and see results
• Press ESCAPE to exit calibration

Any key will start calibration immediately!"""
        
        self.show_message_and_wait(instructions_text, "Eye Tracker Calibration")

        # 1. Verify and prepare
        self.check_points(calibration_points)
        self._prepare_session(calibration_points)

        # Enter calibration mode
        self.calibration.enter_calibration_mode()

        # Retry loop
        while True:
            # 2. Collection phase
            success = self._collection_phase(calibration_points)
            if not success:
                # User pressed escape during calibration - exit
                self.calibration.leave_calibration_mode()
                return False

            # 3. Compute & show
            self.calibration_result = self.calibration.compute_and_apply()
            result_img = self._show_calibration_result()

            # 4. Selection phase
            retries = self._selection_phase(calibration_points, result_img)
            if retries is None:
                # User pressed escape in results - restart calibration
                continue
            elif not retries:
                # User accepted calibration
                break
            else:
                # 5. Discard phase - retry selected points
                self._discard_phase(calibration_points, retries)

        # Leave calibration mode
        self.calibration.leave_calibration_mode()

        # Return success status
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
        # Check if this point is still in remaining points
        if point_idx not in self.remaining_points:
            return False
            
        # Wait focus time then collect
        core.wait(self.focus_time)
        x, y = get_tobii_pos(self.win, target_pos)
        self.calibration.collect_data(x, y)
        
        # Remove from remaining points
        self.remaining_points.remove(point_idx)
        return True
    
    
    def _has_collected_data(self):
        """
        Check if any Tobii calibration data has been collected.
        
        Returns
        -------
        bool
            True if any points have been collected
        """
        total_points = len(self.remaining_points) + len([p for p in range(9) if p not in self.remaining_points])
        return len(self.remaining_points) < total_points
    
    
    def _clear_collected_data(self):
        """
        Clear Tobii calibration data.
        For Tobii, we reset the remaining points list.
        """
        # Reset remaining points to all points
        self.remaining_points = list(range(len(self.calibration_points)))

    def _discard_phase(self, calibration_points, retries):
        """
        Remove collected data for each retry point.
        """
        for idx in retries:
            x, y = get_tobii_pos(self.win, calibration_points[idx])
            self.calibration.discard_data(x, y)

    def _show_calibration_result(self):
        """
        Show calibration results with lines indicating accuracy.

        Draws markers at each point and lines from each sample:
        green = left eye, red = right eye.

        Returns
        -------
        SimpleImageStim
            PsychoPy stimulus containing the result image.
        """
        # Create blank RGBA image matching window size
        img = Image.new("RGBA", tuple(self.win.size))
        img_draw = ImageDraw.Draw(img)
        # Wrap in SimpleImageStim to avoid resizing later
        result_img = visual.SimpleImageStim(self.win, img, autoLog=False)

        # Only draw if not a full failure
        if self.calibration_result.status != tr.CALIBRATION_STATUS_FAILURE:
            for point in self.calibration_result.calibration_points:
                p = point.position_on_display_area
                # Draw small circle at target
                img_draw.ellipse(
                    ((p[0]*self.win.size[0]-3, p[1]*self.win.size[1]-3),
                     (p[0]*self.win.size[0]+3, p[1]*self.win.size[1]+3)),
                    outline=(0, 0, 0, 255)
                )
                # Draw lines for each calibration sample
                for sample in point.calibration_samples:
                    lp = sample.left_eye.position_on_display_area
                    rp = sample.right_eye.position_on_display_area
                    # Left eye
                    if sample.left_eye.validity == tr.VALIDITY_VALID_AND_USED:
                        img_draw.line(
                            ((p[0]*self.win.size[0], p[1]*self.win.size[1]),
                             (lp[0]*self.win.size[0], lp[1]*self.win.size[1])),
                            fill=(0,255,0,255)
                        )
                    # Right eye
                    if sample.right_eye.validity == tr.VALIDITY_VALID_AND_USED:
                        img_draw.line(
                            ((p[0]*self.win.size[0], p[1]*self.win.size[1]),
                             (rp[0]*self.win.size[0], rp[1]*self.win.size[1])),
                            fill=(255,0,0,255)
                        )

        # Display simple results message in console (no accuracy calculation)
        results_text = f"""Eye Tracker Calibration Results:

Calibration Status: {self.calibration_result.status}
Points Calibrated: {len(self.calibration_result.calibration_points)}

Review the visual results above to assess calibration quality."""

        NicePrint(results_text, "Calibration Results")

        # Update stim's image and return
        result_img.setImage(img)
        return result_img


class SimpleCalibrationSession(BaseCalibrationSession):
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
        focus_time=0.5,
        anim_type='zoom',
        animation_settings=None,
        numkey_dict=None
    ):
        """
        Initialize mouse-based calibration session.
        
        Parameters
        ----------
        mouse : psychopy.event.Mouse
            Mouse object for getting positions.
        """
        super().__init__(
            win, infant_stims, shuffle, audio, focus_time,
            anim_type, animation_settings, numkey_dict
        )
        self.mouse = mouse
        self.calibration_data = {}  # point_idx -> list of (target_pos, sample_pos, timestamp)
    
    
    def run(self, calibration_points, num_samples=5):
        """
        Run the mouse-based calibration procedure.
        
        Parameters
        ----------
        calibration_points : list of (float, float)
            Target positions in PsychoPy coordinates
        num_samples : int
            Number of samples to collect per point
            
        Returns
        -------
        bool
            True if calibration completed successfully
        """
        # Show initial instructions
        instructions_text = """Mouse-Based Calibration Setup:

• Press number keys (1-9) to select calibration points
• Move your mouse to the animated stimulus
• Press SPACE to collect samples at the selected point
• Press ENTER to finish collecting and see results
• Press ESCAPE to exit calibration

Any key will start calibration immediately!"""
        
        self.show_message_and_wait(instructions_text, "Calibration Setup")
        
        # Verify and prepare
        self.check_points(calibration_points)
        self._prepare_session(calibration_points)
        
        # Main calibration loop
        while True:
            # Collection phase
            success = self._collection_phase(calibration_points, num_samples=num_samples)
            if not success:
                # User pressed escape during calibration - exit completely
                return False
                
            # Show results
            result_img = self._show_results(calibration_points)
            
            # Allow user to retry points or accept
            retries = self._selection_phase(calibration_points, result_img)
            
            if retries is None:
                # User pressed escape in results - restart calibration
                self.calibration_data.clear()
                continue
            elif not retries:
                # User accepted calibration
                return True
            else:
                # Remove data for retry points and continue
                for idx in retries:
                    if idx in self.calibration_data:
                        del self.calibration_data[idx]
    
    
    def _collect_data_at_point(self, target_pos, point_idx, **kwargs):
        """
        Collect mouse samples at a calibration point.
        
        Parameters
        ----------
        target_pos : tuple
            Target position (x, y)
        point_idx : int
            Index of the calibration point
        **kwargs : dict
            Must contain 'num_samples' for mouse calibration
            
        Returns
        -------
        bool
            True if samples were collected
        """
        num_samples = kwargs.get('num_samples', 5)
        
        # Wait focus time
        core.wait(self.focus_time)
        
        # Collect samples over a short period
        samples = []
        sample_duration = 1.0  # Collect over 1 second
        sample_interval = sample_duration / num_samples
        
        for i in range(num_samples):
            mouse_pos = self.mouse.getPos()
            timestamp = time.time()
            samples.append((target_pos, mouse_pos, timestamp))
            
            if i < num_samples - 1:  # Don't wait after last sample
                core.wait(sample_interval)
        
        # Store samples
        if point_idx not in self.calibration_data:
            self.calibration_data[point_idx] = []
        self.calibration_data[point_idx].extend(samples)
        
        return True
    
    
    def _has_collected_data(self):
        """
        Check if any mouse calibration data has been collected.
        
        Returns
        -------
        bool
            True if any data has been collected
        """
        return bool(self.calibration_data)
    
    
    def _clear_collected_data(self):
        """
        Clear mouse calibration data.
        """
        self.calibration_data.clear()
    
    
    def _draw_collection_ui(self, calibration_points, current_point_idx):
        """
        Draw additional UI elements during collection.
        Override to remove the green markers for collected points.
        
        Parameters
        ----------
        calibration_points : list
            List of calibration points
        current_point_idx : int
            Currently selected point index (-1 if none)
        """
        pass  # Don't draw any markers - clean interface
    
    
    def _show_results(self, calibration_points):
        """
        Create result visualization without accuracy computation.

        Returns
        -------
        visual.SimpleImageStim
            Image showing calibration results
        """
        # Create result image
        img = Image.new("RGBA", tuple(self.win.size), (128, 128, 128, 255))
        draw = ImageDraw.Draw(img)

        # Draw targets and sample lines (no error calculation)
        for point_idx, samples in self.calibration_data.items():
            target_pos = calibration_points[point_idx]
            target_pix = psychopy_to_pixels(self.win, target_pos)

            # Draw the target point
            draw.ellipse(
                (target_pix[0]-5, target_pix[1]-5, target_pix[0]+5, target_pix[1]+5),
                outline=(0, 0, 0, 255),
                fill=(255, 255, 255, 255)
            )

            for _, sample_pos, _ in samples:
                sample_pix = psychopy_to_pixels(self.win, sample_pos)

                # Draw line from target to sample
                draw.line(
                    (target_pix[0], target_pix[1], sample_pix[0], sample_pix[1]),
                    fill=(255, 0, 0, 200),
                    width=1
                )
                # Draw the sample point
                draw.ellipse(
                    (sample_pix[0]-2, sample_pix[1]-2, sample_pix[0]+2, sample_pix[1]+2),
                    fill=(0, 255, 0, 255)
                )

        # Simple results text without accuracy metrics
        total_samples = sum(len(samples) for samples in self.calibration_data.values())
        results_text = f"""Calibration Results:

Total samples collected: {total_samples}
Points calibrated: {len(self.calibration_data)}

Review the visual results above to assess calibration quality."""
        
        NicePrint(results_text, "Calibration Results")

        # Wrap the PIL image in a PsychoPy stimulus and return
        return visual.SimpleImageStim(self.win, img, autoLog=False)