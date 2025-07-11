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
from .Coords import get_tobii_pos


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
    
    
    def _create_niceprint_visual(self, body, title="", pos=(0, -0.15)):
        """
        Create a visual text stimulus that mimics NicePrint formatting.
        
        Parameters
        ----------
        body : str
            The main text content
        title : str, optional
            Title for the box
        pos : tuple, optional
            Position of the text box (default slightly below center)
            
        Returns
        -------
        visual.TextStim
            PsychoPy text stimulus with NicePrint-style formatting
        """
        # Split the body string into lines
        lines = body.splitlines() or [""]
        
        # Calculate the maximum width of the lines
        content_w = max(map(len, lines))
        
        # Calculate the panel width
        title_space = f" {title} " if title else ""
        panel_w = max(content_w, len(title_space)) + 2
        
        # Unicode characters for the corners and sides of the box
        tl, tr, bl, br, h, v = "┌", "┐", "└", "┘", "─", "│"
        
        # Construct the top border of the box
        if title:
            # Calculate the left and right margins for the title
            left = (panel_w - len(title_space)) // 2
            right = panel_w - len(title_space) - left
            # Construct the top border with title
            top = f"{tl}{h * left}{title_space}{h * right}{tr}"
        else:
            # Construct the top border without title
            top = f"{tl}{h * panel_w}{tr}"
        
        # Create the middle lines with content
        middle_lines = [
            f"{v}{line}{' ' * (panel_w - len(line))}{v}"
            for line in lines
        ]
        
        # Create the bottom border
        bottom = f"{bl}{h * panel_w}{br}"
        
        # Combine all lines
        full_text = "\n".join([top] + middle_lines + [bottom])
        
        # Create and return the visual text stimulus
        return visual.TextStim(
            self.win,
            text=full_text,
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
    
    
    def _psychopy_to_pixels(self, pos):
        """Convert PsychoPy coordinates to pixel coordinates."""
        if self.win.units == 'height':
            # Convert height units to pixels
            x_pix = (pos[0] * self.win.size[1] + self.win.size[0]/2)
            y_pix = (-pos[1] * self.win.size[1] + self.win.size[1]/2)
        elif self.win.units == 'norm':
            # Convert normalized units to pixels
            x_pix = (pos[0] + 1) * self.win.size[0] / 2
            y_pix = (1 - pos[1]) * self.win.size[1] / 2
        else:
            # Handle other units - assume they're already close to pixels
            x_pix = pos[0] + self.win.size[0]/2
            y_pix = -pos[1] + self.win.size[1]/2
        
        return (int(x_pix), int(y_pix))


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
        # Display formatted instructions in console and on screen
        instructions_text = """Tobii Eye Tracker Calibration Setup:

• Press number keys (1-9) to select calibration points
• Look at the animated stimulus when it appears
• Press SPACE to collect eye tracking data
• Press ENTER to finish collecting and see results
• Press ESCAPE to exit calibration

Any key will start calibration immediately!"""
        
        NicePrint(instructions_text, "Eye Tracker Calibration")
        
        # Create on-screen instructions
        instructions_visual = self._create_niceprint_visual(
            instructions_text, 
            "Eye Tracker Calibration",
            pos=(0, -0.15)  # Slightly below center
        )
        
        # Show instructions on screen
        self.win.clearBuffer()
        self._draw_calibration_border()
        instructions_visual.draw()
        self.win.flip()
        
        # Wait for user to acknowledge (any key press)
        event.waitKeys()

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

        # 6. Save calibration data if requested
        success = (self.calibration_result.status == tr.CALIBRATION_STATUS_SUCCESS)
        if success and save_calib:
            data = self.calibration.retrieve_calibration_data()
            fname = f"{datetime.now():%Y-%m-%d_%H-%M-%S}_calibration.dat"
            with open(fname, 'wb') as f:
                f.write(data)
            
            save_text = f"""Eye tracker calibration completed successfully!

Calibration data has been saved to:
{fname}

You can now proceed with your experiment."""
            
            NicePrint(save_text, "Calibration Complete")
        else:
            completion_text = """Eye tracker calibration completed successfully!

You can now proceed with your experiment."""
            
            NicePrint(completion_text, "Calibration Complete")
        
        return success

    def _collection_phase(self, calibration_points):
        """
        Let user select points by number and collect data on Space.

        Returns
        -------
        bool
            True if collection completed, False if user pressed escape
        """
        clock = core.Clock()
        cp_num = len(calibration_points)
        remaining = list(range(cp_num))
        point_idx = -1

        while True:
            # Clear screen and draw calibration border
            self.win.clearBuffer()
            self._draw_calibration_border()
            
            # Handle keys
            for key in event.getKeys():
                if key in self._numkey_dict:
                    # select point; play audio if available
                    point_idx = self._numkey_dict[key]
                    if self.audio:
                        self.audio.play()
                elif key == 'space' and point_idx in remaining:
                    # wait, collect, pause audio
                    core.wait(self.focus_time)
                    x, y = get_tobii_pos(self.win, calibration_points[point_idx])
                    self.calibration.collect_data(x, y)
                    if self.audio:
                        self.audio.pause()
                    remaining.remove(point_idx)
                    point_idx = -1
                elif key == 'return':
                    # finish early
                    return True
                elif key == 'escape':
                    # exit calibration
                    return False

            # Animate selected stim
            if 0 <= point_idx < cp_num:
                stim = self.targets.get_stim(point_idx)
                stim.setPos(calibration_points[point_idx])
                self._animate(
                    stim, point_idx, clock,
                    rotation_range=15
                )

            self.win.flip()

    def _selection_phase(self, calibration_points, result_img):
        """
        Show result image; toggle retry points with number keys.
        Confirm with Space; abort (retry all) with Escape.

        Returns
        -------
        list or None
            List of indices to retry, empty list to accept, None to restart all
        """
        cp_num = len(calibration_points)
        retries = set()

        while True:
            # Draw result image, calibration border, and instructions
            result_img.draw()
            self._draw_calibration_border()
            self.result_instructions_visual.draw()

            for key in event.getKeys():
                if key in self._numkey_dict:
                    idx = self._numkey_dict[key]
                    if 0 <= idx < cp_num:
                        if idx in retries:
                            retries.remove(idx)
                        else:
                            retries.add(idx)
                elif key == 'space':
                    return list(retries)  # Accept calibration (with or without retries)
                elif key == 'escape':
                    return None  # Signal to restart entire calibration

            # highlight retries with unit-aware sizing
            for rp in retries:
                visual.Circle(
                    self.win,
                    radius=self.highlight_size,
                    pos=calibration_points[rp],
                    lineColor='yellow',
                    lineWidth=self.line_width,
                    units=self.win.units
                ).draw()
            self.win.flip()

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

        # Display results in console
        results_text = f"""Eye Tracker Calibration Results:

Calibration Status: {self.calibration_result.status}
Points Calibrated: {len(self.calibration_result.calibration_points)}

Next Steps:
• Press SPACE to accept this calibration
• Press numbers (1-9) to select points for retry
• Press ESCAPE to redo entire calibration"""

        NicePrint(results_text, "Calibration Results")

        # Create on-screen results instructions
        result_instructions = """Review eye tracking calibration above.

• Press SPACE to accept calibration
• Press numbers to select points for retry  
• Press ESCAPE to restart calibration

Make your choice now:"""
        
        self.result_instructions_visual = self._create_niceprint_visual(
            result_instructions,
            "Calibration Results",
            pos=(0, -0.25)  # Lower position
        )

        # Update stim's image and return
        result_img.setImage(img)
        return result_img


class SimpleCalibrationSession(BaseCalibrationSession):
    """
    Mouse-based calibration session for simulation mode.
    
    This runs independently of Tobii and collects mouse position
    samples at calibration targets, then calculates accuracy.
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
        # Display formatted instructions in console
        instructions_text = """Mouse-Based Calibration Setup:

• Press number keys (1-9) to select calibration points
• Move your mouse to the animated stimulus
• Press SPACE to collect samples at the selected point
• Press ENTER to finish collecting and see results
• Press ESCAPE to exit calibration

Any key will start calibration immediately!"""
        
        NicePrint(instructions_text, "Calibration Setup")
        
        # Create on-screen instructions
        instructions_visual = self._create_niceprint_visual(
            instructions_text, 
            "Calibration Setup",
            pos=(0, -0.15)  # Slightly below center
        )
        
        # Show instructions on screen
        self.win.clearBuffer()
        self._draw_calibration_border()
        instructions_visual.draw()
        self.win.flip()
        
        # Wait for user to acknowledge (any key press)
        event.waitKeys()
        
        # Verify and prepare
        self.check_points(calibration_points)
        self._prepare_session(calibration_points)
        
        # Main calibration loop
        while True:
            # Collection phase
            success = self._collection_phase(calibration_points, num_samples)
            if not success:
                # User pressed escape during calibration - exit completely
                return False
                
            # Show results
            result_img = self._calculate_and_show_results(calibration_points)
            
            # Allow user to retry points or accept
            retries = self._selection_phase(calibration_points, result_img)
            
            if retries is None:
                # User pressed escape in results - restart calibration
                self.calibration_data.clear()
                continue
            elif not retries:
                # User accepted calibration
                completion_text = """Calibration procedure completed successfully!

The calibration data has been collected and 
accuracy results have been calculated.

You can now proceed with your experiment."""
                
                NicePrint(completion_text, "Calibration Complete")
                return True
            else:
                # Remove data for retry points and continue
                for idx in retries:
                    if idx in self.calibration_data:
                        del self.calibration_data[idx]
    
    
    def _collection_phase(self, calibration_points, num_samples):
        """
        Collect mouse samples at calibration points.
        Uses unit-aware markers for collected points.
        
        Returns
        -------
        bool
            True if collection completed, False if escaped
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
                    point_idx = self._numkey_dict[key]
                    if 0 <= point_idx < len(calibration_points):
                        if self.audio:
                            self.audio.play()
                    else:
                        point_idx = -1
                        
                elif key == 'space' and 0 <= point_idx < len(calibration_points):
                    samples_collected = self._collect_samples_at_point(
                        calibration_points[point_idx], 
                        point_idx, 
                        num_samples
                    )
                    if samples_collected:
                        if self.audio:
                            self.audio.pause()
                        point_idx = -1
                        
                elif key == 'return':
                    if self.calibration_data:
                        return True
                        
                elif key == 'escape':
                    self.calibration_data.clear()
                    return False
            
            # Show stimulus at selected point
            if 0 <= point_idx < len(calibration_points):
                stim = self.targets.get_stim(point_idx)
                stim.setPos(calibration_points[point_idx])
                self._animate(stim, point_idx, clock)
            
            # Show all collected points as unit-aware markers
            for collected_idx in self.calibration_data:
                if collected_idx != point_idx:  # Don't overlay on animated stimulus
                    marker = visual.Circle(
                        self.win,
                        radius=self.marker_size,              # Unit-aware size
                        pos=calibration_points[collected_idx],
                        lineColor='green',
                        fillColor=None,
                        lineWidth=max(1, self.line_width // 2),  # Thinner line for markers
                        units=self.win.units                  # Explicit units
                    )
                    marker.draw()
            
            self.win.flip()
    
    
    def _collect_samples_at_point(self, target_pos, point_idx, num_samples):
        """
        Collect multiple mouse samples at a calibration point.
        
        Parameters
        ----------
        target_pos : tuple
            Target position (x, y)
        point_idx : int
            Index of the calibration point
        num_samples : int
            Number of samples to collect
            
        Returns
        -------
        bool
            True if samples were collected
        """
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
    
    
    def _calculate_and_show_results(self, calibration_points):
        """
        Calculate accuracy and create result visualization.

        Returns
        -------
        visual.SimpleImageStim
            Image showing calibration results
        """
        # Create result image
        img = Image.new("RGBA", tuple(self.win.size), (128, 128, 128, 255))
        draw = ImageDraw.Draw(img)

        total_error = 0.0
        total_samples = 0

        # Draw targets and sample lines, accumulate error
        for point_idx, samples in self.calibration_data.items():
            target_pos = calibration_points[point_idx]
            target_pix = self._psychopy_to_pixels(target_pos)

            # Draw the target point
            draw.ellipse(
                (target_pix[0]-5, target_pix[1]-5, target_pix[0]+5, target_pix[1]+5),
                outline=(0, 0, 0, 255),
                fill=(255, 255, 255, 255)
            )

            errors = []
            for _, sample_pos, _ in samples:
                sample_pix = self._psychopy_to_pixels(sample_pos)
                # Pixel-distance error
                err = np.hypot(target_pix[0] - sample_pix[0],
                               target_pix[1] - sample_pix[1])
                errors.append(err)

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

            total_error += sum(errors)
            total_samples += len(errors)

        # Calculate overall mean error
        overall_mean_error = (total_error / total_samples) if total_samples > 0 else float('nan')

        # Build and print results text
        results_text = f"""Calibration Results:

Overall mean error: {overall_mean_error:.1f} pixels
Total samples collected: {total_samples}
Points calibrated: {len(self.calibration_data)}

Next Steps:
• Press SPACE to accept this calibration
• Press numbers (1-9) to select points for retry
• Press ESCAPE to redo entire calibration"""
        NicePrint(results_text, "Accuracy Results")

        # On-screen instructions
        result_instructions = """Review calibration accuracy above.

• Press SPACE to accept calibration
• Press numbers to select points for retry  
• Press ESCAPE to restart calibration

Make your choice now:"""
        self.result_instructions_visual = self._create_niceprint_visual(
            result_instructions,
            "Calibration Results",
            pos=(0, -0.25)
        )

        # Wrap the PIL image in a PsychoPy stimulus and return
        return visual.SimpleImageStim(self.win, img, autoLog=False)

    
    def _selection_phase(self, calibration_points, result_img):
        """
        Show results and allow user to select points for retry.
        Uses unit-aware sizing for highlight circles.
        
        Returns
        -------
        list or None
            List of indices to retry, empty list to accept, None to restart all
        """
        retries = set()
        
        while True:
            # Draw result image, calibration border, and instructions
            result_img.draw()
            self._draw_calibration_border()
            self.result_instructions_visual.draw()
            
            # Highlight retry points with proper unit-aware sizing
            for retry_idx in retries:
                if retry_idx < len(calibration_points):
                    # Create highlight circle with proper scaling
                    highlight = visual.Circle(
                        self.win,
                        radius=self.highlight_size,           # Unit-aware size
                        pos=calibration_points[retry_idx],
                        lineColor=(0.5, 0.5, 0, 1.0),        # dark olive outline
                        fillColor=(0.5, 0.5, 0, 0.8),        # dark olive with transparency
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
                    
                elif key == 'escape':
                    return None  # Signal to restart entire calibration