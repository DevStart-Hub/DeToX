# Standard library imports
import time
import numpy as np
from PIL import Image, ImageDraw

# Third party imports
import tobii_research as tr
from psychopy import core, event, visual

# Local imports
from . import ETSettings as cfg
from .Utils import InfantStimuli, NicePrint
from .Coords import get_tobii_pos, psychopy_to_pixels, convert_height_to_units


class BaseCalibrationSession:
    """
    Base class with common functionality for both calibration types.
    
    This abstract base class provides shared calibration functionality for both
    Tobii hardware-based and mouse-based simulation calibration sessions. It handles
    visual presentation, user interaction, animation, and result visualization while
    delegating hardware-specific data collection to subclasses.
    
    The class implements an infant-friendly calibration protocol with animated stimuli,
    optional audio feedback, and interactive point selection. It provides a consistent
    interface for calibration regardless of whether real eye tracking hardware or
    mouse simulation is being used.
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
        Initialize base calibration session with common parameters.
        
        Sets up visual elements, animation settings, and stimulus management
        that are shared between both Tobii and mouse calibration modes. Creates
        the red calibration border and prepares stimulus presentation system.
        
        Parameters
        ----------
        win : psychopy.visual.Window
            PsychoPy window for rendering stimuli and instructions. Used for
            all visual presentation and coordinate system conversions.
        infant_stims : list of str
            List of image file paths for attention-getting stimuli. These should
            be engaging images suitable for infant participants (e.g., cartoon
            characters, colorful objects).
        shuffle : bool, optional
            Whether to randomize stimulus order each run. This prevents habituation
            to a fixed sequence. Default True.
        audio : psychopy.sound.Sound, optional
            Sound to play when user selects a calibration point. Provides auditory
            feedback during the calibration process. Default None.
        anim_type : str, optional
            Animation style for calibration targets:
            - 'zoom': Smooth size oscillation using cosine function
            - 'trill': Rapid rotation with intermittent stops
            Default 'zoom'.
        """
        # --- Core Attributes ---
        # Store window and stimulus configuration
        self.win = win
        self.infant_stims = infant_stims
        self.shuffle = shuffle
        self.audio = audio
        self.focus_time = cfg.animation.focus_time
        self.anim_type = anim_type
        
        # --- State Management ---
        # Initialize calibration state variables
        self.targets = None
        self.remaining_points = []  # Track which points still need calibration
        
        # --- Visual Setup ---
        # Create calibration border (red thin border)
        self._create_calibration_border()
    
    
    def _create_calibration_border(self):
        """
        Create a thin red border to indicate calibration mode.
        
        Constructs four rectangular segments forming a border around the entire
        window. The border thickness is automatically scaled based on window units
        to maintain consistent appearance across different display configurations.
        This visual indicator helps experimenters confirm calibration mode is active.
        """
        # --- Window Dimension Retrieval ---
        # Get window dimensions
        win_width = self.win.size[0]
        win_height = self.win.size[1]
        
        # --- Border Scaling ---
        # Convert border thickness from height units to current units
        border_thickness = convert_height_to_units(self.win, cfg.ui_sizes.border)
        
        # --- Unit-Specific Dimension Conversion ---
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
        
        # --- Border Segment Creation ---
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
        """
        Draw the red calibration border.
        
        Renders all four border segments to the current window buffer. This method
        should be called during each frame refresh while in calibration mode to
        maintain the visual indicator.
        """
        self.border_top.draw()
        self.border_bottom.draw()
        self.border_left.draw()
        self.border_right.draw()
    
    
    def _create_message_visual(self, formatted_text, pos=(0, -0.15), font_type="instruction_text"):
        """
        Create a visual text stimulus from pre-formatted text.
        
        Generates a PsychoPy TextStim object with consistent formatting for
        displaying instructions and messages during calibration. Uses monospace
        font to preserve text alignment and box-drawing characters.
        
        Parameters
        ----------
        formatted_text : str
            Pre-formatted text string, typically from NicePrint utility. Should
            include any box-drawing characters and formatting.
        pos : tuple, optional
            Position of the text box center in window units. Default (0, -0.15)
            places text slightly below center.
        font_type : str, optional
            Type of font sizing to use from cfg.FONT_SIZE_MULTIPLIERS dictionary.
            Options include 'instruction_text', 'small_text', etc. Default
            'instruction_text'.
            
        Returns
        -------
        visual.TextStim
            Configured PsychoPy text stimulus ready for drawing.
        """
        # --- Font Configuration ---
        # Get font and size multiplier from configuration
        size_multiplier = getattr(cfg.font_multipliers, font_type)

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
        
        Converts base text height from configuration to current window units
        and applies scaling factor for different text sizes. Ensures consistent
        text sizing across different display configurations.
        
        Parameters
        ----------
        size_percentage : float
            Multiplier for the base text height. Standard value is 2.0.
            Smaller values produce smaller text, larger values produce larger text.
            
        Returns
        -------
        float
            Text height in current window units, scaled by the size percentage.
        """
        # --- Base Height Conversion ---
        # Get base text height from config and convert to current units
        base_height = convert_height_to_units(self.win, cfg.DEFAULT_TEXT_HEIGHT_HEIGHT)
        
        # --- Scaling Application ---
        # Scale by the size percentage (normalized to 2.0 as baseline)
        return base_height * (size_percentage / 2.0)
    
    
    def show_message_and_wait(self, body, title="", pos=(0, -0.15)):
        """
        Display a message on screen and in console, then wait for keypress.
        
        Shows formatted message both in the PsychoPy window and console output,
        then pauses execution until any key is pressed. Useful for instructions
        and status messages during calibration.
        
        Parameters
        ----------
        body : str
            The main message text to display. Will be formatted with box-drawing
            characters via NicePrint.
        title : str, optional
            Title for the message box. Appears at the top of the formatted box.
            Default empty string.
        pos : tuple, optional
            Position of the message box center on screen in window units.
            Default (0, -0.15) places message slightly below center.
            
        Returns
        -------
        None
        """
        # --- Console Output ---
        # Use NicePrint to print to console AND get formatted text
        formatted_text = NicePrint(body, title)
        
        # --- Visual Message Creation ---
        # Create on-screen message using the formatted text
        message_visual = self._create_message_visual(formatted_text, pos)
        
        # --- Display and Wait ---
        # Show message on screen
        self.win.clearBuffer()
        message_visual.draw()
        self.win.flip()
        
        # Wait for any key press
        event.waitKeys()
    
    
    def check_points(self, calibration_points):
        """
        Ensure number of calibration points is within allowed range.
        
        Validates that the provided calibration points fall within the
        supported range for infant calibration protocols. Both Tobii and
        simulation modes support 2-9 calibration points.
        
        Parameters
        ----------
        calibration_points : list
            List of calibration point coordinates to validate.
            
        Raises
        ------
        ValueError
            If number of points is less than 2 or greater than 9.
        """
        if not (2 <= len(calibration_points) <= 9):
            raise ValueError("Calibration points must be between 2 and 9")
    
    
    def _prepare_session(self, calibration_points):
        """
        Initialize stimuli sequence and remaining points list.
        
        Sets up the stimulus presentation system and initializes tracking
        of which calibration points still need data collection. Called at
        the start of each calibration attempt.
        
        Parameters
        ----------
        calibration_points : list
            List of calibration point coordinates for this session.
        """
        # --- Stimulus System Initialization ---
        self.targets = InfantStimuli(
            self.win,
            self.infant_stims,
            shuffle=self.shuffle
        )
        
        # --- Point Tracking Setup ---
        # Initialize remaining points to all points
        self.remaining_points = list(range(len(calibration_points)))
    
    
    def _animate(self, stim, clock):
        """
        Animate a stimulus with zoom or rotation ('trill') effects.
        
        Uses height-based settings that are automatically converted to current window units
        for consistent visual appearance across different screen configurations. Supports
        two animation types designed to maintain infant attention during calibration.
        
        Parameters
        ----------
        stim : psychopy.visual stimulus
            The stimulus object to animate. Must support setSize() and setOri() methods.
        clock : psychopy.core.Clock
            Clock object for timing animations. Used to calculate animation phase
            and control oscillation speed.
            
        Notes
        -----
        Animation timing is controlled by cfg.ANIMATION_SETTINGS.
        Size settings are defined in height units and automatically converted to window units.
        Supported animation types: 'zoom' (cosine oscillation), 'trill' (discrete rotation pulses).
        """
        
        if self.anim_type == 'zoom':
            # --- Zoom Animation: Smooth Size Oscillation ---
            # Calculate elapsed time with zoom-specific speed multiplier
            elapsed_time = clock.getTime() * cfg.ANIMATION_SETTINGS['zoom_speed']
            
            # Retrieve and convert size settings from height units to current window units
            min_size_height = cfg.animation.min_zoom_size
            max_size_height = cfg.animation.max_zoom_size
            
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
            # --- Trill Animation: Rapid Rotation with Pauses ---
            # Set fixed size for trill animation from configuration
            trill_size_height = cfg.animation.trill_size
            trill_size = convert_height_to_units(self.win, trill_size_height)
            stim.setSize([trill_size, trill_size])
            
            # Create rapid trill and stop pattern
            elapsed_time = clock.getTime()
            trill_cycle_duration = cfg.animation.trill_cycle_duration
            trill_active_duration = cfg.animation.trill_active_duration
            
            # Determine position in the cycle
            cycle_position = elapsed_time % trill_cycle_duration
            
            if cycle_position < trill_active_duration:
                # --- TRILL PHASE: Rapid back-and-forth oscillations ---
                
                # Create rapid oscillations using high-frequency sine wave
                trill_frequency = cfg.animation.trill_frequency  # Oscillations per second
                trill_time = cycle_position * trill_frequency * 2 * np.pi
                
                # Create sharp, rapid back-and-forth movement
                rotation_base = np.sin(trill_time)
                
                # Apply rotation range
                rotation_angle = rotation_base * cfg.animation.trill_rotation_range
                stim.setOri(rotation_angle)
                
            else:
                # --- STOP PHASE: No rotation ---
                stim.setOri(0)
        
        # --- Render Animated Stimulus ---
        stim.draw()
    

    def _selection_phase(self, calibration_points, result_img):
        """
        Show results and allow user to select points for retry.
        
        Displays calibration results overlaid with interactive controls for
        selecting which points to retry. Uses height-based sizing for highlight
        circles to maintain consistent appearance across displays.
        
        Parameters
        ----------
        calibration_points : list of (float, float)
            List of calibration point coordinates in window units.
        result_img : visual.SimpleImageStim
            Image showing calibration results with lines from targets to samples.
        
        Returns
        -------
        list or None
            - List of point indices to retry (may be empty to accept all)
            - None to restart entire calibration from beginning
        """
        # --- Selection State Initialization ---
        retries = set()
        
        # --- Instructions Creation ---
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
        
        # --- Interactive Selection Loop ---
        while True:
            # --- Frame Rendering ---
            # Draw result image, calibration border, and instructions
            result_img.draw()
            self._draw_calibration_border()
            result_instructions_visual.draw()
            
            # --- Retry Point Highlighting ---
            # Highlight retry points with height-based sizing
            for retry_idx in retries:
                if retry_idx < len(calibration_points):
                    # Convert highlight size and line width from height units
                    highlight_radius = convert_height_to_units(self.win, cfg.ui_sizes.highlight)
                    line_width_height = cfg.ui_sizes.line_width
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
            
            # --- User Input Processing ---
            for key in event.getKeys():
                if key in cfg.numkey_dict:
                    # --- Point Selection Toggle ---
                    idx = cfg.numkey_dict[key]
                    if 0 <= idx < len(calibration_points):
                        if idx in retries:
                            retries.remove(idx)
                        else:
                            retries.add(idx)
                            
                elif key == 'space':
                    # --- Accept Current Results ---
                    return list(retries)  # Accept calibration (with or without retries)
                    
                elif key == 'escape':
                    # --- Restart All Calibration ---
                    return None  # Restart all calibration
                    
    def _collection_phase(self, calibration_points, **kwargs):
        """
        Unified collection phase for both calibration types.
        
        Uses callback methods for type-specific data collection while providing
        common interaction logic. Only allows interaction with points in the
        remaining_points list to prevent redundant calibration.
        
        Parameters
        ----------
        calibration_points : list of (float, float)
            List of calibration point coordinates in window units.
        **kwargs : dict
            Additional arguments passed to collect method. Mode-specific parameters
            such as 'num_samples' for mouse calibration.
        
        Returns
        -------
        bool
            True if collection completed successfully, False if user pressed escape
            to abort calibration.
        """
        # --- Animation Timing Setup ---
        clock = core.Clock()
        point_idx = -1
        
        # --- Main Collection Loop ---
        while True:
            # --- Frame Setup ---
            # Clear screen and draw calibration border
            self.win.clearBuffer()
            self._draw_calibration_border()
            
            # --- Keyboard Input Processing ---
            for key in event.getKeys():
                if key in cfg.numkey_dict:
                    # --- Point Selection ---
                    # Select point; play audio if available
                    candidate_idx = cfg.numkey_dict[key]
                    # Only allow selection of points that are still remaining
                    if candidate_idx in self.remaining_points:
                        point_idx = candidate_idx
                        if self.audio:
                            self.audio.play()
                    else:
                        # Ignore key press for points not in remaining list
                        point_idx = -1
                        
                elif key == 'space' and point_idx in self.remaining_points:
                    # --- Data Collection Trigger ---
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
                    # --- Early Completion ---
                    # Finish early - check if we have any data
                    if self._has_collected_data():
                        return True
                        
                elif key == 'escape':
                    # --- Abort Calibration ---
                    self._clear_collected_data()
                    return False
            
            # --- Stimulus Presentation ---
            # Show stimulus at selected point (only if it's in remaining points)
            if point_idx in self.remaining_points:
                stim = self.targets.get_stim(point_idx)
                stim.setPos(calibration_points[point_idx])
                self._animate(stim, clock)
            
            self.win.flip()
    
    
    def _create_calibration_result_image(self, calibration_points, sample_data):
        """
        Common function to create calibration result visualization.
        
        Generates a visual representation of calibration quality by drawing
        lines from each target to collected gaze samples. Uses color coding
        to distinguish left eye (green), right eye (red), and mouse (orange)
        data. Line length indicates calibration accuracy.
        
        Parameters
        ----------
        calibration_points : list of (float, float)
            List of calibration target coordinates in window units.
        sample_data : dict
            Dictionary mapping point indices to lists of (target_pos, sample_pos, color)
            tuples. Each tuple represents one gaze sample with its deviation from target.
            
        Returns
        -------
        visual.SimpleImageStim
            PsychoPy image stimulus containing the rendered calibration results.
        """
        # --- Image Canvas Creation ---
        # Create blank RGBA image matching window size (transparent background like Tobii)
        img = Image.new("RGBA", tuple(self.win.size))
        img_draw = ImageDraw.Draw(img)
        
        # --- Line Width Configuration ---
        # Convert plot line width from height units to pixels
        line_width_pixels = cfg.ui_sizes.plot_line * self.win.size[1]
        
        # --- Target Circle Configuration ---
        # Convert target circle size and line width from height units to pixels
        target_circle_radius_pixels = cfg.ui_sizes.target_circle * self.win.size[1]
        target_circle_width_pixels = cfg.ui_sizes.target_circle_width * self.win.size[1]
        
        # --- Calibration Data Rendering ---
        # Draw calibration data for each point
        for point_idx, samples in sample_data.items():
            if point_idx < len(calibration_points):
                # --- Target Position Processing ---
                target_pos = calibration_points[point_idx]
                target_pix = psychopy_to_pixels(self.win, target_pos)
                
                # --- Target Circle Drawing ---
                # Draw target circle with configurable size and line width
                img_draw.ellipse(
                    (target_pix[0] - target_circle_radius_pixels, 
                    target_pix[1] - target_circle_radius_pixels,
                    target_pix[0] + target_circle_radius_pixels, 
                    target_pix[1] + target_circle_radius_pixels),
                    outline=cfg.colors.target_outline,
                    width=max(1, int(target_circle_width_pixels))  # Ensure minimum 1 pixel width
                )
                
                # --- Sample Lines Drawing ---
                # Draw lines from target to samples
                for _, sample_pos, line_color in samples:
                    sample_pix = psychopy_to_pixels(self.win, sample_pos)
                    img_draw.line(
                        (target_pix[0], target_pix[1], sample_pix[0], sample_pix[1]),
                        fill=line_color,
                        width=max(1, int(line_width_pixels))  # Ensure minimum 1 pixel width
                    )
        
        # --- Image Stimulus Creation ---
        # Wrap in SimpleImageStim and return
        return visual.SimpleImageStim(self.win, img, autoLog=False)


class TobiiCalibrationSession(BaseCalibrationSession):
    """
    Tobii-based calibration session for real eye tracking.
    
    This class implements the calibration protocol for physical Tobii eye trackers,
    extending the base calibration functionality with hardware-specific data collection
    and validation. It interfaces directly with the Tobii Pro SDK to collect gaze
    samples, compute calibration models, and visualize tracking accuracy.
    
    The Tobii calibration process involves presenting targets at known positions,
    collecting gaze data while participants look at these targets, and computing
    a mapping between eye features and screen coordinates. This class provides an
    infant-friendly implementation with animated stimuli and interactive controls.
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
        
        Sets up the calibration interface for a connected Tobii eye tracker,
        inheriting common functionality from the base class while adding
        hardware-specific calibration API access.
        
        Parameters
        ----------
        win : psychopy.visual.Window
            PsychoPy window for stimulus presentation and coordinate conversions.
        calibration_api : tobii_research.ScreenBasedCalibration
            Tobii's calibration interface object, pre-configured for the connected
            eye tracker. This handles the low-level calibration data collection.
        infant_stims : list of str
            Paths to engaging image files for calibration targets.
        shuffle : bool, optional
            Whether to randomize stimulus presentation order. Default True.
        audio : psychopy.sound.Sound, optional
            Attention-getting sound for point selection feedback. Default None.
        anim_type : str, optional
            Animation style: 'zoom' or 'trill'. Default 'zoom'.
        """
        # --- Base Class Initialization ---
        super().__init__(
            win, infant_stims, shuffle, audio, anim_type
        )
        
        # --- Tobii-Specific Setup ---
        self.calibration = calibration_api

    def run(self, calibration_points):
        """
        Main routine to run the full Tobii calibration workflow.

        This function presents each calibration target, collects gaze data
        via the eye tracker, shows the results, and allows the user to retry
        any subset of points until satisfied.

        Parameters
        ----------
        calibration_points : list of (float, float)
            List of PsychoPy (x, y) positions for calibration targets. Typically
            5-9 points distributed across the screen for comprehensive coverage.

        Returns
        -------
        bool
            True if calibration was successful and accepted by user, False if
            aborted via escape key or if calibration computation failed.
        """

        # --- 1. Instruction Display ---
        # Show instructions before anything happens
        instructions_text = """Tobii Eye Tracker Calibration Setup:

    • Press number keys (1-9) to select calibration points
    • Look at the animated stimulus when it appears
    • Press SPACE to collect eye tracking data
    • Press ENTER to finish collecting and see results
    • Press ESCAPE to exit calibration

    Any key will start calibration immediately!"""

        self.show_message_and_wait(instructions_text, "Eye Tracker Calibration")

        # --- 2. Setup and Validation ---
        # Initial verification and preparation
        self.check_points(calibration_points)
        self._prepare_session(calibration_points)

        # --- 3. Calibration Mode Activation ---
        # Enter Tobii calibration mode
        self.calibration.enter_calibration_mode()

        # --- 4. Main Calibration Loop ---
        # Main calibration-retry loop
        while True:
            # --- 4a. Data Collection ---
            # Data collection phase
            success = self._collection_phase(calibration_points)
            if not success:
                self.calibration.leave_calibration_mode()
                return False

            # --- 4b. Calibration Computation ---
            # Compute and show calibration results
            self.calibration_result = self.calibration.compute_and_apply()
            result_img = self._show_calibration_result(calibration_points)

            # --- 4c. User Review and Selection ---
            # Let user select points to retry
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

        # --- 5. Calibration Mode Deactivation ---
        # Exit calibration mode
        self.calibration.leave_calibration_mode()

        # --- 6. Final Success Check ---
        success = (self.calibration_result.status == tr.CALIBRATION_STATUS_SUCCESS)

        return success


    def _collect_data_at_point(self, target_pos, point_idx, **kwargs):
        """
        Collect Tobii eye tracking data at a calibration point.
        
        Interfaces with the Tobii SDK to collect gaze samples while the participant
        looks at the calibration target. Converts coordinates from PsychoPy to
        Tobii's coordinate system and manages the timing of data collection.
        
        Parameters
        ----------
        target_pos : tuple
            Target position in PsychoPy coordinates (window units).
        point_idx : int
            Index of the calibration point being collected.
        **kwargs : dict
            Unused for Tobii calibration, included for interface compatibility.
            
        Returns
        -------
        bool
            Always returns True to indicate data collection was initiated.
            Actual success is determined during calibration computation.
        """
        # --- Coordinate Conversion ---
        # Convert from PsychoPy to Tobii ADCS coordinates
        x, y = get_tobii_pos(self.win, target_pos)
        
        # --- Data Cleanup ---
        # Clear any existing data at this point first
        self.calibration.discard_data(x, y)
        
        # --- Data Collection ---
        # Wait focus time then collect NEW data
        core.wait(self.focus_time)
        self.calibration.collect_data(x, y)
        return True

    
    
    def _has_collected_data(self):
        """
        Check if any Tobii calibration data has been collected.
        
        Determines whether the calibration session has accumulated any gaze
        data by comparing the current remaining points to the initial set.
        Used to validate early termination requests.
        
        Returns
        -------
        bool
            True if any calibration points have been collected, False if
            no data has been gathered yet.
        """
        # --- Collection Status Check ---
        # If remaining points is smaller than total points, we've collected some data
        return len(self.remaining_points) < len(range(9))  # Assuming max 9 points
    
    
    def _clear_collected_data(self):
        """
        Clear Tobii calibration data.
        
        Resets the calibration state for a fresh start. For Tobii hardware,
        data clearing is handled internally by the SDK when calibration mode
        is re-entered or when specific points are discarded.
        """
        # --- Tobii Internal Handling ---
        # Tobii handles clearing internally
        pass

    def _discard_phase(self, calibration_points, retries):
        """
        Remove collected data for each retry point.
        
        Discards previously collected gaze data for points that the user
        wants to recalibrate. This ensures fresh data collection without
        interference from potentially poor previous samples.
        
        Parameters
        ----------
        calibration_points : list of (float, float)
            Full list of calibration target coordinates.
        retries : list of int
            Indices of points to retry, whose data should be discarded.
        """
        # --- Selective Data Removal ---
        for idx in retries:
            x, y = get_tobii_pos(self.win, calibration_points[idx])
            self.calibration.discard_data(x, y)

    def _show_calibration_result(self, calibration_points):
        """
        Show Tobii calibration results using the common plotting function.
        
        Processes the calibration computation results from the Tobii SDK and
        creates a visual representation showing the accuracy of gaze estimation
        at each calibration point. Green lines indicate left eye samples, red
        lines indicate right eye samples.

        Parameters
        ----------
        calibration_points : list of (float, float)
            List of calibration point coordinates in window units.

        Returns
        -------
        SimpleImageStim
            PsychoPy stimulus containing the rendered calibration results image.
        """
        # --- Sample Data Preparation ---
        # Prepare sample data in common format
        sample_data = {}
        
        # --- Result Processing ---
        # Only process if not a full failure
        if self.calibration_result.status != tr.CALIBRATION_STATUS_FAILURE:
            for point_idx, point in enumerate(self.calibration_result.calibration_points):
                samples = []
                
                # --- Sample Extraction ---
                # Process each calibration sample
                for sample in point.calibration_samples:
                    target_pos = point.position_on_display_area
                    
                    # --- Left Eye Processing ---
                    # Left eye sample (green)
                    if sample.left_eye.validity == tr.VALIDITY_VALID_AND_USED:
                        left_pos = sample.left_eye.position_on_display_area
                        samples.append((target_pos, left_pos, cfg.CALIBRATION_COLORS['left_eye']))
                    
                    # --- Right Eye Processing ---
                    # Right eye sample (red)
                    if sample.right_eye.validity == tr.VALIDITY_VALID_AND_USED:
                        right_pos = sample.right_eye.position_on_display_area
                        samples.append((target_pos, right_pos, cfg.CALIBRATION_COLORS['right_eye']))
                
                if samples:
                    sample_data[point_idx] = samples
        
        # --- Visualization Generation ---
        # Use common plotting function
        return self._create_calibration_result_image(calibration_points, sample_data)


class MouseCalibrationSession(BaseCalibrationSession):
    """
    Mouse-based calibration session for simulation mode.
    
    This class provides a calibration interface for testing and development when
    no physical eye tracker is available. It simulates the calibration process
    using mouse position as a proxy for gaze, allowing experimenters to test
    calibration procedures and develop experiments without hardware.
    
    The mouse calibration follows the same interaction pattern as Tobii calibration,
    collecting position samples at each calibration target and visualizing the
    results. This ensures consistent user experience between simulation and real
    data collection modes.
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
        
        Sets up the simulation calibration interface using mouse input as a
        stand-in for eye tracking data. Inherits common functionality from
        the base class while adding mouse-specific data collection.
        
        Parameters
        ----------
        win : psychopy.visual.Window
            PsychoPy window for stimulus presentation and coordinate conversions.
        infant_stims : list of str
            Paths to engaging image files for calibration targets.
        mouse : psychopy.event.Mouse
            Mouse object for getting position samples. Should be configured
            for the same window used for display.
        shuffle : bool, optional
            Whether to randomize stimulus presentation order. Default True.
        audio : psychopy.sound.Sound, optional
            Attention-getting sound for point selection feedback. Default None.
        anim_type : str, optional
            Animation style: 'zoom' or 'trill'. Default 'zoom'.
        """
        # --- Base Class Initialization ---
        super().__init__(
            win, infant_stims, shuffle, audio, anim_type
        )
        
        # --- Mouse-Specific Setup ---
        self.mouse = mouse
        self.calibration_data = {}  # point_idx -> list of (target_pos, sample_pos, timestamp)
    
    
    def run(self, calibration_points, num_samples=5):
        """
        Main function to run the mouse-based calibration routine.
        
        Executes the complete calibration workflow using mouse position as a
        proxy for gaze data. Follows the same interaction pattern as Tobii
        calibration to ensure consistency across modes.

        Parameters
        ----------
        calibration_points : list of (float, float)
            List of target positions in PsychoPy coordinates. Typically 5-9
            points distributed across the screen.
        num_samples : int
            How many mouse position samples to collect at each calibration point.
            More samples provide smoother averaging. Default 5.

        Returns
        -------
        bool
            True if calibration finished successfully and was accepted by user,
            False if the user exits early via escape key.
        """

        # --- 1. Instruction Display ---
        # Show the instructions screen
        instructions_text = """Mouse-Based Calibration Setup:

    • Press number keys (1-9) to select calibration points
    • Move your mouse to the animated stimulus
    • Press SPACE to collect samples at the selected point
    • Press ENTER to finish collecting and see results
    • Press ESCAPE to exit calibration

    Any key will start calibration immediately!"""
        
        self.show_message_and_wait(instructions_text, "Calibration Setup")
        
        # --- 2. Setup and Validation ---
        # Sanity check and prepare stimuli
        self.check_points(calibration_points)
        self._prepare_session(calibration_points)
        
        # --- 3. Main Calibration Loop ---
        while True:
            # --- 3a. Data Collection ---
            # Collect calibration data at each point
            success = self._collection_phase(calibration_points, num_samples=num_samples)
            if not success:
                return False
                
            # --- 3b. Results Visualization ---
            # Show results of current calibration
            result_img = self._show_results(calibration_points)
            
            # --- 3c. User Review and Selection ---
            # Let user review and pick points to retry
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
        
        Gathers multiple mouse position samples over a brief period to simulate
        the variability of real gaze data. Samples are distributed over time
        to capture any mouse movement or positioning adjustments.

        Parameters
        ----------
        target_pos : tuple
            The (x, y) coordinates of the current calibration target in PsychoPy units.
        point_idx : int
            The index of this calibration point in the full list.
        **kwargs : dict
            Must contain 'num_samples': number of mouse samples to collect.
            Typically 5-10 samples for reasonable averaging.

        Returns
        -------
        bool
            Always returns True to indicate samples were collected successfully.
        """

        # --- Existing Data Cleanup ---
        # Clear existing data for this point first (Option 2: Replace)
        if point_idx in self.calibration_data:
            del self.calibration_data[point_idx]

        # --- Sampling Configuration ---
        # How many mouse samples to take at this point? (Default: 5)
        num_samples = kwargs.get('num_samples', 5)

        # --- Focus Period ---
        # Wait a moment before sampling
        core.wait(self.focus_time)

        # --- Sample Collection Setup ---
        # Setup: collect all samples in this list
        samples = []
        sample_duration = 1.0  # Total time (seconds) over which to collect samples
        sample_interval = sample_duration / num_samples  # Time between samples

        # --- 1. Mouse Position Sampling ---
        # Collect mouse samples over a brief period
        for i in range(num_samples):
            mouse_pos = self.mouse.getPos()      # Get current mouse position (x, y)
            timestamp = time.time()              # Record current time
            samples.append((target_pos, mouse_pos, timestamp))

            # Don't wait after the final sample
            if i < num_samples - 1:
                core.wait(sample_interval)

        # --- 2. Data Storage ---
        # Store the collected samples
        if point_idx not in self.calibration_data:
            self.calibration_data[point_idx] = []
        self.calibration_data[point_idx].extend(samples)

        # --- 3. Collection Complete ---
        # Done! Return True to indicate success
        return True

        
        
    def _has_collected_data(self):
        """
        Check if any mouse calibration data has been collected yet.
        
        Determines whether the calibration session has accumulated any mouse
        position samples. Used to validate early termination requests and
        ensure at least some data exists before allowing completion.

        Returns
        -------
        bool
            True if there is any calibration data in storage,
            False if no samples have been collected yet.
        """
        # --- Data Presence Check ---
        return bool(self.calibration_data)


    def _clear_collected_data(self):
        """
        Remove all previously collected mouse calibration data.
        
        Clears the calibration data dictionary to prepare for a fresh
        calibration attempt. Called when user chooses to restart the
        entire calibration process.
        """
        # --- Data Dictionary Reset ---
        self.calibration_data.clear()


    def _show_results(self, calibration_points):
        """
        Visualize and return a summary image of the collected mouse calibration data.
        
        Creates a visual representation of calibration quality by drawing lines
        from each target to the collected mouse samples. Orange lines indicate
        mouse position samples, with line length showing the deviation from target.

        Parameters
        ----------
        calibration_points : list of (float, float)
            The (x, y) positions of all calibration targets in window units.

        Returns
        -------
        visual.SimpleImageStim
            A PsychoPy image stimulus with the plotted calibration results,
            ready for display in the selection phase.
        """
        # --- Data Structure Preparation ---
        # Prepare data for plotting
        sample_data = {}

        # --- Sample Formatting ---
        for point_idx, samples in self.calibration_data.items():
            formatted_samples = []
            for target_pos, sample_pos, _ in samples:
                # Draw a line from the target to each sample; use orange color for mouse samples
                formatted_samples.append((target_pos, sample_pos, cfg.CALIBRATION_COLORS['mouse']))
            if formatted_samples:
                sample_data[point_idx] = formatted_samples

        # --- Visualization Generation ---
        # Use the shared plotting function
        return self._create_calibration_result_image(calibration_points, sample_data)