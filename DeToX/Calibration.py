# Standard library imports
import time
import warnings

import numpy as np
from PIL import Image, ImageDraw

# Third party imports
import tobii_research as tr
from psychopy import core, event, visual

# Local imports
from . import ETSettings as cfg
from .Utils import InfantStimuli, NicePrint
from .Coords import get_tobii_pos, psychopy_to_pixels, tobii2pix, convert_height_to_units, norm_to_window_units, get_psychopy_pos


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
        audio=None,
        anim_type='zoom',
        visualization_style='lines'
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
        self.audio = audio
        self.focus_time = cfg.animation.focus_time
        self.anim_type = anim_type
        self.visualization_style = visualization_style
        
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
    
        
    def show_message_and_wait(self, body, title="", pos=(0, -0.15)):
        """
        Display a message on screen and in console, then wait for keypress.
        
        Shows formatted message both in the PsychoPy window and console output,
        enforces a minimum display time for readability and system stabilization,
        then pauses execution until any key is pressed.
        
        Parameters
        ----------
        body : str
            The main message text to display.
        title : str, optional
            Title for the message box. Default empty string.
        pos : tuple, optional
            Position of the message box center on screen. Default (0, -0.15).
        """
        # --- Console Output ---
        formatted_text = NicePrint(body, title)
        
        # --- Visual Message Creation ---
        message_visual = visual.TextStim(
            self.win,
            text=formatted_text,
            pos=pos,
            color='white',
            height=convert_height_to_units(self.win, cfg.ui_sizes.instruction_text),
            alignText='center',
            anchorHoriz='center',
            anchorVert='center',
            units=self.win.units,
            font='Consolas',
            languageStyle='LTR'
        )
        
        # --- Display and Wait ---
        self.win.clearBuffer()
        message_visual.draw()
        self.win.flip()
        
        # --- Clear any buffered keypresses ---
        event.clearEvents()
        
        # --- Minimum stabilization/readability time ---
        core.wait(0.5)
        
        # --- Wait for intentional keypress ---
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
        )

        # --- Store Calibration Points for Visualization ---
        self.calibration_points = calibration_points  # ← ADDED
        
        # --- Point Tracking Setup ---
        # Initialize remaining points to all points
        self.remaining_points = list(range(len(calibration_points)))
    
        
    def _animate(self, stim, clock, point_idx):
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
            elapsed_time = clock.getTime() * cfg.animation.zoom_speed
            
            # Retrieve and convert size settings from height units to current window units
            min_size_height = cfg.animation.min_zoom_size
            max_size_height = cfg.animation.max_zoom_size
            
            min_size = convert_height_to_units(self.win, min_size_height)
            max_size = convert_height_to_units(self.win, max_size_height)
            
            # Calculate smooth oscillation between min and max sizes using cosine
            # Cosine provides smooth acceleration/deceleration at size extremes
            size_range = max_size - min_size
            normalized_oscillation = (np.cos(elapsed_time) + 1) / 2.0
            current_size = min_size + (normalized_oscillation * size_range)
            
            # Get original aspect ratio
            original_size = self.targets.get_stim_original_size(point_idx)
            aspect_ratio = original_size[0] / original_size[1]  # width / height
            
            # Apply size while maintaining aspect ratio
            if aspect_ratio >= 1.0:
                # Landscape or square: scale based on width
                stim.setSize([current_size, current_size / aspect_ratio])
            else:
                # Portrait: scale based on height
                stim.setSize([current_size * aspect_ratio, current_size])
            
        elif self.anim_type == 'trill':
            # --- Trill Animation: Rapid Rotation with Pauses ---
            # Set fixed size for trill animation from configuration
            trill_size_height = cfg.animation.trill_size
            trill_size = convert_height_to_units(self.win, trill_size_height)
            
            # Get original aspect ratio
            original_size = self.targets.get_stim_original_size(point_idx)
            aspect_ratio = original_size[0] / original_size[1]
            
            # Apply size while maintaining aspect ratio
            if aspect_ratio >= 1.0:
                stim.setSize([trill_size, trill_size / aspect_ratio])
            else:
                stim.setSize([trill_size * aspect_ratio, trill_size])
            
            # Rotation logic (unchanged)
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
    

    def _fade_sound(self, sound, fade_duration=0.5, steps=10):
        """
        Gradually fade out a sound over the specified duration.
        
        Reduces sound volume from its current level to silence over the fade
        duration, then stops playback and resets volume to maximum. Useful for 
        smooth audio transitions during calibration without abrupt cutoffs that 
        might startle participants.
        
        Parameters
        ----------
        sound : psychopy.sound.Sound
            The sound object to fade out. Must be currently playing.
        fade_duration : float, optional
            Total time in seconds for the fade-out effect. Default 0.5 seconds.
        steps : int, optional
            Number of volume steps in the fade. More steps create smoother fade
            but increase CPU overhead. Default 20 steps provides smooth transition
            without performance impact.
            
        Notes
        -----
        The sound is automatically stopped after the fade completes, and volume
        is reset to 1.0 (maximum) so it's ready for the next playback at full volume.
        
        Examples
        --------
        >>> # Fade out calibration audio when point is collected
        >>> self._fade_sound(self.audio, fade_duration=0.5)
        >>> # Next play() will start at full volume
        >>> self.audio.play()
        """
        # --- Fade-out Loop ---
        # Gradually decrease volume from current level to 0
        for i in range(steps, 0, -1):
            sound.setVolume(i / steps)
            core.wait(fade_duration / steps)
        
        # --- Final Cleanup ---
        # Stop sound playback after fade completes
        sound.stop()
        
        # Reset volume to maximum for next playback
        sound.setVolume(1.0)


    def _create_legend_visuals(self, base_y_pos):
        """
        Create legend visual elements showing eye color coding.
        
        Parameters
        ----------
        base_y_pos : float
            Vertical position in window units where legend should be centered.
            
        Returns
        -------
        list
            List of PsychoPy visual elements ready for drawing.
        """
        # Eye shape dimensions (wider than tall for almond shape)
        eye_width = convert_height_to_units(self.win, 0.025)   # Width of eye
        eye_height = convert_height_to_units(self.win, 0.012)  # Height of eye
        
        legend_spacing = convert_height_to_units(self.win, 0.10)  # Horizontal spacing between eyes
        text_offset = convert_height_to_units(self.win, 0.025)   # Vertical offset above eye
        
        # Convert RGBA colors to RGB for PsychoPy (drop alpha channel)
        left_eye_color_rgb = tuple(c / 255.0 for c in cfg.colors.left_eye[:3])   # Normalize to 0-1
        right_eye_color_rgb = tuple(c / 255.0 for c in cfg.colors.right_eye[:3])  # Normalize to 0-1
        
        # Left eye legend (green ellipse + text)
        left_eye_legend = visual.Circle(
            self.win,
            size=(eye_width, eye_height),  # Ellipse shape (width, height)
            pos=(-legend_spacing, base_y_pos),
            fillColor= cfg.colors.left_eye,
            colorSpace='rgb255',
            lineColor='black',
            units=self.win.units
        )
        
        left_eye_text = visual.TextStim(
            self.win,
            text='Left Eye',
            pos=(-legend_spacing, base_y_pos + text_offset),
            color='white',
            height=convert_height_to_units(self.win, cfg.ui_sizes.legend_text),  # ← NEW
            anchorHoriz='center',
            anchorVert='center',
            units=self.win.units,
            font='Consolas'
        )
        
        # Right eye legend (red ellipse + text)
        right_eye_legend = visual.Circle(
            self.win,
            size=(eye_width, eye_height),  # Ellipse shape (width, height)
            pos=(legend_spacing, base_y_pos),
            fillColor= cfg.colors.right_eye,
            colorSpace='rgb255',
            lineColor='black',
            units=self.win.units
        )
        
        right_eye_text = visual.TextStim(
            self.win,
            text='Right Eye',
            pos=(legend_spacing, base_y_pos + text_offset),
            color='white',
            height=convert_height_to_units(self.win, cfg.ui_sizes.legend_text),  # ← NEW
            anchorHoriz='center',
            anchorVert='center',
            units=self.win.units,
            font='Consolas'
        )
        
        return [left_eye_legend, left_eye_text, right_eye_legend, right_eye_text]


    def _selection_phase(self, calibration_points, result_img):
        """
        Show results and allow user to select points for retry.
        
        Returns
        -------
        list or None
            - [] (empty list) to accept calibration
            - [list of indices] to retry those specific points
            - None to restart entire calibration from beginning
        """
        retries = set()
        
        result_instructions ="""Review calibration results above.

    - Press ENTER to accept calibration
    - Press Numbers → SPACE to retry some points
    - Press ESCAPE to restart calibration
    """
        
        formatted_instructions = NicePrint(result_instructions, "Calibration Results")
        result_instructions_visual = visual.TextStim(
            self.win,
            text=formatted_instructions,
            pos=(0, -0.25),
            color='white',
            height=convert_height_to_units(self.win, cfg.ui_sizes.instruction_text),  # ← NEW
            alignText='center',
            anchorHoriz='center',
            anchorVert='center',
            units=self.win.units,
            font='Consolas',
            languageStyle='LTR'
        )
        
        # Create legend positioned below the message
        legend_visuals = self._create_legend_visuals(base_y_pos=-0.37)
        
        while True:
            result_img.draw()
            self._draw_calibration_border()
            result_instructions_visual.draw()
            
            # Draw legend
            for visual_element in legend_visuals:
                visual_element.draw()
            
            for retry_idx in retries:
                if retry_idx < len(calibration_points):
                    highlight_radius = convert_height_to_units(self.win, cfg.ui_sizes.highlight)
                    line_width_pixels = cfg.ui_sizes.line_width * self.win.size[1]
                    
                    highlight = visual.Circle(
                        self.win,
                        radius=highlight_radius,
                        pos=calibration_points[retry_idx],
                        lineColor=cfg.colors.highlight,
                        fillColor=None,
                        lineWidth=max(1, int(line_width_pixels)),
                        edges=128,
                        units=self.win.units
                    )
                    highlight.draw()
            self.win.flip()
            
            for key in event.getKeys():
                if key in cfg.numkey_dict:
                    idx = cfg.numkey_dict[key]
                    if 0 <= idx < len(calibration_points):
                        if idx in retries:
                            retries.remove(idx)
                        else:
                            retries.add(idx)
                            
                elif key == 'return':
                    return []
                    
                elif key == 'space':
                    if retries:
                        return list(retries)
                    else:
                        warnings.warn(
                            "No points selected for retry. "
                            "Press number keys to select points first, or press ENTER to accept calibration.",
                            UserWarning
                        )
                    
                elif key == 'escape':
                    return None

                    
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
                            if not self.audio.isPlaying:
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
                        # DON'T remove from remaining points - allow re-doing same point
                        point_idx = -1
                        
                elif key == 'return':
                    # --- Early Completion ---
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
                self._animate(stim, clock, point_idx)
            
            self.win.flip()
    
 
    def _create_calibration_result_image(self, sample_data):
        """
        Common function to create calibration result visualization.
        
        ALWAYS draws target circles for ALL calibration points from 
        self.calibration_points, then draws sample lines only for points
        that have valid data in sample_data.
        
        Parameters
        ----------
        sample_data : dict
            Dictionary mapping point indices to lists of sample lines:
            {
                point_idx: [
                    (sample_pos_pixels, color),
                    (sample_pos_pixels, color),
                    ...
                ]
            }
            Points not in this dict will show circles only (no lines).
            
        Returns
        -------
        visual.SimpleImageStim
            PsychoPy image stimulus containing the rendered calibration results.
        """
        # --- Image Canvas Creation ---
        img = Image.new("RGBA", tuple(self.win.size))
        img_draw = ImageDraw.Draw(img)
        
        # --- Configuration ---
        line_width_pixels = cfg.ui_sizes.plot_line * self.win.size[1]
        target_circle_radius_pixels = cfg.ui_sizes.target_circle * self.win.size[1]
        target_circle_width_pixels = cfg.ui_sizes.target_circle_width * self.win.size[1]
        
        # --- STEP 1: Draw Samples (Style-Dependent) ---
        
        ## Warning and default to circles if unknown style
        if self.visualization_style not in ['lines', 'circles']:
            warnings.warn(
                f"Unknown visualization style: '{self.visualization_style}'. "
                f"Defaulting to 'circles'.",
                UserWarning
            )
            self.visualization_style = 'circles'

        ## LINES STYLE: Draw lines from targets to samples
        elif self.visualization_style == 'lines':
            for point_idx, samples in sample_data.items():
                if point_idx < len(self.calibration_points):
                    target_pos = self.calibration_points[point_idx]
                    target_pix = psychopy_to_pixels(self.win, target_pos)
                    
                    for sample_pix, line_color in samples:
                        img_draw.line(
                            (target_pix[0], target_pix[1], 
                            sample_pix[0], sample_pix[1]),
                            fill=line_color,
                            width=max(1, int(line_width_pixels))
                        )
       
        ## CIRCLES STYLE: Draw filled circles at sample positions
        elif self.visualization_style == 'circles':
            sample_marker_radius = cfg.ui_sizes.sample_marker * self.win.size[1]
            for point_idx, samples in sample_data.items():
                for sample_pix, fill_color in samples:
                    img_draw.ellipse(
                        (sample_pix[0] - sample_marker_radius,
                        sample_pix[1] - sample_marker_radius,
                        sample_pix[0] + sample_marker_radius,
                        sample_pix[1] + sample_marker_radius),
                        fill=fill_color,
                        outline=None
                    )

        # --- STEP 2: Draw ALL Target Circles (Always) ---
        for point_idx, target_pos in enumerate(self.calibration_points):
            # Convert to pixels
            target_pix = psychopy_to_pixels(self.win, target_pos)
            
            # Draw target circle
            img_draw.ellipse(
                (target_pix[0] - target_circle_radius_pixels, 
                target_pix[1] - target_circle_radius_pixels,
                target_pix[0] + target_circle_radius_pixels, 
                target_pix[1] + target_circle_radius_pixels),
                outline=cfg.colors.target_outline,
                width=max(1, int(target_circle_width_pixels))
            )


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
        audio=None,
        anim_type='zoom',
        visualization_style='lines',  # \u2190 ADD THIS
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
        audio : psychopy.sound.Sound, optional
            Attention-getting sound for point selection feedback. Default None.
        anim_type : str, optional
            Animation style: 'zoom' or 'trill'. Default 'zoom'.
        """
        super().__init__(
            win, infant_stims, audio, anim_type, visualization_style  # \u2190 ADD PARAM
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
            List of calibration points in NORMALIZED coordinates [-1, 1].
            Will be converted to window units and then to Tobii ADCS format.

        Returns
        -------
        bool
            True if calibration was successful and accepted by user, False if
            aborted via escape key or if calibration computation failed.
        """

        # --- 1. Calibration Mode Activation ---
        self.calibration.enter_calibration_mode()

        # --- 2. Instruction Display ---
        instructions_text = f"""Tobii Eye Tracker Calibration Setup:

    - Press number keys (1-{len(calibration_points)}) to select calibration points
    - Look at the animated stimulus when it appears
    - Press SPACE to collect samples at the selected point
    - Press ENTER to finish collecting and see results
    - Press ESCAPE to exit calibration

    Any key will start calibration immediately!"""

        self.show_message_and_wait(instructions_text, "Eye Tracker Calibration")

        # --- 3. Convert from Normalized to Window Units ---
        cal_points_window = norm_to_window_units(self.win, calibration_points)

        # --- 4. Setup and Validation ---
        self.check_points(cal_points_window)
        self._prepare_session(cal_points_window)
        
        # --- 5. Pre-convert to Tobii ADCS Coordinates (Once!) ---
        self.tobii_points = [get_tobii_pos(self.win, pt) for pt in cal_points_window]

        # --- 6. Main Calibration Loop ---
        while True:
            # --- 6a. Data Collection ---
            success = self._collection_phase(cal_points_window)
            if not success:
                self.calibration.leave_calibration_mode()
                return False

            # --- 6b. Calibration Computation ---
            self.calibration_result = self.calibration.compute_and_apply()
            result_img = self._show_calibration_result()

            # --- 6c. User Review and Selection ---
            retries = self._selection_phase(cal_points_window, result_img)
            if retries is None:
                # Restart all: reset remaining points and clear data
                self.remaining_points = list(range(len(cal_points_window)))
                self._clear_collected_data()
                continue
            elif not retries:
                # Accept: finished!
                break
            else:
                # Retry specific points: update remaining points and discard data
                self.remaining_points = retries.copy()
                self._discard_phase(cal_points_window, retries)

        # --- 7. Calibration Mode Deactivation ---
        self.calibration.leave_calibration_mode()

        # --- 8. Final Success Check ---
        success = (self.calibration_result.status == tr.CALIBRATION_STATUS_SUCCESS)

        return success


    def _collect_data_at_point(self, target_pos, point_idx, **kwargs):
        """
        Collect Tobii eye tracking data at a calibration point.
        
        Interfaces with the Tobii SDK to collect gaze samples while the participant
        looks at the calibration target. Uses pre-converted coordinates from
        self.tobii_points for efficiency. Provides focus time either through audio
        fade-out (smooth) or silent wait.
        
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
        # --- Use Pre-converted Coordinates ---
        x, y = self.tobii_points[point_idx]
        
        # --- Data Cleanup ---
        self.calibration.discard_data(x, y)
        
        # --- Focus Time (with or without audio) ---
        if self.audio:
            # Fade audio while participant fixates (dual purpose)
            self._fade_sound(self.audio, fade_duration=self.focus_time)
        else:
            # Silent wait for fixation
            core.wait(self.focus_time)
        
        # --- Data Collection ---
        self.calibration.collect_data(x, y)
        return True
    
    
    def _clear_collected_data(self):
        """
        Clear Tobii calibration data for all points.
        
        Discards previously collected gaze data from all calibration points
        to prepare for a fresh calibration attempt. Uses pre-converted 
        Tobii coordinates for efficiency. Called when user chooses to 
        restart the entire calibration process.
        """
        # --- Clear All Points ---
        # Loop through all points and discard their data
        for idx in range(len(self.tobii_points)):
            x, y = self.tobii_points[idx]
            self.calibration.discard_data(x, y)

    def _discard_phase(self, calibration_points, retries):
        """
        Remove collected data for each retry point.
        
        Discards previously collected gaze data for points that the user
        wants to recalibrate. Uses pre-converted coordinates for efficiency.
        
        Parameters
        ----------
        calibration_points : list of (float, float)
            Full list of calibration target coordinates.
        retries : list of int
            Indices of points to retry, whose data should be discarded.
        """
        # --- Selective Data Removal ---
        for idx in retries:
            x, y = self.tobii_points[idx]
            self.calibration.discard_data(x, y)


    def _show_calibration_result(self):
        """
        Show Tobii calibration results.
        
        Extracts sample data from Tobii results and builds line data only.
        Circles are drawn automatically by base class from self.calibration_points.
        
        Returns
        -------
        SimpleImageStim
            PsychoPy stimulus containing the rendered calibration results image.
        """
        # --- Initialize Sample Data (lines only) ---
        sample_data = {}
        
        # --- Extract Lines from Tobii Results ---
        if self.calibration_result.status != tr.CALIBRATION_STATUS_FAILURE:
            for point in self.calibration_result.calibration_points:
                # Find which point_idx this corresponds to
                target_adcs = point.position_on_display_area
                
                # Match to original points by finding closest ADCS coordinate
                for point_idx in range(len(self.tobii_points)):
                    if (abs(self.tobii_points[point_idx][0] - target_adcs[0]) < 0.01 and
                        abs(self.tobii_points[point_idx][1] - target_adcs[1]) < 0.01):
                        
                        # Collect sample lines for this point
                        samples = []
                        for sample in point.calibration_samples:
                            # Left eye
                            if sample.left_eye.validity == tr.VALIDITY_VALID_AND_USED:
                                left_adcs = sample.left_eye.position_on_display_area
                                left_psychopy = get_psychopy_pos(self.win, left_adcs)
                                left_pix = psychopy_to_pixels(self.win, left_psychopy)
                                samples.append((left_pix, cfg.colors.left_eye))

                            # Right eye
                            if sample.right_eye.validity == tr.VALIDITY_VALID_AND_USED:
                                right_adcs = sample.right_eye.position_on_display_area
                                right_psychopy = get_psychopy_pos(self.win, right_adcs)
                                right_pix = psychopy_to_pixels(self.win, right_psychopy)
                                samples.append((right_pix, cfg.colors.right_eye))
                        
                        # Store lines (if any)
                        if samples:
                            sample_data[point_idx] = samples
                        
                        break  # Found the match, move to next point
        
        # --- Generate Visualization ---
        # Base class will draw ALL circles + lines from sample_data
        return self._create_calibration_result_image(sample_data)


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
        audio=None,
        anim_type='zoom',
        visualization_style='lines',  # \u2190 ADD THIS
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
        audio : psychopy.sound.Sound, optional
            Attention-getting sound for point selection feedback. Default None.
        anim_type : str, optional
            Animation style: 'zoom' or 'trill'. Default 'zoom'.
        """
        # --- Base Class Initialization ---
        super().__init__(
            win, infant_stims, audio, anim_type, visualization_style  # \u2190 ADD PARAM
        )
        
        # --- Mouse-Specific Setup ---
        self.mouse = mouse
        self.calibration_data = {}  # point_idx -> list of (target_pos, sample_pos, timestamp)
    

    def run(self, calibration_points):
        """
        Main function to run the mouse-based calibration routine.
        
        Executes the complete calibration workflow using mouse position as a
        proxy for gaze data. Follows the same interaction pattern as Tobii
        calibration to ensure consistency across modes.

        Parameters
        ----------
        calibration_points : list of (float, float)
            List of calibration points in NORMALIZED coordinates [-1, 1].
            Will be converted to window units automatically.

        Returns
        -------
        bool
            True if calibration finished successfully and was accepted by user,
            False if the user exits early via escape key.
        """

        # --- 1. Instruction Display ---
        instructions_text = f"""Mouse-Based Calibration Setup:

    - Press number keys (1-{len(calibration_points)}) to select calibration points
    - Move your mouse to the animated stimulus
    - Press SPACE to collect samples at the selected point
    - Press ENTER to finish collecting and see results
    - Press ESCAPE to exit calibration

    Any key will start calibration immediately!"""
        
        self.show_message_and_wait(instructions_text, "Calibration Setup")
        
        # --- 2. Convert from Normalized to Window Units ---
        # Import here to avoid circular imports
        cal_points_window = norm_to_window_units(self.win, calibration_points)
        
        # --- 3. Setup and Validation ---
        self.check_points(cal_points_window)
        self._prepare_session(cal_points_window)
        
        # --- 4. Main Calibration Loop ---
        while True:
            # --- 5a. Data Collection ---
            success = self._collection_phase(cal_points_window, num_samples=cfg.calibration.num_samples_mouse)
            if not success:
                return False
                
            # --- 4b. Results Visualization ---
            result_img = self._show_results(cal_points_window)
            
            # --- 4c. User Review and Selection ---
            retries = self._selection_phase(cal_points_window, result_img)
            
            if retries is None:
                self.remaining_points = list(range(len(cal_points_window)))
                self.calibration_data.clear()
                continue
            elif not retries:
                return True
            else:
                self.remaining_points = retries.copy()
                for idx in retries:
                    if idx in self.calibration_data:
                        del self.calibration_data[idx]
    
    
    def _collect_data_at_point(self, target_pos, point_idx, **kwargs):
        """
        Collect mouse samples at a single calibration target.
        
        Gathers multiple mouse position samples over a brief period to simulate
        the variability of real gaze data. Samples are distributed over time
        to capture any mouse movement or positioning adjustments. Provides focus
        time either through audio fade-out (smooth) or silent wait.

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
        if point_idx in self.calibration_data:
            del self.calibration_data[point_idx]

        # --- Sampling Configuration ---
        num_samples = kwargs.get('num_samples', 5)

        # --- Focus Time (with or without audio) ---
        if self.audio:
            # Fade audio while participant fixates (dual purpose)
            self._fade_sound(self.audio, fade_duration=self.focus_time)
        else:
            # Silent wait for fixation
            core.wait(self.focus_time)

        # --- Sample Collection Setup ---
        samples = []
        sample_duration = 1.0
        sample_interval = sample_duration / num_samples

        # --- Mouse Position Sampling ---
        for i in range(num_samples):
            mouse_pos = self.mouse.getPos()
            timestamp = time.time()
            samples.append((target_pos, mouse_pos, timestamp))

            if i < num_samples - 1:
                core.wait(sample_interval)

        # --- Data Storage ---
        if point_idx not in self.calibration_data:
            self.calibration_data[point_idx] = []
        self.calibration_data[point_idx].extend(samples)

        return True


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
        Visualize mouse calibration results.
        
        Builds sample line data only. Circles are drawn automatically 
        by base class from self.calibration_points.
        
        Parameters
        ----------
        calibration_points : list of (float, float)
            Calibration targets in window units (kept for API compatibility).
            
        Returns
        -------
        visual.SimpleImageStim
            PsychoPy image stimulus with calibration results.
        """
        # --- Initialize Sample Data (lines only) ---
        sample_data = {}

        # --- Extract Lines from Mouse Data ---
        for point_idx, samples in self.calibration_data.items():
            formatted_samples = []
            
            for _, sample_pos, _ in samples:  # (target, sample, timestamp)
                # Convert sample to pixels
                sample_pix = psychopy_to_pixels(self.win, sample_pos)
                formatted_samples.append((sample_pix, cfg.colors.mouse))
            
            # Store lines
            if formatted_samples:
                sample_data[point_idx] = formatted_samples

        # --- Generate Visualization ---
        # Base class will draw ALL circles + lines from sample_data
        return self._create_calibration_result_image(sample_data)
