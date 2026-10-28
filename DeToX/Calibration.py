# Standard library imports
from concurrent.futures import ThreadPoolExecutor
import random
import time
import warnings

import numpy as np
from PIL import Image, ImageDraw

# Third party imports
import tobii_research as tr
from psychopy import core, event, visual

# Local imports
from . import ETSettings as cfg
from .Utils import NicePrint
from .Coords import get_tobii_pos, psychopy_to_pixels, convert_height_to_units, get_psychopy_pos, norm_to_window_units, get_content_scale


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
        collection_anim_type='zoom',
        stim_size='big',
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
        infant_stims : list of psychopy.visual or list of str
            Calibration target stimuli. Can be:
            - List of PsychoPy visual objects (ImageStim, Circle, Rect, Polygon, ShapeStim)
            - List of image file paths (strings)
            - Single visual object (will be wrapped in list)
            - Single file path string (will be wrapped in list)
            These should be engaging stimuli suitable for the target population.
        audio : psychopy.sound.Sound, optional
            Sound to play when user selects a calibration point. Provides auditory
            feedback during the calibration process. Default None (no sound).
        anim_type : {'zoom', 'trill'}, optional
            Animation style for attracting attention before collection:
            - 'zoom': Smooth size oscillation using cosine function
            - 'trill': Rapid rotation with intermittent stops
            Default 'zoom'.
        collection_anim_type : {'zoom', 'trill'}, optional
            Animation style for settling and sample collection. Kept separate
            from ``anim_type`` so the two phases can use different animations.
            Default 'zoom'.
        stim_size : {'big', 'small'}, optional
            Size preset for calibration stimuli. Attention and collection
            sizes are configured independently in ``cfg.animation``:
            - 'big': Larger stimuli recommended for infants/children (default)
            * Attention zoom: 9.5-16% of screen height
            * Collection zoom: 5-9.5% of screen height
            * Attention trill: 12.75%; collection trill: 7.25%
            - 'small': Smaller stimuli recommended for adults
            * Attention zoom: 3-6% of screen height
            * Collection zoom: 1-3% of screen height
            * Attention trill: 4.5%; collection trill: 2%
            Default 'big'.
        visualization_style : {'lines', 'circles'}, optional
            Style for calibration result visualization:
            - 'lines': Draw lines from targets to gaze samples (default)
            - 'circles': Draw circles at sample positions
            Default 'lines'.
        """
        # --- Core Attributes ---
        # Store window and stimulus configuration
        self.win = win
        self.infant_stims = infant_stims
        self.audio = audio
        self.focus_time = cfg.animation.focus_time
        self.anim_type = anim_type
        self.collection_anim_type = collection_anim_type
        self.stim_size = stim_size  # ← Store stim_size
        self.visualization_style = visualization_style
        
        # --- State Management ---
        # Initialize calibration state variables
        self.stim_objects = None
        self.remaining_points = []  # Track which points still need calibration
        self._defer_collection_abort = False
        
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
        
        border_width = win_width / win_height  # Full width in height units
        border_height = 1.0  # Full height in height units

        
        # --- Border Segment Creation ---
        # Create four rectangles for the border
        self.border_top = visual.Rect(
            self.win,
            width=border_width,
            height=cfg.ui_sizes.border,
            pos=(0, border_height/2 - cfg.ui_sizes.border/2),
            fillColor='red',
            lineColor=None,
            units= 'height'
        )
        
        self.border_bottom = visual.Rect(
            self.win,
            width=border_width,
            height=cfg.ui_sizes.border,
            pos=(0, -border_height/2 + cfg.ui_sizes.border/2),
            fillColor='red',
            lineColor=None,
            units= 'height'
        )
        
        self.border_left = visual.Rect(
            self.win,
            width=cfg.ui_sizes.border,
            height=border_height,
            pos=(-border_width/2 + cfg.ui_sizes.border/2, 0),
            fillColor='red',
            lineColor=None,
            units= 'height'
        )
        
        self.border_right = visual.Rect(
            self.win,
            width=cfg.ui_sizes.border,
            height=border_height,
            pos=(border_width/2 - cfg.ui_sizes.border/2, 0),
            fillColor='red',
            lineColor=None,
            units= 'height'
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
    
        
    def show_message_and_wait(self, body, title="", pos=(0, -0.15), key_list=None):
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
        key_list : list of str, optional
            Accepted keys. Default None accepts any key. Returns the pressed keys.
        """
        # --- Console Output ---
        formatted_text = NicePrint(body, title, self.verbose)
        
        # --- Visual Message Creation ---
        message_visual = visual.TextStim(
            self.win,
            text=formatted_text,
            pos=pos,
            color='white',
            height= cfg.ui_sizes.instruction_text,
            alignText='center',
            anchorHoriz='center',
            anchorVert='center',
            units='height', #self.win.units,
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
        return event.waitKeys(keyList=key_list)

    def _show_auto_paced_instructions(self, looking_instruction):
        """Wait for Space to start, or Escape to exit before calibration begins."""
        instructions = f"""Auto-Paced Calibration:

    - Press SPACE to begin
    - {looking_instruction}
    - Press SPACE to collect samples at the current point
    - The next target appears automatically
    - Results appear after all points are collected
    - Press ESCAPE to exit calibration"""
        keys = self.show_message_and_wait(
            instructions, "Calibration Setup", key_list=['space', 'escape']
        )
        return 'escape' not in keys
            
    
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
        of which calibration points still need data collection. Handles both
        file paths (strings) and pre-loaded PsychoPy shape stimuli.
        
        Parameters
        ----------
        calibration_points : list
            List of calibration point coordinates for this session.
        """
        # --- Stimulus System Initialization ---
        # Check if stimuli are already PsychoPy visual objects or need to be loaded
        if self.infant_stims and hasattr(self.infant_stims[0], 'draw'):
            # Validate that it's an allowed stimulus type
            allowed_types = (visual.ImageStim, visual.Circle, visual.Rect, 
                            visual.Polygon, visual.ShapeStim)
            
            if not isinstance(self.infant_stims[0], allowed_types):
                raise TypeError(
                    f"Pre-loaded stimuli must be ImageStim, Circle, Rect, Polygon, or ShapeStim. "
                    f"Got: {type(self.infant_stims[0]).__name__}"
                )
            
            # Already PsychoPy shape stimuli - use directly
            self.stim_objects = self.infant_stims
        else:
            # File paths - load as ImageStim
            self.stim_objects = [
                visual.ImageStim(self.win, image=path, units='height', interpolate=True) 
                for path in self.infant_stims
            ]

        # --- Store Stimulus Units ---
        self.calstim_units = self.stim_objects[0].units  # Store units of stimuli

        # --- Convert Calibration Points ---
        self.calibration_points = norm_to_window_units(self.win, calibration_points, target_units=self.calstim_units)
        
        # --- Store the target points in pixels ---
        self.calibration_targets_pixels = psychopy_to_pixels(self.win, self.calibration_points, source_units='height')

        # --- Point Tracking Setup ---
        self.remaining_points = list(range(len(calibration_points)))
    
        
    def _zoom_animation(self, elapsed_time, min_size, max_size):
        """
        Calculate a smooth zoom pose.

        The target starts at its maximum size, shrinks to its minimum size,
        then grows again. Attention and collection use the same motion with
        different bounds configured in ``ETSettings``.

        Parameters
        ----------
        elapsed_time : float
            Time in seconds since the current animation phase began.
        min_size : float
            Smallest target size in the stimulus's units.
        max_size : float
            Largest target size in the stimulus's units.

        Returns
        -------
        tuple of float
            Current size and rotation in degrees. Zoom never rotates, so the
            rotation is always zero.
        """

        # --- Smooth Size Oscillation ---
        # A cosine starts at the maximum size and changes direction smoothly
        # at both ends, avoiding abrupt jumps in size.
        scaled_time = elapsed_time * cfg.animation.zoom_speed
        normalized_oscillation = (np.cos(scaled_time) + 1) / 2.0
        current_size = min_size + normalized_oscillation * (max_size - min_size)

        # Returning zero explicitly prevents an old trill angle from remaining
        # on the stimulus when switching to zoom.
        return current_size, 0.0

    def _trill_animation(self, elapsed_time, size, rotation_range):
        """
        Calculate a fixed-size trill pose.

        A trill is a burst of back-and-forth rotation followed by a brief
        pause. The rotation fades in and out smoothly so the target does not
        snap back to upright at the start or end of a burst.

        Parameters
        ----------
        elapsed_time : float
            Time in seconds since the current animation phase began.
        size : float
            Fixed target size in the stimulus's units.
        rotation_range : float
            Maximum rotation in degrees for this animation phase.

        Returns
        -------
        tuple of float
            Fixed size and current rotation in degrees.
        """

        # --- Locate the Current Rotation Burst ---
        cycle_position = elapsed_time % cfg.animation.trill_cycle_duration
        rotation = 0.0

        if cycle_position < cfg.animation.trill_active_duration:
            # --- Smooth Start and End of the Burst ---
            # The sine-squared envelope reaches zero at both ends of the
            # active period, leaving the target upright during the pause.
            burst_progress = (
                cycle_position / cfg.animation.trill_active_duration
            )
            movement_strength = np.sin(np.pi * burst_progress) ** 2

            # --- Back-and-Forth Rotation ---
            trill_time = (
                cycle_position
                * cfg.animation.trill_frequency
                * 2
                * np.pi
            )
            rotation = (
                np.sin(trill_time)
                * rotation_range
                * movement_strength
            )

        return size, rotation

    def _animate(self, stim, clock, point_idx, phase='attention'):
        """
        Animate a calibration target for the current presentation phase.

        Attention and collection use independent animation types and sizes
        from ``ETSettings``. The phase is selected by the presentation loop:
        attention while waiting for the experimenter, collection while the
        participant settles and gaze samples are acquired.

        Parameters
        ----------
        stim : psychopy.visual stimulus
            The stimulus object to animate.
        clock : psychopy.core.Clock
            Clock for timing the current animation phase.
        point_idx : int
            Index of the current calibration point.
        phase : {'attention', 'collection'}, optional
            Animation phase to present. Default is 'attention'.
        """

        if phase not in ('attention', 'collection'):
            raise ValueError(
                f"Invalid animation phase: '{phase}'. "
                "Must be 'attention' or 'collection'."
            )

        # --- Select Phase Animation and Size Settings ---
        if phase == 'attention':
            animation_type = self.anim_type
            if self.stim_size == 'big':
                zoom_min_size = cfg.animation.attention_zoom_min_size_big
                zoom_max_size = cfg.animation.attention_zoom_max_size_big
                trill_size = cfg.animation.attention_trill_size_big
            else:
                zoom_min_size = cfg.animation.attention_zoom_min_size_small
                zoom_max_size = cfg.animation.attention_zoom_max_size_small
                trill_size = cfg.animation.attention_trill_size_small
            trill_rotation_range = (
                cfg.animation.attention_trill_rotation_range
            )
        else:
            animation_type = self.collection_anim_type
            if self.stim_size == 'big':
                zoom_min_size = cfg.animation.collection_zoom_min_size_big
                zoom_max_size = cfg.animation.collection_zoom_max_size_big
                trill_size = cfg.animation.collection_trill_size_big
            else:
                zoom_min_size = cfg.animation.collection_zoom_min_size_small
                zoom_max_size = cfg.animation.collection_zoom_max_size_small
                trill_size = cfg.animation.collection_trill_size_small
            trill_rotation_range = (
                cfg.animation.collection_trill_rotation_range
            )

        # --- Calculate Current Pose ---
        elapsed_time = clock.getTime()

        if animation_type == 'zoom':
            # Convert height-unit settings to the stimulus's own units.
            min_size = convert_height_to_units(
                self.win,
                zoom_min_size,
                target_units=stim.units,
            )
            max_size = convert_height_to_units(
                self.win,
                zoom_max_size,
                target_units=stim.units,
            )
            size, rotation = self._zoom_animation(
                elapsed_time,
                min_size,
                max_size,
            )

        elif animation_type == 'trill':
            size = convert_height_to_units(
                self.win,
                trill_size,
                target_units=stim.units,
            )
            size, rotation = self._trill_animation(
                elapsed_time,
                size,
                trill_rotation_range,
            )

        else:
            raise ValueError(
                f"Invalid animation type: '{animation_type}'. "
                "Must be 'zoom' or 'trill'."
            )

        # --- Apply Pose and Draw ---
        stim.setSize(size)
        stim.setOri(rotation)
        stim.draw()
    

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
        # Left eye legend (green ellipse + text)
        left_eye_legend = visual.Circle(
            self.win,
            size=(cfg.ui_sizes.legend_eye_width, cfg.ui_sizes.legend_eye_height),  # Ellipse shape (width, height)
            pos=(-cfg.ui_sizes.legend_spacing, base_y_pos),
            fillColor= cfg.colors.left_eye,
            colorSpace='rgb255',
            lineColor='black',
            units= 'height'
        )
        
        left_eye_text = visual.TextStim(
            self.win,
            text='Left Eye',
            pos=(-cfg.ui_sizes.legend_spacing, base_y_pos + cfg.ui_sizes.legend_text_offset),
            color='white',
            height= cfg.ui_sizes.legend_text,
            anchorHoriz='center',
            anchorVert='center',
            units= 'height',
            font='Consolas'
        )
        
        # Right eye legend (red ellipse + text)
        right_eye_legend = visual.Circle(
            self.win,
            size=(cfg.ui_sizes.legend_eye_width, cfg.ui_sizes.legend_eye_height), 
            pos=(cfg.ui_sizes.legend_spacing, base_y_pos),
            fillColor= cfg.colors.right_eye,
            colorSpace='rgb255',
            lineColor='black',
            units= 'height'
        )
        
        right_eye_text = visual.TextStim(
            self.win,
            text='Right Eye',
            pos=(cfg.ui_sizes.legend_spacing, base_y_pos + cfg.ui_sizes.legend_text_offset),
            color='white',
            height=cfg.ui_sizes.legend_text,
            anchorHoriz='center',
            anchorVert='center',
            units= 'height',
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
        
        formatted_instructions = NicePrint(result_instructions, "Calibration Results", verbose=self.verbose)
        result_instructions_visual = visual.TextStim(
            self.win,
            text=formatted_instructions,
            pos=(0, -0.25),
            color='white',
            height= cfg.ui_sizes.instruction_text,
            alignText='center',
            anchorHoriz='center',
            anchorVert='center',
            units='height', 
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
                    highlight_radius = cfg.ui_sizes.highlight 
                    line_width_pixels = cfg.ui_sizes.line_width * self.win.size[1]
                    
                    highlight = visual.Circle(
                        self.win,
                        radius=highlight_radius,
                        pos=calibration_points[retry_idx],
                        lineColor=cfg.colors.highlight,
                        fillColor=None,
                        lineWidth=max(1, int(line_width_pixels)),
                        edges=128,
                        units= 'height' #self.win.units
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

                    
    def _prepare_automatic_point_order(self, pacing, presentation_order):
        """
        Prepare the target sequence for an automatic calibration pass.

        Random order prevents the participant from anticipating the next target
        location. An explicit order supports fixed research protocols. Retry
        passes filter the requested order to the selected points while keeping
        their relative sequence.

        Returns
        -------
        list of int
            Zero-based calibration point indices. Empty for manual pacing.
        """

        # --- Manual Pacing ---
        # Manual calibration is driven by number keys, so it needs no queue.
        if pacing != 'auto':
            return []

        # --- Random Presentation Order ---
        # Shuffle only the points belonging to this pass.
        if presentation_order == 'random':
            point_order = list(self.remaining_points)
            random.shuffle(point_order)
            return point_order

        # --- Explicit Presentation Order ---
        # Convert the public 1-based point numbers to zero-based indices and
        # remove points that do not belong to the current retry pass.
        eligible_point_indices = set(self.remaining_points)
        return [
            point_number - 1
            for point_number in presentation_order
            if point_number - 1 in eligible_point_indices
        ]

    def _activate_calibration_target(
        self,
        point_idx,
        target_pos,
        animation_clock,
    ):
        """
        Place a target and restart its attention animation and audio.

        The target remains at this known screen position while the participant
        looks at it and calibration samples are acquired. Restarting the clock
        gives every newly visible target a fresh animation cycle.
        """

        # --- Target Placement ---
        # Reset rotation so an orientation from a previous trill does not carry
        # into the next target.
        stim = self.stim_objects[point_idx % len(self.stim_objects)]
        stim.setPos(target_pos)
        stim.setOri(0)
        animation_clock.reset()

        # --- Attention Audio ---
        # Audio helps attract the participant. Reset its volume because a
        # previous collection may have faded it to silence.
        if self.audio:
            if not self.audio.isPlaying:
                self.audio.play()
            self.audio.setVolume(1.0)

    def _present_and_collect_calibration_points(
        self,
        calibration_points,
        pacing='manual',
        presentation_order='random',
        **kwargs,
    ):
        """
        Present calibration targets and collect data for one pass.

        Calibration shows a target at a known screen position and records gaze
        samples while the participant looks at it. Manual pacing lets the
        experimenter choose every target. Automatic pacing presents targets in
        a prepared order while SPACE still controls when collection begins.

        Returns
        -------
        bool
            True when the pass is complete, False if the experimenter aborts.
        """

        # --- 1. Prepare Points for This Pass ---
        # remaining_points contains every point initially, or only the selected
        # retry points on a later pass.
        eligible_point_indices = set(self.remaining_points)
        automatic_point_queue = self._prepare_automatic_point_order(
            pacing,
            presentation_order,
        )
        current_point_idx = (
            automatic_point_queue[0] if automatic_point_queue else None
        )

        # --- 2. Initialize Presentation State ---
        WAITING_FOR_COLLECTION = 'waiting_for_collection'
        SETTLING_ON_TARGET = 'settling_on_target'
        COLLECTING_DATA = 'collecting_data'

        presentation_state = WAITING_FOR_COLLECTION
        animation_clock = core.Clock()
        settling_clock = core.Clock()
        abort_requested = False

        if current_point_idx is not None:
            self._activate_calibration_target(
                current_point_idx,
                calibration_points[current_point_idx],
                animation_clock,
            )

        # --- 3. Present and Collect Points Frame by Frame ---
        while True:
            # --- 3a. Process Keyboard Input ---
            # Target changes are only accepted while waiting. Changing a target
            # during settling or collection would invalidate its gaze samples.
            for key in event.getKeys():
                if key == 'escape':
                    if (
                        presentation_state == COLLECTING_DATA
                        and self._defer_collection_abort
                    ):
                        # Tobii cannot cancel a running collect_data() call.
                        # Remember the request and clean up after the worker
                        # returns so the display keeps animating meanwhile.
                        abort_requested = True
                    else:
                        # Waiting, settling and mouse sampling can stop now.
                        if self.audio:
                            self.audio.stop()
                        self._clear_collected_data()
                        return False
                    break

                if presentation_state != WAITING_FOR_COLLECTION:
                    continue

                if key == 'return':
                    # Preserve the existing early-results workflow.
                    return True

                if pacing == 'manual' and key in cfg.numkey_dict:
                    candidate_point_idx = cfg.numkey_dict[key]
                    if candidate_point_idx in eligible_point_indices:
                        current_point_idx = candidate_point_idx
                        self._activate_calibration_target(
                            current_point_idx,
                            calibration_points[current_point_idx],
                            animation_clock,
                        )

                elif key == 'space' and current_point_idx is not None:
                    # The experimenter has decided that the participant is
                    # looking at the target. Video and audio now hold attention
                    # while the participant settles before measurement.
                    presentation_state = SETTLING_ON_TARGET
                    settling_clock.reset()
                    self._activate_calibration_target(
                        current_point_idx,
                        calibration_points[current_point_idx],
                        animation_clock,
                    )

            # --- 3b. Start or Update Data Collection ---
            if presentation_state == SETTLING_ON_TARGET:
                elapsed_settling = settling_clock.getTime()
                focus_time = max(self.focus_time, 0.0)

                # --- Fade Attention Audio Without Blocking Drawing ---
                # The fade follows the same elapsed time as settling, so audio
                # reaches silence exactly when sample collection begins.
                if self.audio and focus_time > 0:
                    remaining_volume = max(
                        0.0,
                        1.0 - elapsed_settling / focus_time,
                    )
                    self.audio.setVolume(remaining_volume)

                if focus_time == 0 or elapsed_settling >= focus_time:
                    # --- Settling Complete ---
                    # Measurement starts only after the target has had time to
                    # capture stable fixation.
                    if self.audio:
                        self.audio.setVolume(0.0)
                        self.audio.stop()

                    self._start_point_data_collection(
                        current_point_idx,
                        calibration_points[current_point_idx],
                        **kwargs,
                    )
                    presentation_state = COLLECTING_DATA

            elif presentation_state == COLLECTING_DATA:
                collection_result = self._update_point_data_collection(**kwargs)

                if collection_result is not None:
                    if abort_requested:
                        if self.audio:
                            self.audio.stop()
                        self._clear_collected_data()
                        return False

                    if collection_result:
                        # --- Successful Collection ---
                        if pacing == 'auto':
                            automatic_point_queue.pop(0)
                            if not automatic_point_queue:
                                return True

                            current_point_idx = automatic_point_queue[0]
                            self._activate_calibration_target(
                                current_point_idx,
                                calibration_points[current_point_idx],
                                animation_clock,
                            )
                        else:
                            # Manual pacing hides the target until the
                            # experimenter selects another point.
                            current_point_idx = None

                    else:
                        # --- Failed Collection ---
                        # Keep the same target available for another attempt.
                        # Automatic pacing must not advance the queue.
                        self._activate_calibration_target(
                            current_point_idx,
                            calibration_points[current_point_idx],
                            animation_clock,
                        )

                    presentation_state = WAITING_FOR_COLLECTION

            # --- 3c. Draw Current Frame ---
            self.win.clearBuffer()
            self._draw_calibration_border()

            if current_point_idx is not None:
                stim = self.stim_objects[
                    current_point_idx % len(self.stim_objects)
                ]
                animation_phase = (
                    'collection'
                    if presentation_state in (
                        SETTLING_ON_TARGET,
                        COLLECTING_DATA,
                    )
                    else 'attention'
                )
                self._animate(
                    stim,
                    animation_clock,
                    current_point_idx,
                    phase=animation_phase,
                )

            self.win.flip()

    def _create_calibration_result_image(self, sample_data):
            """
            Common function to create calibration result visualization.
            
            ALWAYS draws target circles for ALL calibration points from 
            self.calibration_points, then draws sample lines only for points
            that have valid data in sample_data.
            """
            # --- Image Canvas Creation ---
            # The screen buffer can be larger than the window size (Retina),
            # so the image and every pixel size are scaled to match it.
            scale = get_content_scale(self.win)
            canvas_size = (int(self.win.size[0] * scale), int(self.win.size[1] * scale))
            img = Image.new("RGBA", canvas_size)
            img_draw = ImageDraw.Draw(img)
            
            # --- Configuration ---
            # Convert sizes to screen-buffer pixels
            line_width_pixels = cfg.ui_sizes.plot_line * self.win.size[1] * scale
            target_circle_radius_pixels = cfg.ui_sizes.target_circle * self.win.size[1] * scale
            target_circle_width_pixels = cfg.ui_sizes.target_circle_width * self.win.size[1] * scale
            
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
                sample_marker_radius = cfg.ui_sizes.sample_marker * self.win.size[1] * scale
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
            for point_idx, target_pix in enumerate(self.calibration_targets_pixels):
                # Convert to pixels
                # target_pix = psychopy_to_pixels(self.win, target_pos)
                
                # Draw target circle
                img_draw.ellipse(
                    (target_pix[0] - target_circle_radius_pixels, 
                    target_pix[1] - target_circle_radius_pixels,
                    target_pix[0] + target_circle_radius_pixels, 
                    target_pix[1] + target_circle_radius_pixels),
                    outline=cfg.colors.target_outline,
                    width=max(1, int(target_circle_width_pixels))
                )

            # ImageStim draws the image as a texture over the whole window, which
            # keeps it aligned with the screen buffer; SimpleImageStim does not.
            return visual.ImageStim(
                self.win, image=img, size=tuple(self.win.size),
                units='pix', interpolate=True, autoLog=False
            )


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
        collection_anim_type='zoom',
        stim_size='big',
        visualization_style='lines',
        verbose=True
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
            All visual elements are rendered in this window.
        calibration_api : tobii_research.ScreenBasedCalibration
            Tobii's calibration interface object, pre-configured for the connected
            eye tracker. This handles the low-level calibration data collection
            and computation. Typically created via:
            `tr.ScreenBasedCalibration(eyetracker)`.
        infant_stims : list of psychopy.visual or list of str
            Calibration target stimuli. Can be:
            - List of PsychoPy visual objects (ImageStim, Circle, Rect, etc.)
            - List of image file paths (strings)
            - Single visual object (will be wrapped in list)
            - Single file path string (will be wrapped in list)
        audio : psychopy.sound.Sound, optional
            Attention-getting sound to play when user selects a calibration point.
            Provides auditory feedback during the calibration process. If None,
            no sound is played. Default None.
        anim_type : {'zoom', 'trill'}, optional
            Animation style used to attract attention before data collection.
            Default 'zoom'.
        collection_anim_type : {'zoom', 'trill'}, optional
            Animation style used during settling and data collection.
            Default 'zoom'.
        stim_size : {'big', 'small'}, optional
            Size preset for calibration stimuli. Attention and collection
            sizes are configured independently in ``cfg.animation``:
            - 'big': Attention zoom 9.5-16%, collection zoom 5-9.5%;
              attention trill 12.75%, collection trill 7.25%.
            - 'small': Attention zoom 3-6%, collection zoom 1-3%;
              attention trill 4.5%, collection trill 2%.
            Default 'big'.
        visualization_style : {'lines', 'circles'}, optional
            Style for calibration result visualization:
            - 'lines': Draw lines from targets to gaze samples
            - 'circles': Draw circles at sample positions
            Default 'lines'.
        verbose : bool, optional
            If True, print status messages during calibration (e.g., point collection,
            computation results). If False, suppress informational output. Default True.
        """
        # --- Base Class Initialization ---
        super().__init__(
            win=win,
            infant_stims=infant_stims,
            audio=audio,
            anim_type=anim_type,
            collection_anim_type=collection_anim_type,
            stim_size=stim_size,
            visualization_style=visualization_style
        )
        
        # --- Tobii-Specific Setup ---
        self.calibration = calibration_api
        self.verbose = verbose

        # Tobii collection runs in one background worker. The executor is
        # created by run() and shut down before calibration mode is left.
        self._collection_executor = None
        self._collection_future = None
        self._defer_collection_abort = True

    def run(self, calibration_points, pacing='manual', presentation_order='random'):
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
        pacing : {'manual', 'auto'}, optional
            Manual pacing waits for number-key point selection. Auto pacing
            presents targets progressively while SPACE controls collection.
        presentation_order : {'random'} or tuple of int, optional
            Automatic presentation order as 1-based point numbers. Retry
            passes retain this relative order after filtering.

        Returns
        -------
        bool
            True if calibration was successful and accepted by user, False if
            aborted via escape key or if calibration computation failed.
        """

        # --- Instruction Display (before entering calibration mode) ---
        instructions_text = f"""Tobii Eye Tracker Calibration Setup:

    - Press number keys (1-{len(calibration_points)}) to select calibration points
    - Look at the animated stimulus when it appears
    - Press SPACE to collect samples at the selected point
    - Press ENTER to finish collecting and see results
    - Press ESCAPE to exit calibration

    Any key will start calibration immediately!"""

        if pacing == 'auto':
            if not self._show_auto_paced_instructions(
                "Wait until the participant looks at the animated target"
            ):
                return False
        else:
            self.show_message_and_wait(instructions_text, "Eye Tracker Calibration")

        # --- Start the Collection Worker ---
        # One worker handles all points in this calibration session. No dummy
        # task is submitted: the thread starts with the first real collection.
        self._collection_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix='detox-calibration',
        )

        entered_calibration_mode = False
        try:
            self.calibration.enter_calibration_mode()
            entered_calibration_mode = True
            return self._run_calibration_session(
                calibration_points, pacing, presentation_order
            )
        finally:
            # --- Finish Pending Collection Before Leaving Calibration Mode ---
            # shutdown(wait=True) keeps the SDK call alive until it has
            # completed. Escape cleanup and leaving calibration mode must not
            # race with an active worker.
            self._collection_executor.shutdown(wait=True)
            self._collection_executor = None

            if entered_calibration_mode:
                self.calibration.leave_calibration_mode()

    def _run_calibration_session(self, calibration_points, pacing='manual',
                                 presentation_order='random'):
        """Run the calibration steps while calibration mode is active."""
        # --- 3. Setup and Validation ---
        self.check_points(calibration_points)
        self._prepare_session(calibration_points)
        
        # --- 4. Pre-convert to Tobii ADCS Coordinates (Once!) ---
        self.tobii_points = get_tobii_pos(self.win, self.calibration_points, source_units = self.calstim_units)  
        # self.tobii_points = [get_tobii_pos(self.win, pt) for pt in cal_points_window]

        # --- 5. Main Calibration Loop ---
        while True:
            # --- 5a. Data Collection ---
            success = self._present_and_collect_calibration_points(
                self.calibration_points,
                pacing=pacing,
                presentation_order=presentation_order,
            )
            
            if not success:
                return False

            # --- 5b. Calibration Computation ---
            self.calibration_result = self.calibration.compute_and_apply()
            result_img = self._show_calibration_result()

            # --- 5c. User Review and Selection ---
            retries = self._selection_phase(self.calibration_points, result_img)

            if retries is None:
                # Restart all: reset remaining points and clear data
                self.remaining_points = list(range(len(self.calibration_points)))
                self._clear_collected_data()
                continue
            elif not retries:
                # Accept: finished!
                break
            else:
                # Retry specific points: update remaining points and discard data
                self.remaining_points = retries.copy()
                self._discard_phase(self.calibration_points, retries)

        # --- 6. Final Success Check ---
        return (self.calibration_result.status == tr.CALIBRATION_STATUS_SUCCESS)


    def _start_point_data_collection(self, point_idx, target_pos, **kwargs):
        """
        Start Tobii collection for one calibration point.

        The public collect_data() call blocks until the tracker has enough
        samples. It runs in a single background worker so the presentation loop
        can continue drawing the collection animation.
        """

        # --- Prevent Overlapping Collection Attempts ---
        if self._collection_future is not None:
            raise RuntimeError("A calibration collection is already pending.")
        if self._collection_executor is None:
            raise RuntimeError("The Tobii collection executor is not running.")

        # --- Use Pre-converted Coordinates ---
        x, y = self.tobii_points[point_idx]

        # --- Remove Previous Samples for This Point ---
        self.calibration.discard_data(x, y)

        # --- Start Eye-Tracker Collection ---
        # Only the blocking SDK call belongs in the worker. PsychoPy, audio
        # and calibration state remain owned by the main thread.
        self._collection_future = self._collection_executor.submit(
            self.calibration.collect_data,
            x,
            y,
        )

    def _update_point_data_collection(self, **kwargs):
        """
        Check whether Tobii collection for the current point has finished.

        Returns
        -------
        None
            Collection is still running.
        bool
            True on success, False when Tobii reports collection failure.
        """

        # --- Poll Without Blocking the Display ---
        future = self._collection_future
        if future is None:
            raise RuntimeError("No calibration collection has been started.")
        if not future.done():
            return None

        # --- Read the Completed SDK Result ---
        # result() raises any worker exception on the main thread. Clearing the
        # Future in finally lets cleanup recover even when that happens.
        try:
            status = future.result()
        finally:
            self._collection_future = None

        return status == tr.CALIBRATION_STATUS_SUCCESS
    
    
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
            
            Extracts sample data from Tobii results, converts to pixels, and 
            formats as (pixel_pos, color) tuples for the base class.
            """
            # --- Initialize Sample Data ---
            sample_data = {}
            
            # --- Extract Lines from Tobii Results ---
            if self.calibration_result.status != tr.CALIBRATION_STATUS_FAILURE:
                for point in self.calibration_result.calibration_points:
                    # Get the target position in ADCS (0-1)
                    target_adcs = point.position_on_display_area
                    
                    # Match to our original points to find the correct point_idx
                    # (We match by proximity because floats might not be identical)
                    found_idx = -1
                    for idx, tobii_pt in enumerate(self.tobii_points):
                        if (abs(tobii_pt[0] - target_adcs[0]) < 0.001 and
                            abs(tobii_pt[1] - target_adcs[1]) < 0.001):
                            found_idx = idx
                            break
                    
                    if found_idx != -1:
                        # Collect samples for this point
                        samples = []
                        for sample in point.calibration_samples:
                            # --- Process Left Eye ---
                            if sample.left_eye.validity == tr.VALIDITY_VALID_AND_USED:
                                left_adcs = sample.left_eye.position_on_display_area
                                # 1. Convert ADCS to PsychoPy units
                                left_psychopy = get_psychopy_pos(self.win, left_adcs)
                                # 2. Convert PsychoPy to Pixels (integers)
                                left_pix = psychopy_to_pixels(self.win, left_psychopy)
                                # 3. Add to list as 2-item tuple
                                samples.append((left_pix, cfg.colors.left_eye))

                            # --- Process Right Eye ---
                            if sample.right_eye.validity == tr.VALIDITY_VALID_AND_USED:
                                right_adcs = sample.right_eye.position_on_display_area
                                # 1. Convert ADCS to PsychoPy units
                                right_psychopy = get_psychopy_pos(self.win, right_adcs)
                                # 2. Convert PsychoPy to Pixels (integers)
                                right_pix = psychopy_to_pixels(self.win, right_psychopy)
                                # 3. Add to list as 2-item tuple
                                samples.append((right_pix, cfg.colors.right_eye))
                        
                        # Store if we have samples
                        if samples:
                            sample_data[found_idx] = samples
            
            # --- Generate Visualization ---
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
        collection_anim_type='zoom',
        stim_size='big',
        visualization_style='lines',
        verbose=True
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
            All visual elements are rendered in this window.
        infant_stims : list of psychopy.visual or list of str
            Calibration target stimuli. Can be:
            - List of PsychoPy visual objects (ImageStim, Circle, Rect, etc.)
            - List of image file paths (strings)
            - Single visual object (will be wrapped in list)
            - Single file path string (will be wrapped in list)
        mouse : psychopy.event.Mouse
            Mouse object for getting position samples. Should be configured
            for the same window used for display. Typically created via:
            `event.Mouse(win=win)`.
        audio : psychopy.sound.Sound, optional
            Attention-getting sound to play when user selects a calibration point.
            Provides auditory feedback during the calibration process. If None,
            no sound is played. Default None.
        anim_type : {'zoom', 'trill'}, optional
            Animation style used to attract attention before data collection.
            Default 'zoom'.
        collection_anim_type : {'zoom', 'trill'}, optional
            Animation style used during settling and data collection.
            Default 'zoom'.
        stim_size : {'big', 'small'}, optional
            Size preset for calibration stimuli. Attention and collection
            sizes are configured independently in ``cfg.animation``:
            - 'big': Attention zoom 9.5-16%, collection zoom 5-9.5%;
              attention trill 12.75%, collection trill 7.25%.
            - 'small': Attention zoom 3-6%, collection zoom 1-3%;
              attention trill 4.5%, collection trill 2%.
            Default 'big'.
        visualization_style : {'lines', 'circles'}, optional
            Style for calibration result visualization:
            - 'lines': Draw lines from targets to gaze samples
            - 'circles': Draw circles at sample positions
            Default 'lines'.
        verbose : bool, optional
            If True, print status messages during calibration (e.g., point collection,
            sample counts). If False, suppress informational output. Default True.
        """
        # --- Base Class Initialization ---
        super().__init__(
            win=win,
            infant_stims=infant_stims,
            audio=audio,
            anim_type=anim_type,
            collection_anim_type=collection_anim_type,
            stim_size=stim_size,
            visualization_style=visualization_style
        )
        
        # --- Mouse-Specific Setup ---
        self.mouse = mouse
        self.calibration_data = {}  # point_idx -> list of (target_pos, sample_pos, timestamp)
        self.verbose = verbose

        # --- Current Simulated Collection ---
        # Mouse sampling is advanced by the presentation loop instead of
        # blocking the display for the full sample period.
        self._mouse_collection_point_idx = None
        self._mouse_collection_target_pos = None
        self._mouse_collection_samples = []
        self._mouse_collection_sample_count = 0
        self._mouse_collection_sample_interval = 0.0
        self._mouse_collection_next_sample_time = 0.0


    def run(self, calibration_points, pacing='manual', presentation_order='random'):
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
        pacing : {'manual', 'auto'}, optional
            Manual pacing waits for number-key point selection. Auto pacing
            presents targets progressively while SPACE controls collection.
        presentation_order : {'random'} or tuple of int, optional
            Automatic presentation order as 1-based point numbers. Retry
            passes retain this relative order after filtering.

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
        
        if pacing == 'auto':
            if not self._show_auto_paced_instructions(
                "Move your mouse to the animated target"
            ):
                return False
        else:
            self.show_message_and_wait(instructions_text, "Calibration Setup")
        
        
        # --- 3. Setup and Validation ---
        self.check_points(calibration_points)
        self._prepare_session(calibration_points)
        
        # --- 4. Main Calibration Loop ---
        while True:
            # --- 5a. Data Collection ---
            success = self._present_and_collect_calibration_points(
                self.calibration_points,
                pacing=pacing,
                presentation_order=presentation_order,
                num_samples=cfg.calibration.num_samples_mouse,
            )
            if not success:
                return False
                
            # --- 4b. Results Visualization ---
            result_img = self._show_results(self.calibration_points)
            
            # --- 4c. User Review and Selection ---
            retries = self._selection_phase(self.calibration_points, result_img)

            if retries is None:
                self.remaining_points = list(range(len(self.calibration_points)))
                self.calibration_data.clear()
                continue
            elif not retries:
                return True
            else:
                self.remaining_points = retries.copy()
                for idx in retries:
                    if idx in self.calibration_data:
                        del self.calibration_data[idx]

    
    
    def _start_point_data_collection(self, point_idx, target_pos, **kwargs):
        """
        Start simulated mouse collection for one calibration point.

        Mouse positions stand in for gaze samples during calibration
        development. Samples are spread across later display frames so the
        target can keep animating; the mouse is never read from a worker thread.
        """

        # --- Existing Data Cleanup ---
        # A retry replaces this point's previous samples.
        if point_idx in self.calibration_data:
            del self.calibration_data[point_idx]

        # --- Sampling Configuration ---
        # Keep the existing approximately one-second sampling window. With the
        # default five samples, observations are spaced about 0.2 seconds apart.
        sample_count = max(1, int(kwargs.get('num_samples', 5)))
        sample_interval = 1.0 / sample_count
        current_time = time.perf_counter()

        self._mouse_collection_point_idx = point_idx
        self._mouse_collection_target_pos = target_pos
        self._mouse_collection_samples = []
        self._mouse_collection_sample_count = sample_count
        self._mouse_collection_sample_interval = sample_interval
        self._mouse_collection_next_sample_time = (
            current_time + sample_interval
        )

        # --- Take the First Real Observation ---
        # Collection starts immediately after settling, without another wait.
        self._mouse_collection_samples.append((
            target_pos,
            self.mouse.getPos(),
            time.time(),
        ))

    def _update_point_data_collection(self, **kwargs):
        """
        Advance simulated mouse collection by at most one sample.

        Reading the mouse on the main thread is quick, so it needs no worker.
        If a frame arrives late, only one observation is taken; the next sample
        is scheduled from that actual reading rather than inventing positions
        for the missed interval.

        Returns
        -------
        None
            Collection is still in progress.
        bool
            True when all samples for this point have been stored.
        """

        # --- Collect One Due Sample ---
        if self._mouse_collection_point_idx is not None:
            current_time = time.perf_counter()

            if (
                current_time >= self._mouse_collection_next_sample_time
                and len(self._mouse_collection_samples)
                < self._mouse_collection_sample_count
            ):
                self._mouse_collection_samples.append((
                    self._mouse_collection_target_pos,
                    self.mouse.getPos(),
                    time.time(),
                ))
                self._mouse_collection_next_sample_time = (
                    current_time + self._mouse_collection_sample_interval
                )

        # --- Store Completed Collection ---
        if (
            self._mouse_collection_point_idx is not None
            and len(self._mouse_collection_samples)
            >= self._mouse_collection_sample_count
        ):
            self.calibration_data.setdefault(
                self._mouse_collection_point_idx, []
            ).extend(self._mouse_collection_samples)
            self._mouse_collection_point_idx = None
            self._mouse_collection_target_pos = None
            self._mouse_collection_samples = []
            return True

        return None


    def _clear_collected_data(self):
        """
        Remove all previously collected mouse calibration data.
        
        Clears the calibration data dictionary to prepare for a fresh
        calibration attempt. Called when user chooses to restart the
        entire calibration process.
        """
        # --- Data Dictionary Reset ---
        self.calibration_data.clear()

        # --- Pending Sample Reset ---
        # An abort must not leave observations that could be added to a later
        # collection attempt.
        self._mouse_collection_point_idx = None
        self._mouse_collection_target_pos = None
        self._mouse_collection_samples = []
        self._mouse_collection_sample_count = 0
        self._mouse_collection_sample_interval = 0.0
        self._mouse_collection_next_sample_time = 0.0


    def _show_results(self, calibration_points):
            """
            Visualize mouse calibration results.
            
            Converts stored mouse samples to pixels and formats as 
            (pixel_pos, color) tuples for the base class.
            """
            # --- Initialize Sample Data ---
            sample_data = {}

            # --- Extract Lines from Mouse Data ---
            for point_idx, samples in self.calibration_data.items():
                formatted_samples = []
                
                # Iterate raw stored data: (target_pos, sample_pos, timestamp)
                for _, sample_pos, _ in samples:
                    # 1. Convert PsychoPy position to Pixels (integers)
                    sample_pix = psychopy_to_pixels(self.win, sample_pos)
                    
                    # 2. Add to list as 2-item tuple: (pixels, color)
                    #    No dummy "None" value is needed anymore!
                    formatted_samples.append((sample_pix, cfg.colors.mouse))
                
                # Store if we have samples
                if formatted_samples:
                    sample_data[point_idx] = formatted_samples

            # --- Generate Visualization ---
            return self._create_calibration_result_image(sample_data)
