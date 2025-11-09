"""Eye Tracking Settings Configuration.

This module contains all configurable settings for the DeToX package,
including animation parameters, colors, UI element sizes, and key mappings.
Settings are organized into dataclasses for better structure and documentation.

All size values are specified in height units (percentage of screen height)
and are automatically converted to appropriate units as needed by the package.

Examples
--------
Modify settings in your experiment script:

>>> from DeToX import ETSettings as cfg
>>> 
>>> # Access animation settings
>>> cfg.animation.max_zoom_size = 0.15
>>> cfg.animation.focus_time = 1.0
>>> 
>>> # Change colors
>>> cfg.colors.highlight = (0, 255, 255, 255)  # Cyan
>>> 
>>> # Modify UI sizes
>>> cfg.ui_sizes.text = 0.035

Notes
-----
The module provides both a modern dataclass interface (via `config`) and
backward-compatible module-level dictionaries for existing code.
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class AnimationSettings:
    """Animation parameters for calibration stimuli.
    
    Controls the behavior and appearance of animated calibration targets
    including zoom and trill animations. All size parameters are specified
    in height units (percentage of screen height).
    
    Attributes
    ----------
    focus_time : float
        Wait time in seconds before collecting calibration data at each point.
        Allows participant to fixate on the target. Default is 0.5 seconds.
    zoom_speed : float
        Speed multiplier for the zoom animation. Higher values make the
        size oscillation faster. Default is 6.0.
    max_zoom_size : float
        Maximum size for zoom animation as percentage of screen height.
        Default is 0.11 (11% of screen height).
    min_zoom_size : float
        Minimum size for zoom animation as percentage of screen height.
        Default is 0.05 (5% of screen height).
    trill_size : float
        Fixed size for trill animation as percentage of screen height.
        Default is 0.075 (7.5% of screen height).
    trill_rotation_range : float
        Maximum rotation angle in degrees for trill animation.
        Default is 20 degrees.
    trill_cycle_duration : float
        Total cycle time for trill animation in seconds (active + pause).
        Default is 1.5 seconds.
    trill_active_duration : float
        Duration of active trill rotation in seconds, within each cycle.
        Default is 1.1 seconds (leaves 0.4s pause).
    trill_frequency : float
        Number of back-and-forth rotation oscillations per second during
        active trill phase. Default is 3.0 oscillations/second.
    
    Examples
    --------
    >>> settings = AnimationSettings()
    >>> settings.max_zoom_size = 0.15  # Increase max size to 15%
    >>> settings.trill_frequency = 5.0  # Faster trill
    """
    
    focus_time: float = 0.5
    zoom_speed: float = 6.0
    max_zoom_size: float = 0.11
    min_zoom_size: float = 0.05
    trill_size: float = 0.075
    trill_rotation_range: float = 20
    trill_cycle_duration: float = 1.5
    trill_active_duration: float = 1.1
    trill_frequency: float = 3.0


@dataclass
class CalibrationPatterns:
    """Standard calibration point patterns in normalized coordinates.
    
    Defines commonly used calibration patterns in normalized units where
    the screen ranges from -1 to +1 in both dimensions. These universal
    coordinates work across different screen sizes, aspect ratios, and
    PsychoPy unit systems.
    
    Convert to window-specific coordinates at runtime using the
    norm_to_window_units() function from the Coords module.
    
    Attributes
    ----------
    points_5 : list of tuple
        5-point calibration pattern (4 corners + center).
        Standard for quick calibrations with good coverage.
        Pattern: corners at ±0.4 from edges to avoid screen boundaries.
    points_9 : list of tuple
        9-point calibration pattern (3×3 grid).
        Standard for comprehensive calibrations requiring high accuracy.
        Pattern: 3 rows × 3 columns with ±0.4 positioning.
    num_samples_mouse : int
        Number of mouse position samples to collect per calibration point
        in simulation mode.
        Default 5.
    
    Examples
    --------
    >>> from DeToX import ETSettings as cfg
    >>> from DeToX.Coords import norm_to_window_units
    >>> 
    >>> # Change number of mouse samples collected per point
    >>> cfg.calibration.num_samples_mouse = 10
    """
    
    points_5: list = field(default_factory=lambda: [
        (0.0, 0.0),     # Center
        (-0.8,  0.8),   # Top-left
        ( 0.8,  0.8),   # Top-right
        (-0.8, -0.8),   # Bottom-left
        ( 0.8, -0.8)    # Bottom-right
    ])
    
    points_9: list = field(default_factory=lambda: [
        (-0.8,  0.8), (0.0,  0.8), ( 0.8,  0.8),   # Top row
        (-0.8,  0.0), (0.0,  0.0), ( 0.8,  0.0),   # Middle row
        (-0.8, -0.8), (0.0, -0.8), ( 0.8, -0.8)    # Bottom row
    ])
    
    num_samples_mouse: int = 5


@dataclass
class CalibrationColors:
    """Color settings for calibration visual elements.
    
    Defines RGBA color values for various calibration display components
    including eye tracking samples, target outlines, and highlights.
    
    Attributes
    ----------
    left_eye : tuple of int
        RGBA color for Tobii left eye gaze samples (R, G, B, A).
        Default is (0, 255, 0, 255) - bright green.
    right_eye : tuple of int
        RGBA color for Tobii right eye gaze samples (R, G, B, A).
        Default is (255, 0, 0, 255) - bright red.
    mouse : tuple of int
        RGBA color for simulated mouse position samples (R, G, B, A).
        Default is (255, 128, 0, 255) - orange.
    target_outline : tuple of int
        RGBA color for calibration target circle outlines (R, G, B, A).
        Default is (24, 24, 24, 255) - dark gray/black.
    highlight : tuple of int
        RGBA color for highlighting selected calibration points (R, G, B, A).
        Default is (255, 255, 0, 255) - bright yellow.
    
    Notes
    -----
    All color values use 8-bit channels (0-255 range) in RGBA format.
    The alpha channel (A) controls opacity where 255 is fully opaque.
    
    Examples
    --------
    >>> colors = CalibrationColors()
    >>> colors.highlight = (0, 255, 255, 255)  # Change to cyan
    >>> colors.left_eye = (0, 200, 0, 200)  # Semi-transparent green
    """
    
    left_eye: Tuple[int, int, int, int] = (0, 255, 0, 255)
    right_eye: Tuple[int, int, int, int] = (255, 0, 0, 255)
    mouse: Tuple[int, int, int, int] = (255, 128, 0, 255)
    target_outline: Tuple[int, int, int, int] = (24, 24, 24, 255)
    highlight: Tuple[int, int, int, int] = (255, 255, 0, 255)


@dataclass
class UIElementSizes:
    """Size settings for user interface elements.
    
    Defines sizes for various UI components in the calibration interface.
    All sizes are specified in height units (as fraction of screen height)
    and are automatically converted to appropriate units based on the
    PsychoPy window configuration.
    
    Attributes
    ----------
    highlight : float
        Radius of circles highlighting selected calibration points for retry.
        Default is 0.04 (4% of screen height).
    line_width : float
        Thickness of lines drawn in calibration visualizations.
        Default is 0.003 (0.3% of screen height).
    marker : float
        Size of markers indicating data collection points.
        Default is 0.02 (2% of screen height).
    border : float
        Thickness of the red calibration mode border around the screen.
        Default is 0.005 (0.5% of screen height).
    plot_line : float
        Width of lines in calibration result plots connecting targets to samples.
        Default is 0.002 (0.2% of screen height).
    text : float
        Base text height for all text displays in the calibration interface.
        Default is 0.025 (2.5% of screen height).
    target_circle : float
        Radius of target circles drawn in calibration result visualizations.
        Default is 0.012 (1.2% of screen height).
    target_circle_width : float
        Line width for target circle outlines in result visualizations.
        Default is 0.006 (0.6% of screen height).
    
    Notes
    -----
    Height units provide consistent visual appearance across different
    screen sizes and aspect ratios. The conversion to pixels or other
    units is handled automatically by the coordinate conversion functions.
    
    Examples
    --------
    >>> ui_sizes = UIElementSizes()
    >>> ui_sizes.highlight = 0.06  # Larger highlight circles
    >>> ui_sizes.text = 0.035  # Larger text
    """
    
    highlight: float = 0.04
    line_width: float = 0.003
    marker: float = 0.02
    border: float = 0.005
    plot_line: float = 0.002
    text: float = 0.025
    target_circle: float = 0.012
    target_circle_width: float = 0.006
    sample_marker: float = 0.008  # ← ADD THIS LINE


@dataclass
class FontSizeMultipliers:
    """Font size multipliers for different text types.
    
    Defines scaling factors applied to the base text size (from UIElementSizes)
    for different types of text displays in the calibration interface.
    
    Attributes
    ----------
    instruction_text : float
        Multiplier for instruction text displayed during calibration.
        Default is 1.5 (150% of base text size).
    message_text : float
        Multiplier for general message text.
        Default is 1.3 (130% of base text size).
    title_text : float
        Multiplier for title text in message boxes.
        Default is 1.4 (140% of base text size).
    
    Notes
    -----
    The final text size is calculated as: base_text_size * multiplier
    where base_text_size comes from UIElementSizes.text.
    
    Examples
    --------
    >>> font_sizes = FontSizeMultipliers()
    >>> font_sizes.instruction_text = 2.0  # Larger instructions
    >>> font_sizes.title_text = 1.8  # Larger titles
    """
    
    instruction_text: float = 1.5
    message_text: float = 1.3
    title_text: float = 1.4

class RawDataColumns:
    """
    Column specifications for raw Tobii SDK data format.
    
    This class defines the complete structure for raw format data including:
    - Column order (matching pandas' dtype grouping for HDF5 compatibility)
    - Data types for each column
    - Default values for dummy data creation
    
    The order is optimized for HDF5 storage where related measurements
    (coordinates + validity) are grouped together for easier analysis.
    """
    
    # Column order (list)
    ORDER = [
        # Timestamps
        'device_time_stamp', 'system_time_stamp',
        
        # Left gaze point on display + validity
        'left_gaze_point_on_display_area_x', 
        'left_gaze_point_on_display_area_y',
        'left_gaze_point_validity',
        
        # Right gaze point on display + validity
        'right_gaze_point_on_display_area_x', 
        'right_gaze_point_on_display_area_y',
        'right_gaze_point_validity',
        
        # Left gaze point in user coords
        'left_gaze_point_in_user_coordinate_system_x',
        'left_gaze_point_in_user_coordinate_system_y',
        'left_gaze_point_in_user_coordinate_system_z',
        
        # Right gaze point in user coords
        'right_gaze_point_in_user_coordinate_system_x',
        'right_gaze_point_in_user_coordinate_system_y',
        'right_gaze_point_in_user_coordinate_system_z',
        
        # Left pupil + validity
        'left_pupil_diameter',
        'left_pupil_validity',
        
        # Right pupil + validity
        'right_pupil_diameter',
        'right_pupil_validity',
        
        # Left gaze origin in user coords + validity
        'left_gaze_origin_in_user_coordinate_system_x',
        'left_gaze_origin_in_user_coordinate_system_y',
        'left_gaze_origin_in_user_coordinate_system_z',
        'left_gaze_origin_validity',
        
        # Right gaze origin in user coords + validity
        'right_gaze_origin_in_user_coordinate_system_x',
        'right_gaze_origin_in_user_coordinate_system_y',
        'right_gaze_origin_in_user_coordinate_system_z',
        'right_gaze_origin_validity',
        
        # Events
        'Events'
    ]
    
    # Data types (dict)
    DTYPES = {
        # Timestamps and validity - int64
        'device_time_stamp': 'int64',
        'system_time_stamp': 'int64',
        'left_gaze_point_validity': 'int64',
        'right_gaze_point_validity': 'int64',
        'left_pupil_validity': 'int64',
        'right_pupil_validity': 'int64',
        'left_gaze_origin_validity': 'int64',
        'right_gaze_origin_validity': 'int64',
        
        # All coordinate and diameter values - float64
        'left_gaze_point_on_display_area_x': 'float64',
        'left_gaze_point_on_display_area_y': 'float64',
        'right_gaze_point_on_display_area_x': 'float64',
        'right_gaze_point_on_display_area_y': 'float64',
        'left_gaze_point_in_user_coordinate_system_x': 'float64',
        'left_gaze_point_in_user_coordinate_system_y': 'float64',
        'left_gaze_point_in_user_coordinate_system_z': 'float64',
        'right_gaze_point_in_user_coordinate_system_x': 'float64',
        'right_gaze_point_in_user_coordinate_system_y': 'float64',
        'right_gaze_point_in_user_coordinate_system_z': 'float64',
        'left_pupil_diameter': 'float64',
        'right_pupil_diameter': 'float64',
        'left_gaze_origin_in_user_coordinate_system_x': 'float64',
        'left_gaze_origin_in_user_coordinate_system_y': 'float64',
        'left_gaze_origin_in_user_coordinate_system_z': 'float64',
        'right_gaze_origin_in_user_coordinate_system_x': 'float64',
        'right_gaze_origin_in_user_coordinate_system_y': 'float64',
        'right_gaze_origin_in_user_coordinate_system_z': 'float64',
        
        # Events - string
        'Events': 'string'
    }
    
    # Default values for dummy data creation (dict)
    DEFAULTS = {
        # Timestamps
        'device_time_stamp': -999999,
        'system_time_stamp': -999999,
        
        # Validity flags
        'left_gaze_point_validity': 0,
        'right_gaze_point_validity': 0,
        'left_pupil_validity': 0,
        'right_pupil_validity': 0,
        'left_gaze_origin_validity': 0,
        'right_gaze_origin_validity': 0,
        
        # All float columns default to NaN
        'left_gaze_point_on_display_area_x': float('nan'),
        'left_gaze_point_on_display_area_y': float('nan'),
        'right_gaze_point_on_display_area_x': float('nan'),
        'right_gaze_point_on_display_area_y': float('nan'),
        'left_gaze_point_in_user_coordinate_system_x': float('nan'),
        'left_gaze_point_in_user_coordinate_system_y': float('nan'),
        'left_gaze_point_in_user_coordinate_system_z': float('nan'),
        'right_gaze_point_in_user_coordinate_system_x': float('nan'),
        'right_gaze_point_in_user_coordinate_system_y': float('nan'),
        'right_gaze_point_in_user_coordinate_system_z': float('nan'),
        'left_pupil_diameter': float('nan'),
        'right_pupil_diameter': float('nan'),
        'left_gaze_origin_in_user_coordinate_system_x': float('nan'),
        'left_gaze_origin_in_user_coordinate_system_y': float('nan'),
        'left_gaze_origin_in_user_coordinate_system_z': float('nan'),
        'right_gaze_origin_in_user_coordinate_system_x': float('nan'),
        'right_gaze_origin_in_user_coordinate_system_y': float('nan'),
        'right_gaze_origin_in_user_coordinate_system_z': float('nan'),
        
        # Events
        'Events': '__DUMMY__'
    }
    
    @classmethod
    def get_dummy_dict(cls):
        """
        Get dictionary for creating dummy DataFrame with proper structure.
        
        Returns
        -------
        dict
            Dictionary with column names as keys and lists containing default
            values as values, ready for pd.DataFrame() constructor.
        """
        return {col: [cls.DEFAULTS[col]] for col in cls.ORDER}
    
    @classmethod
    def get_validity_dtypes(cls):
        """
        Get dictionary of validity column dtypes for optimization.
        
        Returns
        -------
        dict
            Dictionary mapping validity column names to 'int8' dtype.
        """
        return {col: dtype for col, dtype in cls.DTYPES.items() if 'validity' in col}


class SimplifiedDataColumns:
    """
    Column specifications for simplified user-friendly data format.
    
    This class defines the structure for simplified format data that:
    - Uses short, intuitive column names
    - Contains only essential gaze tracking data
    - Has coordinates already converted to PsychoPy units
    - Is optimized for quick analysis and visualization
    """
    
    # Column order (list)
    ORDER = [
        'TimeStamp',
        'Left_X', 'Left_Y', 'Left_Validity',
        'Left_Pupil', 'Left_Pupil_Validity',
        'Right_X', 'Right_Y', 'Right_Validity',
        'Right_Pupil', 'Right_Pupil_Validity',
        'Events'
    ]
    
    # Data types (dict)
    DTYPES = {
        # Timestamp - int64
        'TimeStamp': 'int64',
        
        # Coordinates - float64
        'Left_X': 'float64',
        'Left_Y': 'float64',
        'Right_X': 'float64',
        'Right_Y': 'float64',
        
        # Pupil diameters - float64
        'Left_Pupil': 'float64',
        'Right_Pupil': 'float64',
        
        # Validity flags
        'Left_Validity': 'int64',
        'Right_Validity': 'int64',
        'Left_Pupil_Validity': 'int64',
        'Right_Pupil_Validity': 'int64',
        
        # Events - string
        'Events': 'string'
    }
    
    # Default values for dummy data creation (dict)
    DEFAULTS = {
        'TimeStamp': -999999,
        'Left_X': float('nan'),
        'Left_Y': float('nan'),
        'Left_Validity': 0,
        'Left_Pupil': float('nan'),
        'Left_Pupil_Validity': 0,
        'Right_X': float('nan'),
        'Right_Y': float('nan'),
        'Right_Validity': 0,
        'Right_Pupil': float('nan'),
        'Right_Pupil_Validity': 0,
        'Events': '__DUMMY__'
    }
    
    @classmethod
    def get_dummy_dict(cls):
        """
        Get dictionary for creating dummy DataFrame with proper structure.
        
        Returns
        -------
        dict
            Dictionary with column names as keys and lists containing default
            values as values, ready for pd.DataFrame() constructor.
        """
        return {col: [cls.DEFAULTS[col]] for col in cls.ORDER}
    
    @classmethod
    def get_validity_dtypes(cls):
        """
        Get dictionary of validity column dtypes for optimization.
        
        Returns
        -------
        dict
            Dictionary mapping validity column names to 'int8' dtype.
        """
        return {col: dtype for col, dtype in cls.DTYPES.items() if 'Validity' in col}



# =============================================================================
# Module-Level Configuration Instances
# =============================================================================

#: Animation settings for calibration stimuli.
#:
#: Access animation parameters directly through this object.
#:
#: Examples
#: --------
#: >>> from DeToX import ETSettings as cfg
#: >>> cfg.animation.max_zoom_size = 0.15
#: >>> cfg.animation.focus_time = 1.0
animation = AnimationSettings()

#: Standard calibration point patterns.
#:
#: Access calibration patterns directly through this object.
#:
#: Examples
#: --------
#: >>> from DeToX import ETSettings as cfg
#: >>> from DeToX.Coords import norm_to_window_units
#: >>> cal_points = norm_to_window_units(win, cfg.calibration.points_5)
calibration = CalibrationPatterns()

#: Color settings for calibration visual elements.
#:
#: Access color definitions directly through this object.
#:
#: Examples
#: --------
#: >>> from DeToX import ETSettings as cfg
#: >>> cfg.colors.highlight = (0, 255, 255, 255)
#: >>> cfg.colors.left_eye = (0, 200, 0, 255)
colors = CalibrationColors()

#: Size settings for UI elements.
#:
#: Access UI element sizes directly through this object.
#:
#: Examples
#: --------
#: >>> from DeToX import ETSettings as cfg
#: >>> cfg.ui_sizes.text = 0.035
#: >>> cfg.ui_sizes.highlight = 0.06
ui_sizes = UIElementSizes()

#: Font size multipliers for different text types.
#:
#: Access font scaling factors directly through this object.
#:
#: Examples
#: --------
#: >>> from DeToX import ETSettings as cfg
#: >>> cfg.font_multipliers.instruction_text = 2.0
font_multipliers = FontSizeMultipliers()

#: Keyboard key to calibration point index mapping.
#:
#: Maps key names (str) to calibration point indices (int).
#:
#: Examples
#: --------
#: >>> from DeToX import ETSettings as cfg
#: >>> cfg.numkey_dict['1']  # Returns 0 (first point)
#: 0
numkey_dict = {
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

#: Simulation mode framerate in Hz.
#:
#: Target framerate for mouse-based simulation mode.
#:
#: Examples
#: --------
#: >>> from DeToX import ETSettings as cfg
#: >>> cfg.simulation_framerate = 60
simulation_framerate = 120


__all__ = [
    'AnimationSettings',
    'CalibrationColors',
    'UIElementSizes',
    'FontSizeMultipliers',
    'CalibrationPatterns',  # Add this
    'animation',
    'colors',
    'ui_sizes',
    'font_multipliers',
    'calibration',  # Add this
    'numkey_dict',
    'simulation_framerate',
    'RawDataColumns',
    'SimplifiedDataColumns',
]