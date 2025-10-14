"""Eye Tracking Settings Configuration.

This module contains all configurable settings for the DeToX package,
including animation parameters, colors, UI element sizes, and key mappings.
Settings are organized into dataclasses for better structure and documentation.

All size values are specified in height units (percentage of screen height)
and are automatically converted to appropriate units as needed by the package.

Examples
--------
Modify settings in your experiment script:

>>> from DeToX import ETSettings
>>> 
>>> # Access settings through the config object
>>> ETSettings.config.animation.max_zoom_size = 0.15
>>> 
>>> # Or use the module-level constants for backward compatibility
>>> ETSettings.ANIMATION_SETTINGS['max_zoom_size'] = 0.15
>>> 
>>> # Change colors
>>> ETSettings.config.colors.highlight = (0, 255, 255, 255)  # Cyan

Notes
-----
The module provides both a modern dataclass interface (via `config`) and
backward-compatible module-level dictionaries for existing code.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple


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


@dataclass
class Settings:
    """Main configuration class for DeToX eye tracking package.
    
    This class serves as the central configuration container, organizing
    all settings into logical groups. It provides both a structured interface
    for accessing settings and maintains backward compatibility with
    dictionary-based access patterns.
    
    Attributes
    ----------
    numkey_dict : dict of str to int
        Mapping from keyboard key names to calibration point indices.
        Supports both standard number keys ('1'-'9') and numpad keys
        ('num_1'-'num_9'). Key '0' or 'num_0' maps to index -1 (no selection).
    animation : AnimationSettings
        Animation parameters for calibration stimuli including zoom and
        trill settings. See AnimationSettings for details.
    colors : CalibrationColors
        RGBA color definitions for all visual elements in calibration
        displays. See CalibrationColors for details.
    ui_sizes : UIElementSizes
        Size parameters for UI elements in height units.
        See UIElementSizes for details.
    font_multipliers : FontSizeMultipliers
        Scaling factors for different text types relative to base text size.
        See FontSizeMultipliers for details.
    simulation_framerate : int
        Target framerate in Hz for simulation mode when using mouse input
        instead of real eye tracker. Default is 120 Hz to match Tobii Pro
        Spectrum specifications.
    
    Examples
    --------
    Access and modify settings through the config object:
    
    >>> from DeToX import ETSettings
    >>> 
    >>> # Modify animation settings
    >>> ETSettings.config.animation.max_zoom_size = 0.15
    >>> ETSettings.config.animation.focus_time = 1.0
    >>> 
    >>> # Change colors
    >>> ETSettings.config.colors.highlight = (0, 255, 255, 255)  # Cyan
    >>> 
    >>> # Adjust UI element sizes
    >>> ETSettings.config.ui_sizes.text = 0.035  # Larger text
    >>> 
    >>> # Modify font multipliers
    >>> ETSettings.config.font_multipliers.instruction_text = 2.0
    >>> 
    >>> # Change simulation framerate
    >>> ETSettings.config.simulation_framerate = 60
    
    Notes
    -----
    The Settings class uses dataclasses for clean structure and automatic
    initialization. All nested settings objects are created automatically
    with their default values when a Settings instance is created.
    
    For backward compatibility, module-level dictionaries are also provided
    (ANIMATION_SETTINGS, CALIBRATION_COLORS, etc.) that mirror the dataclass
    structure. These are updated from the config object at module load time.
    """
    
    numkey_dict: Dict[str, int] = field(default_factory=lambda: {
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
    })
    
    animation: AnimationSettings = field(default_factory=AnimationSettings)
    colors: CalibrationColors = field(default_factory=CalibrationColors)
    ui_sizes: UIElementSizes = field(default_factory=UIElementSizes)
    font_multipliers: FontSizeMultipliers = field(default_factory=FontSizeMultipliers)
    simulation_framerate: int = 120


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
    'Settings',
    'animation',
    'colors',
    'ui_sizes',
    'font_multipliers',
    'numkey_dict',
    'simulation_framerate',
]