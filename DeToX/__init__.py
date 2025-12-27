# DeToX/__init__.py
# Import main classes for direct access
from .Base import ETracker
from .Calibration import BaseCalibrationSession, TobiiCalibrationSession, MouseCalibrationSession
from .Utils import NicePrint
from . import ETSettings 
from .Coords import (
    get_psychopy_pos,
    get_tobii_pos,
    pix2tobii,
    get_psychopy_pos_from_trackbox,
    psychopy_to_pixels,
    convert_height_to_units,
    norm_to_window_units
)

# Define the version
__version__ = '0.1.1'

# Define what gets exported with "from DeToX import *"
__all__ = [
    'ETracker',
    'BaseCalibrationSession',
    'TobiiCalibrationSession',
    'MouseCalibrationSession',
    'NicePrint',
    'ETSettings', 
    'get_psychopy_pos',
    'get_tobii_pos',
    'pix2tobii',
    'get_psychopy_pos_from_trackbox',
    'psychopy_to_pixels',
    'convert_height_to_units',
    'norm_to_window_units'
]