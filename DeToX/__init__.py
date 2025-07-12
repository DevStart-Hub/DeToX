# DeToX/__init__.py

# Import main classes for direct access
from .Base import TobiiController
from .Calibration import CalibrationSession
from .Utils import InfantStimuli, NicePrint
from . import calibration_config
from .Coords import (
    get_psychopy_pos, 
    get_tobii_pos, 
    pix2tobii, 
    tobii2pix, 
    get_psychopy_pos_from_trackbox,
    psychopy_to_pixels
)

# Define the version
__version__ = '0.1.0'

# Define what gets exported with "from DeToX import *"
__all__ = [
    'TobiiController',
    'CalibrationSession',
    'InfantStimuli',
    'NicePrint',
    'get_psychopy_pos',
    'get_tobii_pos',
    'pix2tobii',
    'tobii2pix',
    'get_psychopy_pos_from_trackbox',
    'psychopy_to_pixels'
]
