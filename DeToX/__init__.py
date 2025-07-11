# Import main classes and functions for direct access
from .Base import TobiiController
from .Utils import InfantStimuli, NicePrint
from .Coords import (
    get_psychopy_pos, 
    get_tobii_pos, 
    pix2tobii, 
    tobii2pix, 
    get_psychopy_pos_from_trackbox,
    psychopy_to_pixels  # Add this line
)
from .Calibration import CalibrationSession

# Define the version
__version__ = '0.1.0'

# Define what gets exported with "from package import *"
__all__ = [
    'TobiiController',
    'InfantStimuli',
    'NicePrint',
    'CalibrationSession',
    'get_psychopy_pos',
    'get_tobii_pos',
    'pix2tobii',
    'tobii2pix',
    'get_psychopy_pos_from_trackbox',
    'psychopy_to_pixels'  # Add this line
]


# In DeToX/__init__.py, add to the imports:
from .Coords import (
    get_psychopy_pos, 
    get_tobii_pos, 
    pix2tobii, 
    tobii2pix, 
    get_psychopy_pos_from_trackbox,
    psychopy_to_pixels  # Add this line
)

