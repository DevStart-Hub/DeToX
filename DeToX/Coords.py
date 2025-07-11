# Third party imports
import numpy as np
from psychopy.tools.monitorunittools import cm2pix, deg2pix, pix2cm, pix2deg


def get_psychopy_pos(win, p, units=None):
    """
    Convert Tobii ADCS coordinates to PsychoPy coordinates.

    Parameters
    ----------
    win : psychopy.visual.Window
        The PsychoPy window which provides information about units and size.
    p : tuple
        The Tobii ADCS coordinates to convert.
    units : str, optional
        The units for the PsychoPy coordinates. Default is None, which uses the window's default units.

    Returns
    -------
    tuple
        The converted PsychoPy coordinates.

    Raises
    ------
    ValueError
        If the provided units are not supported.
    """
    if units is None:
        units = win.units

    if units == "norm":
        # Convert to normalized units, where screen ranges from -1 to 1
        return (2 * p[0] - 1, -2 * p[1] + 1)
    elif units == "height": 
        # Convert to height units, where screen height is 1 and width is adjusted
        return ((p[0] - 0.5) * (win.size[0] / win.size[1]), -p[1] + 0.5)
    elif units in ["pix", "cm", "deg", "degFlat", "degFlatPos"]:
        # Convert to pixel units first
        p_pix = tobii2pix(win, p)
        if units == "pix":
            return p_pix
        elif units == "cm":
            # Convert pixels to centimeters
            return tuple(pix2cm(pos, win.monitor) for pos in p_pix)
        elif units == "deg":
            # Convert pixels to degrees
            return tuple(pix2deg(pos, win.monitor) for pos in p_pix)
        else:
            # Convert pixels to degrees with correction for flatness
            return tuple(pix2deg(np.array(p_pix), win.monitor, correctFlat=True))
    else:
        raise ValueError(f"unit ({units}) is not supported.")


def psychopy_to_pixels(win, pos):
    """
    Convert PsychoPy coordinates to pixel coordinates.
    
    Parameters
    ----------
    win : psychopy.visual.Window
        The PsychoPy window which provides information about units and size.
    pos : tuple
        The PsychoPy coordinates to convert (x, y).
    
    Returns
    -------
    tuple
        The converted pixel coordinates as (int, int).
    
    Notes
    -----
    This function handles the main PsychoPy coordinate systems:
    - 'height': Screen height = 1, width adjusted by aspect ratio
    - 'norm': Screen ranges from -1 to 1 in both dimensions
    - Other units: Assumes coordinates are already close to pixel values
    """
    if win.units == 'height':
        # Convert height units to pixels
        x_pix = (pos[0] * win.size[1] + win.size[0]/2)
        y_pix = (-pos[1] * win.size[1] + win.size[1]/2)
    elif win.units == 'norm':
        # Convert normalized units to pixels
        x_pix = (pos[0] + 1) * win.size[0] / 2
        y_pix = (1 - pos[1]) * win.size[1] / 2
    else:
        # Handle other units - assume they're already close to pixels
        x_pix = pos[0] + win.size[0]/2
        y_pix = -pos[1] + win.size[1]/2
    
    return (int(x_pix), int(y_pix))



def get_tobii_pos(win, p, units=None):
    """
    Convert PsychoPy coordinates to Tobii ADCS coordinates.

    Parameters
    ----------
    win : psychopy.visual.Window
        The PsychoPy window which provides information about units and size.
    p : tuple
        The PsychoPy coordinates to convert.
    units : str, optional
        The units for the PsychoPy coordinates. Default is None, which uses the window's default units.

    Returns
    -------
    tuple
        The converted Tobii ADCS coordinates.

    Raises
    ------
    ValueError
        If the provided units are not supported.
    """
    if units is None:
        units = win.units

    if units == "norm":
        # Convert to normalized units, where screen ranges from 0 to 1
        return (p[0] / 2 + 0.5, p[1] / -2 + 0.5)
    elif units == "height":
        # Convert to height units, where screen height is 1 and width is adjusted
        return (p[0] * (win.size[1] / win.size[0]) + 0.5, -p[1] + 0.5)
    elif units == "pix":
        # Convert to pixel units
        return pix2tobii(win, p)
    elif units in ["cm", "deg", "degFlat", "degFlatPos"]:
        # Convert to pixel units first
        if units == "cm":
            p_pix = (cm2pix(p[0], win.monitor), cm2pix(p[1], win.monitor))
        elif units == "deg":
            p_pix = (deg2pix(p[0], win.monitor), deg2pix(p[1], win.monitor))
        elif units in ["degFlat", "degFlatPos"]:
            p_pix = deg2pix(np.array(p), win.monitor, correctFlat=True)
        p_pix = tuple(round(pos, 0) for pos in p_pix)
        # Convert pixels to Tobii ADCS coordinates
        return pix2tobii(win, p_pix)
    else:
        raise ValueError(f"unit ({units}) is not supported")


def pix2tobii(win, p):
    """
    Convert PsychoPy pixel coordinates to Tobii ADCS coordinates.

    Parameters
    ----------
    win : psychopy.visual.Window
        The PsychoPy window which provides information about units and size.
    p : tuple
        The PsychoPy pixel coordinates to convert.

    Returns
    -------
    tuple
        The converted Tobii ADCS coordinates.

    Notes
    -----
    The conversion is done by dividing the pixel coordinates by the window size
    and adding 0.5 to the x and y coordinates to center the origin at the
    middle of the screen.
    """
    return (p[0] / win.size[0] + 0.5, -p[1] / win.size[1] + 0.5)


def tobii2pix(win, p):
    """
    Convert Tobii ADCS coordinates to PsychoPy pixel coordinates.

    Parameters
    ----------
    win : psychopy.visual.Window
        The PsychoPy window which provides information about units and size.
    p : tuple
        The Tobii ADCS coordinates to convert.

    Returns
    -------
    tuple
        The converted PsychoPy pixel coordinates.

    Notes
    -----
    The conversion is done by multiplying the ADCS coordinates by the window
    size and subtracting 0.5 from the x and y coordinates to move the origin
    to the top-left corner of the screen.
    """
    return (round(win.size[0] * (p[0] - 0.5), 0), 
            round(-win.size[1] * (p[1] - 0.5), 0))


def get_psychopy_pos_from_trackbox(win, p, units=None):
    """
    Convert Tobii TBCS coordinates to PsychoPy coordinates.

    Parameters
    ----------
    win : psychopy.visual.Window
        The PsychoPy window which provides information about units and size.
    p : tuple
        The Tobii TBCS coordinates to convert.
    units : str, optional
        The units for the PsychoPy coordinates. Default is None, which uses the window's default units.

    Returns
    -------
    tuple
        The converted PsychoPy coordinates.

    Raises
    ------
    ValueError
        If the provided units are not supported.
    """
    if units is None:
        units = win.units

    if units == "norm":
        # TBCS coordinates are in range [0, 1], so subtract from 1 to flip y
        return (-2 * p[0] + 1, -2 * p[1] + 1)
    elif units == "height":
        # Convert to height units, where screen height is 1 and width is adjusted
        return ((-p[0] + 0.5) * (win.size[0] / win.size[1]), -p[1] + 0.5)
    elif units in ["pix", "cm", "deg", "degFlat", "degFlatPos"]:
        # Convert to pixel units first
        p_pix = (round((-p[0] + 0.5) * win.size[0], 0),
                 round((-p[1] + 0.5) * win.size[1], 0))
        if units == "pix":
            return p_pix
        elif units == "cm":
            # Convert pixels to centimeters
            return tuple(pix2cm(pos, win.monitor) for pos in p_pix)
        elif units == "deg":
            # Convert pixels to degrees
            return tuple(pix2deg(pos, win.monitor) for pos in p_pix)
        else:
            # Convert pixels to degrees with correction for flatness
            return tuple(pix2deg(np.array(p_pix), win.monitor, correctFlat=True))
    else:
        raise ValueError(f"unit ({units}) is not supported")
