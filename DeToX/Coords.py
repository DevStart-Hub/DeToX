# Third party imports
import numpy as np
from psychopy.tools.monitorunittools import cm2pix, deg2pix, pix2cm, pix2deg


def convert_height_to_units(win, height_value):
    """
    Convert a size from height units to the current window units.
    
    Provides unit-agnostic size conversion for consistent visual appearance across
    different PsychoPy coordinate systems. This function is essential for maintaining
    proper stimulus sizing when the window units differ from the standard height units
    used in configuration files.
    
    The conversion maintains the visual size of objects regardless of the coordinate
    system in use, ensuring that calibration targets, borders, and other visual
    elements appear at the intended size on screen.
    
    Parameters
    ----------
    win : psychopy.visual.Window
        The PsychoPy window which provides information about units and size.
        The window's current unit system determines the conversion method.
    height_value : float
        Size in height units (fraction of screen height). For example, 0.1
        represents 10% of the screen height.
        
    Returns
    -------
    float
        Size converted to current window units. The returned value maintains
        the same visual size on screen as the original height specification.
        
    Notes
    -----
    Height units are PsychoPy's recommended unit system for maintaining consistent
    appearance across different screen sizes and aspect ratios. This function
    enables that consistency when working with other unit systems.
    
    Supported unit conversions:
    - height: No conversion needed (identity transform)
    - norm: Scales by 2.0 to match normalized coordinate range
    - pix: Multiplies by screen height in pixels
    - cm/deg: Converts through pixels using monitor calibration
    """
    # --- Unit System Detection ---
    current_units = win.units
    
    if current_units == "height":
        # --- Identity Transform ---
        # Already in height units, no conversion needed
        return height_value
        
    elif current_units == "norm":
        # --- Normalized Units Conversion ---
        # In norm units, need to account for aspect ratio
        # Height of 1.0 in height units = height of 2.0 in norm units
        # But we want the same visual size, so scale by aspect ratio
        return height_value * 2.0
        
    elif current_units == "pix":
        # --- Pixel Units Conversion ---
        # Direct conversion: height fraction * screen height in pixels
        return height_value * win.size[1]
        
    elif current_units in ["cm", "deg", "degFlat", "degFlatPos"]:
        # --- Physical/Angular Units Conversion ---
        # Convert to pixels first, then use PsychoPy's conversion tools
        height_pixels = height_value * win.size[1]
        
        if current_units == "cm":
            # Convert pixels to centimeters using monitor calibration
            return pix2cm(height_pixels, win.monitor)
        elif current_units == "deg":
            # Convert pixels to visual degrees
            return pix2deg(height_pixels, win.monitor)
        else:  # degFlat, degFlatPos
            # Convert with flat screen correction
            return pix2deg(np.array([height_pixels]), win.monitor, correctFlat=True)[0]
    else:
        # --- Fallback ---
        # Unknown units - return as height units
        return height_value


def norm_to_window_units(win, norm_coords):
    """
    Convert normalized coordinates to current window units.
    
    Transforms calibration points from normalized units (-1 to +1 range) to
    the window's active coordinate system. Enables universal calibration patterns
    that work across different screens and unit systems.
    
    Parameters
    ----------
    win : psychopy.visual.Window
        PsychoPy window providing unit and size information.
    norm_coords : list of tuple
        Calibration points as (x, y) tuples in normalized coordinates [-1, 1].
        
    Returns
    -------
    list of tuple
        Points converted to current window units.
        
    Examples
    --------
    >>> from DeToX import ETSettings as cfg
    >>> cal_points = norm_to_window_units(win, cfg.calibration_5_points)
    """
    # --- Unit System Detection ---
    current_units = win.units
    
    # --- Conversion Based on Current Units ---
    converted = []
    
    for x_norm, y_norm in norm_coords:
        if current_units == "norm":
            converted.append((x_norm, y_norm))
            
        elif current_units == "height":
            aspect = win.size[0] / win.size[1]
            x_height = x_norm * (aspect / 2.0)
            y_height = y_norm * 0.5
            converted.append((x_height, y_height))
            
        elif current_units == "pix":
            x_pix = x_norm * (win.size[0] / 2.0)
            y_pix = y_norm * (win.size[1] / 2.0)
            converted.append((x_pix, y_pix))
            
        elif current_units in ["cm", "deg", "degFlat", "degFlatPos"]:
            x_pix = x_norm * (win.size[0] / 2.0)
            y_pix = y_norm * (win.size[1] / 2.0)
            
            if current_units == "cm":
                converted.append((pix2cm(x_pix, win.monitor), pix2cm(y_pix, win.monitor)))
            elif current_units == "deg":
                converted.append((pix2deg(x_pix, win.monitor), pix2deg(y_pix, win.monitor)))
            else:
                converted.append((pix2deg(x_pix, win.monitor, correctFlat=True), 
                                pix2deg(y_pix, win.monitor, correctFlat=True)))
        else:
            converted.append((x_norm, y_norm))
    
    return converted


def get_psychopy_pos(win, p, units=None):
    """
    Convert Tobii ADCS coordinates to PsychoPy coordinates.
    
    Transforms eye tracker coordinates from Tobii's Active Display Coordinate System
    (ADCS) to PsychoPy's coordinate system. ADCS uses normalized coordinates where
    (0,0) is top-left and (1,1) is bottom-right, while PsychoPy typically uses
    centered coordinates with various unit systems.
    
    This function is critical for correctly positioning gaze data within PsychoPy
    stimuli and for accurate visualization of eye tracking results.

    Parameters
    ----------
    win : psychopy.visual.Window
        The PsychoPy window which provides information about units and size.
        Window properties determine the target coordinate system.
    p : tuple or array-like
        The Tobii ADCS coordinates to convert. Can be:
        - Single coordinate: (x, y) tuple
        - Multiple coordinates: (N, 2) array where N is number of samples
        Values should be in range [0, 1] where (0, 0) is top-left and (1, 1) is bottom-right.
    units : str, optional
        The target units for the PsychoPy coordinates. If None, uses the
        window's default units. Supported: 'norm', 'height', 'pix', 'cm',
        'deg', 'degFlat', 'degFlatPos'.

    Returns
    -------
    tuple or ndarray
        The converted PsychoPy coordinates in the specified unit system.
        - Single input: returns (x, y) tuple
        - Array input: returns (N, 2) array
        Origin is at screen center for most unit systems.

    Raises
    ------
    ValueError
        If the provided units are not supported by PsychoPy.
        
    Examples
    --------
    >>> # Single coordinate
    >>> pos = get_psychopy_pos(win, (0.5, 0.5))  # Returns (0, 0) in most units
    
    >>> # Multiple coordinates (vectorized)
    >>> coords = np.array([[0.5, 0.5], [0.0, 0.0], [1.0, 1.0]])
    >>> positions = get_psychopy_pos(win, coords)  # Returns (N, 2) array
    """
    # --- Unit System Resolution ---
    if units is None:
        units = win.units

    # --- Check if input is array or single coordinate ---
    p_array = np.asarray(p)
    is_single = (p_array.ndim == 1)
    
    # Ensure we have a 2D array for processing
    if is_single:
        p_array = p_array.reshape(1, -1)
    
    # Extract x and y columns
    x = p_array[:, 0]
    y = p_array[:, 1]

    if units == "norm":
        # --- Normalized Units ---
        # Convert to normalized units, where screen ranges from -1 to 1
        # ADCS (0,1) -> norm (-1,1) with Y-axis inversion
        result_x = 2 * x - 1
        result_y = -2 * y + 1
        
    elif units == "height": 
        # --- Height Units ---
        # Convert to height units, where screen height is 1 and width is adjusted
        # Maintains aspect ratio with centered origin
        aspect = win.size[0] / win.size[1]
        result_x = (x - 0.5) * aspect
        result_y = -y + 0.5
        
    elif units == "pix":
        # --- Pixel Units ---
        result_x = (x - 0.5) * win.size[0]
        result_y = -(y - 0.5) * win.size[1]
        
    elif units in ["cm", "deg", "degFlat", "degFlatPos"]:
        # --- Physical and Pixel Units ---
        # Convert to pixel units first as intermediate step
        x_pix = (x - 0.5) * win.size[0]
        y_pix = -(y - 0.5) * win.size[1]
        
        if units == "cm":
            # Convert pixels to centimeters using monitor calibration
            result_x = pix2cm(x_pix, win.monitor)
            result_y = pix2cm(y_pix, win.monitor)
        elif units == "deg":
            # Convert pixels to visual degrees
            result_x = pix2deg(x_pix, win.monitor)
            result_y = pix2deg(y_pix, win.monitor)
        else:
            # Convert pixels to degrees with flat screen correction
            result_x = pix2deg(x_pix, win.monitor, correctFlat=True)
            result_y = pix2deg(y_pix, win.monitor, correctFlat=True)
    else:
        # --- Unsupported Units ---
        raise ValueError(f"unit ({units}) is not supported.")
    
    # --- Return in original format ---
    if is_single:
        return (float(result_x[0]), float(result_y[0]))
    else:
        return np.column_stack([result_x, result_y])


def psychopy_to_pixels(win, pos):
    """
    Convert PsychoPy coordinates to pixel coordinates.
    
    Transforms coordinates from any PsychoPy coordinate system to pixel coordinates
    suitable for image drawing operations. This function is essential for creating
    calibration result visualizations and other pixel-based graphics.
    
    The conversion accounts for PsychoPy's centered coordinate system and transforms
    to a top-left origin system used by image libraries like PIL.
    
    Parameters
    ----------
    win : psychopy.visual.Window
        The PsychoPy window which provides information about units and size.
        Window units and dimensions determine the conversion method.
    pos : tuple
        The PsychoPy coordinates to convert as (x, y) in current window units.
    
    Returns
    -------
    tuple
        The converted pixel coordinates as (int, int) with origin at top-left.
        Values are rounded to nearest integer for pixel alignment.
    
    Notes
    -----
    This function handles the main PsychoPy coordinate systems:
    - 'height': Screen height = 1, width adjusted by aspect ratio, centered origin
    - 'norm': Screen ranges from -1 to 1 in both dimensions, centered origin
    - Other units: Assumes coordinates are already close to pixel values
    
    The output uses standard image coordinates where (0,0) is top-left and
    y increases downward, suitable for PIL and similar libraries.
    """
    if win.units == 'height':
        # --- Height Units to Pixels ---
        # Convert height units to pixels with aspect ratio correction
        x_pix = (pos[0] * win.size[1] + win.size[0]/2)
        y_pix = (-pos[1] * win.size[1] + win.size[1]/2)
        
    elif win.units == 'norm':
        # --- Normalized Units to Pixels ---
        # Convert normalized units (-1 to 1) to pixels
        x_pix = (pos[0] + 1) * win.size[0] / 2
        y_pix = (1 - pos[1]) * win.size[1] / 2
        
    else:
        # --- Other Units ---
        # Handle other units - assume they're already close to pixels
        # Apply centering transformation
        x_pix = pos[0] + win.size[0]/2
        y_pix = -pos[1] + win.size[1]/2
    
    # --- Integer Conversion ---
    # Round to nearest pixel for clean rendering
    return (int(x_pix), int(y_pix))


def get_tobii_pos(win, p, units=None):
    """
    Convert PsychoPy coordinates to Tobii ADCS coordinates.
    
    Transforms coordinates from PsychoPy's coordinate system to Tobii's Active
    Display Coordinate System (ADCS). This conversion is essential for sending
    calibration target positions to the Tobii eye tracker during calibration
    procedures.
    
    ADCS uses normalized coordinates where (0,0) is top-left and (1,1) is
    bottom-right, regardless of screen size or resolution. This provides a
    hardware-independent coordinate system for eye tracking data.

    Parameters
    ----------
    win : psychopy.visual.Window
        The PsychoPy window which provides information about units and size.
        Window properties determine the source coordinate system.
    p : tuple
        The PsychoPy coordinates to convert as (x, y) in specified units.
    units : str, optional
        The units of the input PsychoPy coordinates. If None, uses the
        window's default units. Supported: 'norm', 'height', 'pix', 'cm',
        'deg', 'degFlat', 'degFlatPos'.

    Returns
    -------
    tuple
        The converted Tobii ADCS coordinates as (x, y) where both values
        are in range [0, 1]. (0, 0) is top-left, (1, 1) is bottom-right.

    Raises
    ------
    ValueError
        If the provided units are not supported.
        
    Notes
    -----
    This function is the inverse of get_psychopy_pos() and is primarily used
    during calibration to inform the eye tracker where calibration targets
    are displayed on screen.
    """
    # --- Unit System Resolution ---
    if units is None:
        units = win.units

    if units == "norm":
        # --- Normalized Units to ADCS ---
        # Convert from normalized units where screen ranges from -1 to 1
        # to ADCS where screen ranges from 0 to 1
        return (p[0] / 2 + 0.5, p[1] / -2 + 0.5)
        
    elif units == "height":
        # --- Height Units to ADCS ---
        # Convert from height units with aspect ratio adjustment
        # Account for centered origin and Y-axis direction
        return (p[0] * (win.size[1] / win.size[0]) + 0.5, -p[1] + 0.5)
        
    elif units == "pix":
        # --- Pixel Units to ADCS ---
        # Direct conversion from pixels
        return pix2tobii(win, p)
        
    elif units in ["cm", "deg", "degFlat", "degFlatPos"]:
        # --- Physical/Angular Units to ADCS ---
        # Convert to pixel units first as intermediate step
        if units == "cm":
            # Convert centimeters to pixels
            p_pix = (cm2pix(p[0], win.monitor), cm2pix(p[1], win.monitor))
        elif units == "deg":
            # Convert visual degrees to pixels
            p_pix = (deg2pix(p[0], win.monitor), deg2pix(p[1], win.monitor))
        elif units in ["degFlat", "degFlatPos"]:
            # Convert degrees with flat screen correction
            p_pix = deg2pix(np.array(p), win.monitor, correctFlat=True)
            
        # Round to nearest pixel
        p_pix = tuple(round(pos, 0) for pos in p_pix)
        
        # Convert pixels to Tobii ADCS coordinates
        return pix2tobii(win, p_pix)
    else:
        # --- Unsupported Units ---
        raise ValueError(f"unit ({units}) is not supported")


def pix2tobii(win, p):
    """
    Convert PsychoPy pixel coordinates to Tobii ADCS coordinates.
    
    Low-level conversion function that transforms pixel coordinates with a
    centered origin (PsychoPy convention) to Tobii's normalized ADCS coordinates
    with top-left origin. This is a fundamental building block for other
    coordinate conversion functions.

    Parameters
    ----------
    win : psychopy.visual.Window
        The PsychoPy window which provides screen dimensions for normalization.
    p : tuple
        The PsychoPy pixel coordinates to convert as (x, y). Origin is at
        screen center, x increases rightward, y increases upward.

    Returns
    -------
    tuple
        The converted Tobii ADCS coordinates as (x, y) in range [0, 1].
        Origin is top-left, x increases rightward, y increases downward.

    Notes
    -----
    The conversion involves:
    1. Translating the origin from center to top-left (+0.5 offset)
    2. Normalizing by screen dimensions to get [0, 1] range
    3. Inverting the Y-axis to match Tobii's top-down convention
    
    This function assumes PsychoPy's pixel coordinate convention where
    (0, 0) is at screen center.
    """
    # --- Coordinate Transformation ---
    # Normalize by screen size and shift origin from center to top-left
    # Y-axis is inverted to match Tobii's top-down convention
    return (p[0] / win.size[0] + 0.5, -p[1] / win.size[1] + 0.5)


def tobii2pix(win, p):
    """
    Convert Tobii ADCS coordinates to PsychoPy pixel coordinates.
    
    Low-level conversion function that transforms Tobii's normalized ADCS
    coordinates to PsychoPy pixel coordinates. This is the inverse of pix2tobii()
    and is essential for displaying gaze data in PsychoPy windows.

    Parameters
    ----------
    win : psychopy.visual.Window
        The PsychoPy window which provides screen dimensions for scaling.
    p : tuple
        The Tobii ADCS coordinates to convert as (x, y) in range [0, 1].
        Origin is top-left, x increases rightward, y increases downward.

    Returns
    -------
    tuple
        The converted PsychoPy pixel coordinates as (x, y) with origin at
        screen center. Values are rounded to nearest integer for pixel alignment.

    Notes
    -----
    The conversion involves:
    1. Shifting origin from top-left to center (-0.5 offset)
    2. Scaling by screen dimensions to get pixel values
    3. Inverting the Y-axis to match PsychoPy's bottom-up convention
    
    Output coordinates follow PsychoPy's pixel convention where (0, 0)
    is at screen center.
    """
    # --- Coordinate Transformation ---
    # Scale by screen size and shift origin from top-left to center
    # Y-axis is inverted to match PsychoPy's bottom-up convention
    return (round(win.size[0] * (p[0] - 0.5), 0), 
            round(-win.size[1] * (p[1] - 0.5), 0))


def get_psychopy_pos_from_trackbox(win, p, units=None):
    """
    Convert Tobii TBCS coordinates to PsychoPy coordinates.
    
    Transforms coordinates from Tobii's Track Box Coordinate System (TBCS) to
    PsychoPy's coordinate system. TBCS is used for the user position guide,
    showing where the participant's eyes are located within the eye tracker's
    track box (the 3D volume where eyes can be tracked).
    
    In TBCS, coordinates represent position within the track box where (0,0)
    indicates the participant is positioned at the right edge from the tracker's
    perspective, and (1,1) indicates the left edge. This apparent reversal is
    because TBCS uses the tracker's perspective, not the user's.

    Parameters
    ----------
    win : psychopy.visual.Window
        The PsychoPy window which provides information about units and size.
        Window properties determine the target coordinate system.
    p : tuple
        The Tobii TBCS coordinates to convert as (x, y). Values are in range
        [0, 1] representing position within the track box from the tracker's
        perspective.
    units : str, optional
        The target units for the PsychoPy coordinates. If None, uses the
        window's default units. Supported: 'norm', 'height', 'pix', 'cm',
        'deg', 'degFlat', 'degFlatPos'.

    Returns
    -------
    tuple
        The converted PsychoPy coordinates in the specified unit system.
        Suitable for positioning visual feedback about user position.

    Raises
    ------
    ValueError
        If the provided units are not supported.
        
    Notes
    -----
    TBCS coordinates are primarily used in the show_status() method to provide
    visual feedback about participant positioning. The X-axis is reversed
    compared to ADCS because TBCS uses the tracker's perspective.
    
    This function handles the perspective reversal and transforms to PsychoPy's
    coordinate conventions for proper visualization.
    """
    # --- Unit System Resolution ---
    if units is None:
        units = win.units

    if units == "norm":
        # --- Normalized Units ---
        # TBCS coordinates are in range [0, 1], so subtract from 1 to flip x
        # Note the x-axis reversal due to tracker perspective
        return (-2 * p[0] + 1, -2 * p[1] + 1)
        
    elif units == "height":
        # --- Height Units ---
        # Convert to height units with aspect ratio adjustment
        # X-axis is reversed for tracker perspective
        return ((-p[0] + 0.5) * (win.size[0] / win.size[1]), -p[1] + 0.5)
        
    elif units in ["pix", "cm", "deg", "degFlat", "degFlatPos"]:
        # --- Physical and Pixel Units ---
        # Convert to pixel units first with perspective reversal
        p_pix = (round((-p[0] + 0.5) * win.size[0], 0),
                 round((-p[1] + 0.5) * win.size[1], 0))
                 
        if units == "pix":
            # Return pixel coordinates directly
            return p_pix
        elif units == "cm":
            # Convert pixels to centimeters
            return tuple(pix2cm(pos, win.monitor) for pos in p_pix)
        elif units == "deg":
            # Convert pixels to visual degrees
            return tuple(pix2deg(pos, win.monitor) for pos in p_pix)
        else:
            # Convert pixels to degrees with flat screen correction
            return tuple(pix2deg(np.array(p_pix), win.monitor, correctFlat=True))
    else:
        # --- Unsupported Units ---
        raise ValueError(f"unit ({units}) is not supported")