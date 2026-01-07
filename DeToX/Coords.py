"""Coordinate conversion utilities for eye tracking data.

This module provides coordinate system transformations between Tobii Pro SDK
and PsychoPy, enabling integration of eye tracking data with visual
stimuli. All functions handle multiple PsychoPy unit systems (height, norm, 
pix, cm, deg) and maintain coordinate consistency across different displays.

Main coordinate systems:
    - ADCS (Active Display Coordinate System): Tobii's normalized 0-1 system
      where (0,0) is top-left and (1,1) is bottom-right
    - User Position coordinates: Tobii's normalized 0-1 system for eye position
      in the tracking volume (from User Position Guide stream)
    - PsychoPy: Centered coordinate system with configurable units

Typical usage:
    ```python
    from DeToX import Coords
    
    # Convert gaze data to PsychoPy coordinates
    gaze_pos = Coords.get_psychopy_pos(win, (0.5, 0.3))
    
    # Convert calibration target to Tobii coordinates
    tobii_pos = Coords.get_tobii_pos(win, (0.1, -0.2))
    ```
"""

# Third party imports
import numpy as np
from psychopy.tools.monitorunittools import cm2pix, deg2pix, pix2cm, pix2deg

def convert_height_to_units(win, height_value, target_units=None):
    """
    Convert a size from height units to the specified units.
    
    Parameters
    ----------
    win : psychopy.visual.Window
        The PsychoPy window providing unit and size information.
    height_value : float
        Size in height units (fraction of screen height).
    target_units : str, optional
        Target unit system. If None, uses window's current units.
        
    Returns
    -------
    float
        Size converted to target units.
    """
    if target_units is None:
        target_units = win.units
    
    if target_units == "height":
        return height_value
        
    elif target_units == "norm":
        return height_value * 2.0
        
    elif target_units == "pix":
        return height_value * win.size[1]
        
    elif target_units in ["cm", "deg", "degFlat", "degFlatPos"]:
        height_pixels = height_value * win.size[1]
        
        if target_units == "cm":
            return pix2cm(height_pixels, win.monitor)
        elif target_units == "deg":
            return pix2deg(height_pixels, win.monitor)
        else:  # degFlat, degFlatPos
            return pix2deg(np.array([height_pixels]), win.monitor, correctFlat=True)[0]
    else:
        return height_value


def get_psychopy_pos(win, p, units=None):
    """
    Convert Tobii ADCS coordinates to PsychoPy coordinates.
    
    Transforms eye tracker coordinates from Tobii's Active Display Coordinate System
    (ADCS) to PsychoPy's coordinate system. ADCS uses normalized coordinates where
    (0,0) is top-left and (1,1) is bottom-right. This function is critical for
    correctly positioning gaze data within PsychoPy stimuli.
    
    Supports both single coordinate conversion and vectorized batch conversion
    for efficient processing of recorded gaze data.

    Parameters
    ----------
    win : psychopy.visual.Window
        The PsychoPy window providing unit and size information.
    p : tuple or array-like
        Tobii ADCS coordinates to convert:
        - Single coordinate: (x, y) tuple
        - Multiple coordinates: (N, 2) array where N is number of samples
        Values should be in range [0, 1] where (0,0) is top-left.
    units : str, optional
        Target PsychoPy units. If None, uses window's default units.
        Supported: 'norm', 'height', 'pix', 'cm', 'deg', 'degFlat', 'degFlatPos'.

    Returns
    -------
    tuple or ndarray
        Converted PsychoPy coordinates in specified unit system:
        - Single input: returns (x, y) tuple
        - Array input: returns (N, 2) array
        Origin is at screen center for most unit systems.

    Raises
    ------
    ValueError
        If the provided units are not supported.
        
    Examples
    --------
    ```python
    from DeToX import Coords
    import numpy as np
    
    # Single coordinate conversion
    tobii_gaze = (0.5, 0.5)  # Center of screen in ADCS
    psychopy_gaze = Coords.get_psychopy_pos(win, tobii_gaze)
    # Returns (0.0, 0.0) in most units - screen center
    
    # Vectorized conversion for recorded data (efficient!)
    recorded_gaze = np.array([
        [0.5, 0.5],   # Center
        [0.0, 0.0],   # Top-left
        [1.0, 1.0]    # Bottom-right
    ])
    psychopy_positions = Coords.get_psychopy_pos(win, recorded_gaze)
    # Returns (3, 2) array with all positions converted
    
    # Specify target units explicitly
    gaze_in_pixels = Coords.get_psychopy_pos(win, (0.5, 0.3), units='pix')
    ```
    """
    if units is None:
        units = win.units

    p_array = np.asarray(p)
    is_single = (p_array.ndim == 1)
    
    if is_single:
        p_array = p_array.reshape(1, -1)
    
    x = p_array[:, 0]
    y = p_array[:, 1]

    if units == "norm":
        result_x = 2 * x - 1
        result_y = -2 * y + 1
        
    elif units == "height": 
        aspect = win.size[0] / win.size[1]
        result_x = (x - 0.5) * aspect
        result_y = -y + 0.5
        
    elif units == "pix":
        result_x = (x - 0.5) * win.size[0]
        result_y = -(y - 0.5) * win.size[1]
        
    elif units in ["cm", "deg", "degFlat", "degFlatPos"]:
        x_pix = (x - 0.5) * win.size[0]
        y_pix = -(y - 0.5) * win.size[1]
        
        if units == "cm":
            result_x = pix2cm(x_pix, win.monitor)
            result_y = pix2cm(y_pix, win.monitor)
        elif units == "deg":
            result_x = pix2deg(x_pix, win.monitor)
            result_y = pix2deg(y_pix, win.monitor)
        else:
            result_x = pix2deg(x_pix, win.monitor, correctFlat=True)
            result_y = pix2deg(y_pix, win.monitor, correctFlat=True)
    else:
        raise ValueError(f"unit ({units}) is not supported.")
    
    if is_single:
        return (float(result_x[0]), float(result_y[0]))
    else:
        return np.column_stack([result_x, result_y])


def psychopy_to_pixels(win, pos):
    """
    Convert PsychoPy coordinates to pixel coordinates for image drawing.
    
    Transforms coordinates from any PsychoPy coordinate system to pixel coordinates
    with top-left origin, suitable for PIL image drawing operations. Essential for
    creating calibration result visualizations.
    
    Supports both single coordinate conversion and vectorized batch conversion
    for efficient processing of multiple positions.
    
    Parameters
    ----------
    win : psychopy.visual.Window
        The PsychoPy window providing unit and size information.
    pos : tuple or array-like
        PsychoPy coordinates to convert:
        - Single coordinate: (x, y) tuple in current window units
        - Multiple coordinates: (N, 2) array where N is number of positions
    
    Returns
    -------
    tuple or ndarray
        Pixel coordinates with origin at top-left:
        - Single input: returns (int, int) tuple
        - Array input: returns (N, 2) array of integers
        Values are rounded to nearest integer for pixel alignment.
    
    Notes
    -----
    The output uses standard image coordinates where (0,0) is top-left and
    y increases downward, suitable for PIL and similar libraries.
    
    Supported input units: height, norm, pix, and others (treated as centered)
    
    Examples
    --------
    ```python
        from DeToX import Coords
        from PIL import Image, ImageDraw
        import numpy as np
        
        # Single coordinate conversion
        target_pos = (0.2, -0.1)  # PsychoPy coordinates (height units)
        pixel_pos = Coords.psychopy_to_pixels(win, target_pos)
        # Returns (1152, 432) for 1920x1080 display
        
        # Vectorized batch conversion (efficient!)
        gaze_positions = np.array([
            [0.2, -0.1],
            [-0.1, 0.3],
            [0.0, 0.0]
        ])
        pixel_positions = Coords.psychopy_to_pixels(win, gaze_positions)
        # Returns (3, 2) array: [[1152, 432], [864, 216], [960, 540]]
        
        # Use for drawing calibration results with multiple samples
        img = Image.new("RGBA", tuple(win.size))
        draw = ImageDraw.Draw(img)
        
        # Convert all sample positions at once (vectorized!)
        sample_positions = np.array([(0.1, 0.2), (-0.1, 0.1), (0.3, -0.2)])
        samples_pix = Coords.psychopy_to_pixels(win, sample_positions)
        
        # Draw circles at each sample position
        for sample_pix in samples_pix:
            draw.ellipse([sample_pix[0]-5, sample_pix[1]-5, 
                        sample_pix[0]+5, sample_pix[1]+5], fill='red')
    ```
    """
    # Check if input is single coordinate or array
    pos_array = np.asarray(pos)
    is_single = (pos_array.ndim == 1)
    
    # Reshape single input for vectorized processing
    if is_single:
        pos_array = pos_array.reshape(1, -1)
    
    # Extract x and y columns
    x = pos_array[:, 0]
    y = pos_array[:, 1]
    
    # Vectorized conversion based on window units
    if win.units == 'height':
        x_pix = x * win.size[1] + win.size[0] / 2
        y_pix = -y * win.size[1] + win.size[1] / 2
        
    elif win.units == 'norm':
        x_pix = (x + 1) * win.size[0] / 2
        y_pix = (1 - y) * win.size[1] / 2
        
    else:
        x_pix = x + win.size[0] / 2
        y_pix = -y + win.size[1] / 2
    
    # Round to integers for pixel alignment
    x_pix = np.round(x_pix).astype(int)
    y_pix = np.round(y_pix).astype(int)
    
    # Return in original format
    if is_single:
        return (int(x_pix[0]), int(y_pix[0]))
    else:
        return np.column_stack([x_pix, y_pix])
        
        
def get_tobii_pos(win, p, source_units=None):
    """
    Convert PsychoPy coordinates to Tobii ADCS coordinates.
    
    Transforms coordinates from PsychoPy's coordinate system to Tobii's Active
    Display Coordinate System (ADCS). Essential for sending calibration target
    positions to the Tobii eye tracker during calibration procedures.
    
    ADCS uses normalized coordinates where (0,0) is top-left and (1,1) is
    bottom-right, providing a hardware-independent coordinate system.
    
    Supports both single coordinate conversion and vectorized batch conversion
    for efficient processing of multiple positions.

    Parameters
    ----------
    win : psychopy.visual.Window
        The PsychoPy window providing unit and size information.
    p : tuple or array-like
        PsychoPy coordinates to convert:
        - Single coordinate: (x, y) tuple in specified units
        - Multiple coordinates: (N, 2) array where N is number of positions
    source_units : str, optional
        Units of the input coordinates. If None, uses window's default units.
        Supported: 'norm', 'height', 'pix', 'cm', 'deg', 'degFlat', 'degFlatPos'.

    Returns
    -------
    tuple or ndarray
        Tobii ADCS coordinates where values are in range [0, 1]:
        - Single input: returns (x, y) tuple
        - Array input: returns (N, 2) array
        (0, 0) is top-left, (1, 1) is bottom-right.

    Raises
    ------
    ValueError
        If the provided units are not supported.
        
    Notes
    -----
    This function is the inverse of get_psychopy_pos() and is primarily used
    during calibration to inform the eye tracker where targets are displayed.
    
    Examples
    --------
    ```python
        from DeToX import Coords
        import numpy as np
        
        # Single coordinate conversion
        target_psychopy = (0.2, -0.1)  # Height units
        target_tobii = Coords.get_tobii_pos(win, target_psychopy, source_units='height')
        # Returns (x, y) in [0, 1] range for Tobii SDK
        
        # Vectorized conversion for multiple points (efficient!)
        targets = np.array([
            [0.2, -0.1],
            [-0.1, 0.3],
            [0.0, 0.0]
        ])
        tobii_positions = Coords.get_tobii_pos(win, targets, source_units='height')
        # Returns (3, 2) array with all positions converted
        
        # Use during calibration
        target_tobii = Coords.get_tobii_pos(win, target_psychopy, source_units='height')
        calibration.collect_data(target_tobii[0], target_tobii[1])
        
        # Convert from different unit systems
        target_norm = (-0.5, 0.5)  # Normalized units
        tobii_pos = Coords.get_tobii_pos(win, target_norm, source_units='norm')
        
        # Works with pixel coordinates too
        target_pix = (960, 540)  # Center of 1920x1080 screen
        tobii_pos = Coords.get_tobii_pos(win, target_pix, source_units='pix')
        # Returns (0.5, 0.5) - center in ADCS
    ```
    """
    # --- Units Detection ---
    if source_units is None:
        source_units = win.units
    
    # --- Vectorization Setup ---
    p_array = np.asarray(p)
    is_single = (p_array.ndim == 1)
    
    if is_single:
        p_array = p_array.reshape(1, -1)
    
    x = p_array[:, 0]
    y = p_array[:, 1]
    
    # --- Unit-Specific Conversion ---
    if source_units == "norm":
        result_x = x / 2 + 0.5
        result_y = -y / 2 + 0.5
        
    elif source_units == "height":
        aspect_ratio = win.size[1] / win.size[0]
        result_x = x * aspect_ratio + 0.5
        result_y = -y + 0.5
        
    elif source_units == "pix":
        result_x = x / win.size[0] + 0.5
        result_y = -y / win.size[1] + 0.5
        
    elif source_units in ["cm", "deg", "degFlat", "degFlatPos"]:
        # Convert to pixels first
        if source_units == "cm":
            x_pix = cm2pix(x, win.monitor)
            y_pix = cm2pix(y, win.monitor)
        elif source_units == "deg":
            x_pix = deg2pix(x, win.monitor)
            y_pix = deg2pix(y, win.monitor)
        else:  # degFlat, degFlatPos
            x_pix = deg2pix(x, win.monitor, correctFlat=True)
            y_pix = deg2pix(y, win.monitor, correctFlat=True)
        
        # Convert pixels to Tobii
        result_x = x_pix / win.size[0] + 0.5
        result_y = -y_pix / win.size[1] + 0.5
    else:
        raise ValueError(f"unit ({source_units}) is not supported")
    
    # --- Return in Original Format ---
    if is_single:
        return (float(result_x[0]), float(result_y[0]))
    else:
        return np.column_stack([result_x, result_y])


def pix2tobii(win, p):
    """
    Convert PsychoPy pixel coordinates to Tobii ADCS coordinates.
    
    Low-level conversion function transforming pixel coordinates with centered
    origin (PsychoPy convention) to Tobii's normalized ADCS coordinates with
    top-left origin. Used internally by get_tobii_pos().

    Parameters
    ----------
    win : psychopy.visual.Window
        The PsychoPy window providing screen dimensions for normalization.
    p : tuple
        PsychoPy pixel coordinates as (x, y). Origin at screen center,
        x increases rightward, y increases upward.

    Returns
    -------
    tuple
        Tobii ADCS coordinates as (x, y) in range [0, 1].
        Origin is top-left, x increases rightward, y increases downward.

    Notes
    -----
    The conversion process:
    1. Normalize by screen dimensions to get [0, 1] range
    2. Translate origin from center to top-left (+0.5 offset)
    3. Invert Y-axis to match Tobii's top-down convention
    
    Examples
    --------
    ```python
    from DeToX import Coords
    
    # Center of 1920x1080 screen
    pixel_center = (0, 0)  # PsychoPy pixel center
    tobii_center = Coords.pix2tobii(win, pixel_center)
    # Returns (0.5, 0.5) - center in ADCS
    
    # Top-left corner
    pixel_tl = (-960, 540)
    tobii_tl = Coords.pix2tobii(win, pixel_tl)
    # Returns (0.0, 0.0) - top-left in ADCS
    ```
    """
    return (p[0] / win.size[0] + 0.5, -p[1] / win.size[1] + 0.5)


def get_psychopy_pos_from_user_position(win, p, units=None):
    """
    Convert User Position Guide coordinates to PsychoPy coordinates.
    
    Transforms coordinates from Tobii's User Position Guide stream
    to PsychoPy's coordinate system. The User Position Guide provides real-time
    eye position within the eye tracker's detection volume, used for positioning
    feedback displays.
    
    User Position coordinates are normalized 0-1 values representing eye location
    in the tracking volume. The coordinate system uses the tracker's perspective:
    X: 0 (right edge) to 1 (left edge) - reversed from user's view
    Y: 0 (top) to 1 (bottom)
    Z: 0 (far) to 1 (near)

    Parameters
    ----------
    win : psychopy.visual.Window
        The PsychoPy window providing unit and size information.
    p : tuple
        User Position coordinates as (x, y). Values in range [0, 1]
        representing position within the tracking volume from tracker's perspective.
    units : str, optional
        Target PsychoPy units. If None, uses window's default units.
        Supported: 'norm', 'height', 'pix', 'cm', 'deg', 'degFlat', 'degFlatPos'.

    Returns
    -------
    tuple
        Converted PsychoPy coordinates in specified unit system.
        Suitable for positioning visual feedback about user position.

    Raises
    ------
    ValueError
        If the provided units are not supported.
        
    Notes
    -----
    SDK Compatibility: Works with Tobii Pro SDK 1.6+ and 2.x
    
    This function replaces the deprecated Track Box API (removed in SDK 2.1).
    It's used primarily in show_status() to provide real-time positioning
    feedback during setup.
    
    The X-axis is reversed compared to ADCS because User Position coordinates
    use the tracker's perspective, not the user's.
    
    Examples
    --------
    ```python
    from DeToX import Coords
    import tobii_research as tr
    
    # Subscribe to User Position Guide
    def position_callback(data):
        if data['left_eye']['validity']:
            x, y, z = data['left_eye']['user_position']
            
            # Convert to PsychoPy for visualization
            pos = Coords.get_psychopy_pos_from_user_position(win, [x, y], 'height')
            
            # Display eye position indicator
            eye_circle.pos = pos
            eye_circle.draw()
    
    eyetracker.subscribe_to(tr.EYETRACKER_USER_POSITION_GUIDE, 
                           position_callback, 
                           as_dictionary=True)
    
    # Manual conversion example
    user_pos = (0.5, 0.6)  # Centered horizontally, slightly below center
    screen_pos = Coords.get_psychopy_pos_from_user_position(win, user_pos)
    # Returns position for drawing positioning feedback
    ```
    """
    if units is None:
        units = win.units

    if units == "norm":
        return (-2 * p[0] + 1, -2 * p[1] + 1)
        
    elif units == "height":
        return ((-p[0] + 0.5) * (win.size[0] / win.size[1]), -p[1] + 0.5)
        
    elif units in ["pix", "cm", "deg", "degFlat", "degFlatPos"]:
        p_pix = (round((-p[0] + 0.5) * win.size[0], 0),
                 round((-p[1] + 0.5) * win.size[1], 0))
                 
        if units == "pix":
            return p_pix
        elif units == "cm":
            return tuple(pix2cm(pos, win.monitor) for pos in p_pix)
        elif units == "deg":
            return tuple(pix2deg(pos, win.monitor) for pos in p_pix)
        else:
            return tuple(pix2deg(np.array(p_pix), win.monitor, correctFlat=True))
    else:
        raise ValueError(f"unit ({units}) is not supported")


def norm_to_window_units(win, norm_coords, target_units=None):
    """
    Convert normalized coordinates to specified units.
    
    Parameters
    ----------
    win : psychopy.visual.Window
        PsychoPy window providing unit and size information.
    norm_coords : list of tuple or tuple
        Calibration points as (x, y) tuples in normalized coordinates [-1, 1].
    target_units : str, optional
        Target unit system. If None, uses window's current units.
        Options: 'height', 'norm', 'pix', 'cm', 'deg', 'degFlat', 'degFlatPos'
        
    Returns
    -------
    list of tuple or tuple
        Points converted to target units.
    """
    # --- Input Format Detection ---
    single_point = isinstance(norm_coords[0], (int, float))
    if single_point:
        norm_coords = [norm_coords]
    
    # --- Unit System Detection ---
    current_units = target_units if target_units is not None else win.units
    
    # --- Vectorized Conversion ---
    if current_units == "norm":
        converted = norm_coords
        
    elif current_units == "height":
        aspect = win.size[0] / win.size[1]
        converted = [(x * aspect / 2.0, y * 0.5) for x, y in norm_coords]
        
    elif current_units == "pix":
        half_w = win.size[0] / 2.0
        half_h = win.size[1] / 2.0
        converted = [(x * half_w, y * half_h) for x, y in norm_coords]
        
    elif current_units in ["cm", "deg", "degFlat", "degFlatPos"]:
        # Convert to pixels first
        half_w = win.size[0] / 2.0
        half_h = win.size[1] / 2.0
        pix_coords = [(x * half_w, y * half_h) for x, y in norm_coords]
        
        if current_units == "cm":
            converted = [(pix2cm(x, win.monitor), pix2cm(y, win.monitor)) 
                        for x, y in pix_coords]
        elif current_units == "deg":
            converted = [(pix2deg(x, win.monitor), pix2deg(y, win.monitor)) 
                        for x, y in pix_coords]
        else:  # degFlat, degFlatPos
            converted = [(pix2deg(x, win.monitor, correctFlat=True), 
                         pix2deg(y, win.monitor, correctFlat=True)) 
                        for x, y in pix_coords]
    else:
        # Fallback: return as-is
        converted = norm_coords
    
    # --- Return Original Format ---
    return converted[0] if single_point else converted