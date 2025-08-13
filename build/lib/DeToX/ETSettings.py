# ETSettings.py
# =====================
# Centralized configuration for all default settings and dictionaries
# for calibration sessions (Tobii and simulation).
#
# All sizes are specified in height units (% of screen height) and automatically
# converted to other units as needed. This makes configuration much simpler!
#
# Edit these here, or override in your main script:
#   import calibration_config
#   calibration_config.ANIMATION_SETTINGS['max_zoom_size_height'] = 0.20

# ---------------------------------------------------------
# 1. Key mapping: which key selects which calibration point
#    (Makes it easy to use different keyboards, or change which key triggers each point.)
NUMKEY_DICT = {
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
# Example: calibration_config.NUMKEY_DICT["1"] = 3  # Now '1' selects point index 3

# ---------------------------------------------------------
# 2. Animation settings for calibration stimuli
#    All sizes in height units (% of screen height) - automatically converted to other units
ANIMATION_SETTINGS = {
    # Focus time
    'focus_time': 0.5,                   # Wait time before collecting data in s

    # Zoom
    'zoom_speed': 6.0,                   # Speed of zoom animation
    'max_zoom_size': 0.11,               # 15% of screen height (zoom animation max)
    'min_zoom_size': 0.05,              # 2.5% of screen height (zoom animation min)

    # Trill - Real rapid back-and-forth oscillations
    'trill_size': 0.075,                 # 7.5% of screen height (trill fixed size)
    'trill_rotation_range': 20,          # Maximum rotation angle in degrees for trill
    'trill_cycle_duration': 1.5,         # Total cycle time: 1s trill + 0.5s stop = 1.5s
    'trill_active_duration': 1.1,        # Trill for 1 second, then stop for 0.5 second
    'trill_frequency': 3.0,              # How many back-and-forth oscillations per second
}

# Examples:
# calibration_config.ANIMATION_SETTINGS['max_zoom_size_height'] = 0.20  # Bigger stimuli
# calibration_config.ANIMATION_SETTINGS['trill_speed'] = 6.0            # Slower trill

# ---------------------------------------------------------
# 3. Colors for all visual elements (lines, dots, highlights, etc.)
CALIBRATION_COLORS = {
    "left_eye":      (0, 255, 0, 255),       # Green (Tobii left eye)
    "right_eye":     (255, 0, 0, 255),       # Red   (Tobii right eye)
    "mouse":         (255, 128, 0, 255),     # Orange (simulated mouse sample)
    "target_outline": (24, 24, 24, 255),        # Black outline for calibration targets
    "highlight":     (255, 255, 0, 255),     # Yellow highlight for selected points
}
# Example: calibration_config.CALIBRATION_COLORS["highlight"] = (0,255,255,255)  # Cyan

# ---------------------------------------------------------
# 4. UI element sizes in height units (% of screen height)
#    These are automatically converted to the appropriate units for each window
DEFAULT_HIGHLIGHT_SIZE_HEIGHT = 0.04      # 4% of screen height (retry selection circles)
DEFAULT_LINE_WIDTH_HEIGHT = 0.003         # 0.3% of screen height (line thickness)
DEFAULT_MARKER_SIZE_HEIGHT = 0.02         # 1% of screen height (collection markers)
DEFAULT_BORDER_THICKNESS_HEIGHT = 0.005   # 0.5% of screen height (calibration border)
DEFAULT_PLOT_LINE_WIDTH_HEIGHT = 0.002    # 0.2% of screen height (result plot lines)
DEFAULT_TEXT_HEIGHT_HEIGHT = 0.025        # 2.5% of screen height (base text size)
DEFAULT_TARGET_CIRCLE_SIZE_HEIGHT = 0.012 # 1.2% of screen height (target circles in results)
DEFAULT_TARGET_CIRCLE_WIDTH_HEIGHT = 0.006 #0.8% of screen height (target circle line width)

# Examples:
# calibration_config.DEFAULT_HIGHLIGHT_SIZE_HEIGHT = 0.06    # Bigger highlight circles
# calibration_config.DEFAULT_TEXT_HEIGHT_HEIGHT = 0.035     # Bigger text

# ---------------------------------------------------------
# 5. Font size multipliers (relative to DEFAULT_TEXT_HEIGHT_HEIGHT)
FONT_SIZE_MULTIPLIERS = {
    "instruction_text": 1.5,      # 150% of base text size
    "message_text": 1.3,          # 130% of base text size
    "title_text": 1.4,            # 140% of base text size
}
# Example: calibration_config.FONT_SIZE_MULTIPLIERS["instruction_text"] = 2.0  # Even bigger


# ---------------------------------------------------------
# 6. Calibration session settings
simulation_settings = {
    'framerate': 120,  # Default to Tobii Pro Spectrum rate
}





# ===== SUMMARY =====
# Now you only need to edit these simple height values:
#
# Animation sizes:
#   - ANIMATION_SETTINGS['max_zoom_size_height'] = 0.15    (15% of screen height)
#   - ANIMATION_SETTINGS['min_zoom_size_height'] = 0.025   (2.5% of screen height)
#   - ANIMATION_SETTINGS['trill_size_height'] = 0.075      (7.5% of screen height)
#
# UI element sizes:
#   - DEFAULT_HIGHLIGHT_SIZE_HEIGHT = 0.04                 (4% of screen height)
#   - DEFAULT_TEXT_HEIGHT_HEIGHT = 0.025                   (2.5% of screen height)
#   - etc.
#
# Everything else is automatically converted to the correct units!
# ===== END OF CONFIG =====