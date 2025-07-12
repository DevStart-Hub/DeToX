# calibration_config.py
# =====================
# Centralized configuration for all default settings/dictionaries
# for calibration sessions (Tobii and simulation).
# 
# Each dictionary is explained below.
# -----
# EXAMPLE: To change a value from your main experiment script:
# import calibration_config
# calibration_config.DEFAULT_HIGHLIGHT_SIZE["pix"] = 100.0  # Make highlight bigger in pixel mode

# ---------------------------------------------------------
# 1. Size of highlight circles (yellow outline for selected calibration points)
#    Controls how large the highlight appears around a point during the results/retry phase.
#    -----
#    To override from your main script:
#    import calibration_config
#    calibration_config.DEFAULT_HIGHLIGHT_SIZE["pix"] = 60.0
DEFAULT_HIGHLIGHT_SIZE = {
    "norm": 0.08,
    "height": 0.04,
    "pix": 40.0,
    "degFlatPos": 1.0,
    "deg": 1.0,
    "degFlat": 1.0,
    "cm": 1.0,
}

# ---------------------------------------------------------
# 2. Size of small marker circles (shows points already collected)
#    -----
#    To override:
#    calibration_config.DEFAULT_MARKER_SIZE["height"] = 0.02
DEFAULT_MARKER_SIZE = {
    "norm": 0.02,
    "height": 0.01,
    "pix": 10.0,
    "degFlatPos": 0.25,
    "deg": 0.25,
    "degFlat": 0.25,
    "cm": 0.25,
}

# ---------------------------------------------------------
# 3. Width of outline lines (highlights, markers)
#    -----
#    To override:
#    calibration_config.DEFAULT_LINE_WIDTH["pix"] = 12
DEFAULT_LINE_WIDTH = {
    "norm": 4,
    "height": 3,
    "pix": 6,
    "degFlatPos": 2,
    "deg": 2,
    "degFlat": 2,
    "cm": 2,
}

# ---------------------------------------------------------
# 4. Thickness of the red border during calibration mode (visual feedback)
#    -----
#    To override:
#    calibration_config.DEFAULT_BORDER_THICKNESS["norm"] = 0.02
DEFAULT_BORDER_THICKNESS = {
    "norm": 0.01,
    "height": 0.005,
    "pix": 3.0,
    "degFlatPos": 0.1,
    "deg": 0.1,
    "degFlat": 0.1,
    "cm": 0.1,
}

# ---------------------------------------------------------
# 5. Base text height (for all on-screen text, by units)
#    -----
#    To override:
#    calibration_config.DEFAULT_TEXT_HEIGHT["deg"] = 1.0
DEFAULT_TEXT_HEIGHT = {
    "norm": 0.05,
    "height": 0.025,
    "pix": 20.0,
    "degFlatPos": 0.5,
    "deg": 0.5,
    "degFlat": 0.5,
    "cm": 0.5,
}

# ---------------------------------------------------------
# 6. Width of lines in calibration result plots (accuracy lines)
#    -----
#    To override:
#    calibration_config.DEFAULT_PLOT_LINE_WIDTH["pix"] = 6
DEFAULT_PLOT_LINE_WIDTH = {
    "norm": 2,
    "height": 2,
    "pix": 3,
    "degFlatPos": 1,
    "deg": 1,
    "degFlat": 1,
    "cm": 1,
}

# ---------------------------------------------------------
# 7. Colors for each type of line/dot in results (RGBA: Red, Green, Blue, Alpha)
#    -----
#    To override mouse line color to blue from main script:
#    calibration_config.CALIBRATION_LINE_COLORS["mouse"] = (0, 128, 255, 255)
CALIBRATION_LINE_COLORS = {
    "left_eye":      (0, 255, 0, 255),    # Green (Tobii left eye)
    "right_eye":     (255, 0, 0, 255),    # Red   (Tobii right eye)
    "mouse":         (255, 128, 0, 255),  # Orange (simulated mouse sample)
    "target_outline": (0, 0, 0, 255),     # Black outline for calibration targets
}

# ---------------------------------------------------------
# 8. Font settings for all text
#    -----
#    To override instruction font:
#    calibration_config.CALIBRATION_FONTS["instruction_font"] = "Consolas"
CALIBRATION_FONTS = {
    "instruction_font": "Courier New",
    "message_font": "Arial",
}

# ---------------------------------------------------------
# 9. Font size multipliers (relative to base text height)
#    -----
#    To override instruction text to be even bigger:
#    calibration_config.FONT_SIZE_MULTIPLIERS["instruction_text"] = 2.0
FONT_SIZE_MULTIPLIERS = {
    "instruction_text": 1.5,
    "message_text": 1.3,
    "title_text": 1.4,
}

# ---------------------------------------------------------
# 10. Key mapping: which key selects which calibration point
#     -----
#     To swap which number triggers which point:
#     calibration_config.NUMKEY_DICT["1"] = 3  # Now '1' key selects point index 3
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

# ---------------------------------------------------------
# 11. Animation settings for calibration stimuli
#     -----
#     To make the animation much faster:
#     calibration_config.ANIMATION_SETTINGS['animation_speed'] = 3.0
ANIMATION_SETTINGS = {
    'animation_speed': 1.0,
    'target_min': 0.2,
    'focus_time': 0.5,

}



# ===== END OF CONFIG =====
# Edit values here, or override them in your main experiment script
# to control the style and behavior of all calibration routines!
