#%% Libraries and Setup

from psychopy import visual, core
from DeToX import ETracker

from DeToX import ETSettings as cfg

# Option 1: Moderately Vibrant (40% more saturated)
cfg.colors.left_eye = (100, 200, 255, 120)   # Brighter Sky Blue
cfg.colors.right_eye = (255, 100, 120, 120)  # Coral Pink


#%% Preparation 

# Create window - adjust size based on your monitor
win = visual.Window(
    size=[1920, 1080], 
    units='height', 
    fullscr=True,      # Set to True for real experiments
    allowGUI=True,      # Allows window controls for debugging
    screen=1
)
win.setMouseVisible(True)  # Set to False for real experiments
  

#%% Positionning and calibration

# # Create controller in simulation mode
controller = ETracker(win, simulate=False)

# Run calibration
# Show participant position
controller.show_status()

success = controller.calibrate(
    calibration_points=5,
    shuffle=True,
    anim_type='zoom',
    visualization_style='circles')


win.close()
