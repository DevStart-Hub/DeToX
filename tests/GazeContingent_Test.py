#%% Imports
from psychopy import visual, core, event
from DeToX import ETracker

#%% Create window and stimuli

# Create window - adjust size based on your monitor
win = visual.Window(
    size=[1920, 1080], 
    units='height', 
    fullscr=False,      # Set to True for real experiments
    allowGUI=True,      # Allows window controls for debugging
    color='grey',       # Neutral background
    monitor='testMonitor'  # Use your calibrated monitor name
)
win.setMouseVisible(True)  # Set to False for real experiments

circle = visual.Circle(
    win=win,
    radius=0.1,              # in pixels (or degrees, etc. depending on `units`)
    fillColor='red',         # color inside the circle
    lineColor='black',       # outline color
    lineWidth=2,             # outline thickness
    pos=(0, 0)               # position (x, y)
)

circle.draw()
win.flip()

#%% Initialize eye tracker

## Create controller in simulation mode
controller = ETracker(win, simulate=True)

## Start recording
controller.start_recording('TEST.h5', raw_format=True) # save to hdf5 ( set to csv for easier debug)

## Start gaze contingent
controller.gaze_contingent()


#%% Draw the eye position
core.wait(2)  # wait for 2 seconds to collect some data

while True:
    
    # Get mouse position (x, y)
    positon = controller.get_gaze_position()

    # Draw circle at gaze position
    circle.pos = positon
    circle.draw()
    win.flip()

    # Exit if ESC is pressed
    if 'escape' in event.getKeys():
        break

#%% Cleanup
controller.stop_recording()
win.close()
