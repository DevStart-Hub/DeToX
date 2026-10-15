"""Manual recording example; requires a connected Tobii tracker and PsychoPy."""

from psychopy import visual, core
from DeToX import ETracker

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

# # Create controller in simulation mode
controller = ETracker(win, simulate=False)

## Start recording
controller.start_recording('TEST4.h5',  raw_format=True) # save to hdf5 ( set to csv for easier debug)

win.flip()
core.wait(4)

controller.record_event('Event1')
controller.save_data() # Right after the event to test the saving function

core.wait(60)

controller.record_event('Event2')
controller.save_data() # Right after the event to test the saving function

core.wait(1)

controller.record_event('Event3')
controller.save_data() # Right after the event to test the saving function

core.wait(1)

# Clean up
controller.stop_recording() # this closes and saves 
win.close()


