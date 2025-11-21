#%% Libraries and Setup

from psychopy import visual, core
from DeToX import ETracker


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

print(f"✓ Window created: {win.size[0]}x{win.size[1]} pixels")
print(f"✓ Units: {win.units}")



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


if success:
    print("✓ Calibration completed successfully!")
else:
    print("✗ Calibration failed or was aborted")


win.close()


#%% Recording

## Start recording
controller.start_recording('TEST.h5', raw_format=True) # save to hdf5 ( set to csv for easier debug)

win.flip()
core.wait(4)

controller.record_event('Event1')
controller.save_data() # Right after the event to test the saving function

core.wait(2)

controller.record_event('Event2')
core.wait(2)
controller.save_data()

core.wait(2)


#%% Closing

# Clean up
controller.stop_recording() # this closes and saves 
win.close()
