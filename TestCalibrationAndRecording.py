
"""
ETracker Testing Script
===============================

This script tests the ETracker eyetracking system in simulation mode.
It validates:
- Initialization and connection
- Calibration procedures (both successful and retry scenarios)
- Data recording with events
- File saving (HDF5 format)

"""

#%% Libraries and Setup

from psychopy import visual, core
from DeToX import ETracker

#%% Preparation 

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

print(f"✓ Window created: {win.size[0]}x{win.size[1]} pixels")
print(f"✓ Units: {win.units}")



#%% Positionning and calibration

# # Create controller in simulation mode
controller = ETracker(win, simulate=True)

# Define calibration points - 5-point calibration
cal_points = [
    (-0.4, 0.4),  (0.4, 0.4),      # Top row
    (0.0, 0.0),      # Middle row  
    (-0.4, -0.4), (0.4, -0.4)    # Bottom row
]

# Stimulus images (here are all the same, replace with your actual stimulus files)
# Note: These should be engaging images for participants (animals, toys, etc.)
stims = ['1.png', '2.png', '3.png', '4.png', '5.png']



# Run calibration
# Show participant position
controller.show_status()

success = controller.calibrate(
    calibration_points=cal_points,
    infant_stims=stims,
    shuffle=True,
    anim_type='zoom', # you can also try trill
    save_calib=True,
    num_samples=5  # Collect 5 samples per point (only for simulation to be removed)
)


if success:
    print("✓ Calibration completed successfully!")
else:
    print("✗ Calibration failed or was aborted")


#%% Recording

## Start recording
controller.start_recording('TEST.h5') # save to hdf5 ( set to csv for easier debug)

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
controller.close() # this closes and saves 
win.close()

