from psychopy import visual, sound
from DeToX import TobiiController

# Create window
win = visual.Window(size=[1920, 1080], units='height', fullscr=False, allowGUI=False)

# Create controller in simulation mode
controller = TobiiController(win, simulate=True)

# Define calibration points - 9-point calibration
cal_points = [
    (-0.4, 0.4),  (0.4, 0.4),      # Top row
    (0.0, 0.0),      # Middle row  
    (-0.4, -0.4), (0.4, -0.4)    # Bottom row
]

# Stimulus images - make sure these exist!
stims = ['1.png', '2.png', '3.png', '4.png', '5.png']


# Run calibration
success = controller.calibrate(
    calibration_points=cal_points,
    infant_stims=stims,
    shuffle=True,
    focus_time=0.5,
    anim_type='zoom',
    save_calib=True,
    num_samples=5  # Collect 5 samples per point
)

if success:
    print("\n✓ Calibration completed successfully!")
else:
    print("\n✗ Calibration was cancelled or failed")

# Clean up
controller.close()
win.close()
