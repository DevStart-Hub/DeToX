import os
from psychopy import visual, core
from DeToX import TobiiController

# Create a Window to control the monitor
win = visual.Window(
    size=[920, 920],
    units='norm',
    fullscr=False,
    allowGUI=True)

# Initialize TobiiController to communicate with the eyetracker
controller = TobiiController(win, simulate=True)
controller._simulation_settings['framerate'] = 300  # Set to 60 Hz instead of default 120 Hz

# Start recording
controller.start_recording('demo1-test', event_mode='samplebased')

# Let's record some simulated data for a few seconds
core.wait(2)

# Record an event
controller.record_event('test_event')

# Wait a bit more
core.wait(2)

# Stop recording
controller.stop_recording()

# Clean up
controller.close()
win.close()

import pandas as pd
df= pd.read_csv('demo1-test.csv')
df['TimeDiff']=df.TimeStamp.diff()




