import DeToX

import os
from psychopy import visual, core

from DeToX import TobiiController

# create a Window to control the monitor
win = visual.Window(
    size=[920, 920],
    units='norm',
    fullscr=False,
    allowGUI=True)
# initialize TobiiInfantController to communicate with the eyetracker
controller = TobiiController(win,simulate=True)


# Start recording
controller.start_recording('demo1-test.csv')
core.wait(3) # record for 3 seconds

# stop recording
controller.stop_recording()
# close the file
controller.close()

# shut down the experiment
win.close()
core.quit()