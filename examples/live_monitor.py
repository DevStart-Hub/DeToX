"""Manual test of ETracker's integrated monitor, without recording any data.

Run with the full runtime environment:
    python examples/live_monitor.py --screen 1 --size 500 350

Requires a connected tracker and PsychoPy. The experiment window is fullscreen
on screen 0; choose the monitor's screen explicitly if needed. Escape or 30
seconds ends the test. No calibration or tracker settings are changed.
"""

import argparse
import math


def main():
    """Create the controller, enable independent feedback, then stop cleanly."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--screen', type=int, default=None, help='Display for the feedback window')
    parser.add_argument('--size', type=int, nargs=2, default=(500, 350), metavar=('WIDTH', 'HEIGHT'))
    parser.add_argument('--duration', type=float, default=30)
    parser.add_argument('--show-images', action='store_true', help='Show optional camera feedback')
    parser.add_argument('--reverse-eye', action='store_true', help='Swap camera image panels only')
    args = parser.parse_args()
    if not math.isfinite(args.duration) or args.duration <= 0:
        parser.error('duration must be finite and greater than zero')
    if any(value <= 0 for value in args.size):
        parser.error('size dimensions must be positive')

    from psychopy import core, event, visual
    from DeToX import ETracker

    win = visual.Window(fullscr=True, screen=0, color='grey')
    controller = None
    try:
        controller = ETracker(win)
        controller.enable_live_monitor(screen=args.screen, size=args.size, update_rate=20,
                                       show_images=args.show_images, reverse_eye=args.reverse_eye)
        print('Feedback is enabled without starting a recording. Press Escape to finish.')
        clock = core.Clock()
        while clock.getTime() < args.duration:
            if 'escape' in event.getKeys():
                break
            win.flip()
    finally:
        if controller is not None:
            controller.stop_live_monitor()
        win.close()


if __name__ == '__main__':
    main()
