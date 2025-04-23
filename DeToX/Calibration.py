import tobii_research as tr
from psychopy import core, event, visual
from PIL import Image, ImageDraw
from datetime import datetime
import numpy as np

from .Base import InfantStimuli
from .coord_utils import get_tobii_pos


class CalibrationSession:
    """
    Manages an infant-friendly calibration procedure.

    Splits the process into steps:
      1. Verify points
      2. Prepare stimuli
      3. Collect data interactively
      4. Compute and show results
      5. Let user retry points
      6. Discard and repeat if needed
      7. Save calibration data
    """

    # Animation defaults: speed multiplier and min zoom
    _animation_settings = {
        'animation_speed': 1.0,
        'target_min': 0.2,
    }

    # Key mappings for number keys -> point indices
    _numkey_dict = {
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


    def __init__(
        self,
        win,
        calibration_api,
        infant_stims,
        shuffle=True,
        audio=None,
        focus_time=0.5,
        anim_type='zoom'
    ):
        """
        Initialize session parameters.

        win            : PsychoPy Window for display.
        calibration_api: Tobii calibration interface.
        infant_stims   : list of image paths for attention.
        shuffle        : randomize stimuli order.
        audio          : optional sound on selection.
        focus_time     : delay before data capture.
        anim_type      : 'zoom' or 'trill' animation style.
        """
        self.win = win
        self.calibration = calibration_api
        self.infant_stims = infant_stims
        self.shuffle = shuffle
        self.audio = audio
        self.focus_time = focus_time
        self.anim_type = anim_type


    def run(self, calibration_points, save_calib=False):
        """
        Execute full calibration: collect, compute, retry, save.

        calibration_points : list of (x, y) positions.
        save_calib         : whether to write binary `.dat` file.
        returns True on success.
        """
        # Step 1: verify allowed number of points
        self._check_points(calibration_points)
        # Step 2: set up stimuli
        self._prepare_session(calibration_points)

        # Enter Tobii calibration mode
        self.calibration.enter_calibration_mode()

        # Repeat until no retries
        while True:
            # Step 3: interactive data collection
            retries = self._collection_phase(calibration_points)

            # Step 4: compute and show
            self.calibration_result = self.calibration.compute_and_apply()
            result_img = self._show_calibration_result()

            # Step 5: select retry points
            retries = self._selection_phase(calibration_points, result_img)
            if not retries:
                break

            # Step 6: discard and loop
            self._discard_phase(calibration_points, retries)

        # Exit calibration mode
        self.calibration.leave_calibration_mode()

        # Step 7: save if requested
        success = (self.calibration_result.status == tr.CALIBRATION_STATUS_SUCCESS)
        if success and save_calib:
            data = self.calibration.retrieve_calibration_data()
            fname = f"{datetime.now():%Y-%m-%d_%H-%M-%S}_calibration.dat"
            with open(fname, 'wb') as f:
                f.write(data)
        return success


    def _check_points(self, calibration_points):
        """
        Ensure there are between 2 and 9 calibration points.
        """
        if not (2 <= len(calibration_points) <= 9):
            raise ValueError("Calibration points must be between 2 and 9")


    def _prepare_session(self, calibration_points):
        """
        Create and shuffle stimuli; mark all points uncollected.
        """
        self.targets = InfantStimuli(
            self.win, self.infant_stims, shuffle=self.shuffle
        )
        self.remaining = list(range(len(calibration_points)))


    def _collection_phase(self, calibration_points):
        """
        Let user press number keys to select a point.
        Press Space to collect data after focus_time.
        Return uncollected point indices.
        """
        clock = core.Clock()
        cp_num = len(calibration_points)
        remaining = list(range(cp_num))
        point_idx = -1

        while True:
            # handle key events
            for key in event.getKeys():
                if key in self._numkey_dict:
                    point_idx = self._numkey_dict[key]
                    if self.audio:
                        self.audio.play()
                elif key == 'space' and point_idx in remaining:
                    core.wait(self.focus_time)  # allow gaze to settle
                    x, y = get_tobii_pos(self.win, calibration_points[point_idx])
                    self.calibration.collect_data(x, y)
                    if self.audio:
                        self.audio.pause()
                    remaining.remove(point_idx)
                    point_idx = -1
                elif key == 'return':
                    return remaining

            # animate selected stimulus
            if 0 <= point_idx < cp_num:
                stim = self.targets.get_stim(point_idx)
                stim.setPos(calibration_points[point_idx])
                self._animate(
                    stim, point_idx, clock,
                    self.anim_type, rotation_range=15
                )

            self.win.flip()  # update screen


    def _selection_phase(self, calibration_points, result_img):
        """
        Display result image; toggle retry with number keys.
        Confirm with Space; abort (retry all) with Esc.
        """
        cp_num = len(calibration_points)
        retries = set()
        instructions = visual.TextStim(
            self.win,
            text="Press numbers to toggle retry; Space accept; Esc abort",
            pos=(0, -self.win.size[1]/4), color='white', height=20
        )

        while True:
            result_img.draw()
            instructions.draw()
            for key in event.getKeys():
                if key in self._numkey_dict:
                    idx = self._numkey_dict[key]
                    if 0 <= idx < cp_num:
                        if idx in retries:
                            retries.remove(idx)
                        else:
                            retries.add(idx)
                elif key == 'space':
                    return list(retries)
                elif key == 'escape':
                    return list(range(cp_num))

            # highlight retry points
            for rp in retries:
                visual.Circle(
                    self.win, radius=10,
                    pos=calibration_points[rp],
                    lineColor='yellow'
                ).draw()
            self.win.flip()


    def _discard_phase(self, calibration_points, retries):
        """
        Remove collected data for each retry point.
        """
        for idx in retries:
            x, y = get_tobii_pos(self.win, calibration_points[idx])
            self.calibration.discard_data(x, y)


    def _show_calibration_result(self):
        """
        Draws points and gaze-sample lines on an image.
        Returns a PsychoPy ImageStim of the result.
        """
        img = Image.new("RGBA", tuple(self.win.size))
        img_draw = ImageDraw.Draw(img)
        result_img = visual.SimpleImageStim(self.win, img, autoLog=False)

        if self.calibration_result.status != tr.CALIBRATION_STATUS_FAILURE:
            for point in self.calibration_result.calibration_points:
                p = point.position_on_display_area
                # marker at calibration target
                img_draw.ellipse(
                    ((p[0]*self.win.size[0]-3, p[1]*self.win.size[1]-3),
                     (p[0]*self.win.size[0]+3, p[1]*self.win.size[1]+3)),
                    outline=(0,0,0,255)
                )
                # draw each valid eye sample
                for sample in point.calibration_samples:
                    lp = sample.left_eye.position_on_display_area
                    rp = sample.right_eye.position_on_display_area
                    if sample.left_eye.validity == tr.VALIDITY_VALID_AND_USED:
                        img_draw.line(
                            ((p[0]*self.win.size[0], p[1]*self.win.size[1]),
                             (lp[0]*self.win.size[0], lp[1]*self.win.size[1])),
                            fill=(0,255,0,255)
                        )
                    if sample.right_eye.validity == tr.VALIDITY_VALID_AND_USED:
                        img_draw.line(
                            ((p[0]*self.win.size[0], p[1]*self.win.size[1]),
                             (rp[0]*self.win.size[0], rp[1]*self.win.size[1])),
                            fill=(255,0,0,255)
                        )

        result_img.setImage(img)
        return result_img


    def _animate(self, stim, point_idx, clock, anim_type='zoom', rotation_range=15):
        """
        Apply zoom or trill animation to a stimulus.
        """
        elapsed = clock.getTime() * self._animation_settings['animation_speed']
        if anim_type == 'zoom':
            orig = self.targets.get_stim_original_size(point_idx)
            scale = np.sin(elapsed)**2 + self._animation_settings['target_min']
            stim.setSize([scale * s for s in orig])
        elif anim_type == 'trill':
            angle = np.sin(elapsed) * rotation_range
            stim.setOri(angle)
        stim.draw()
