import tobii_research as tr
from psychopy import core, event, visual
from PIL import Image, ImageDraw
from datetime import datetime
import numpy as np

from .Base import InfantStimuli
from .coord_utils import get_tobii_pos


class CalibrationSession:
    """
    Infant-friendly calibration session manager.

    Encapsulates the flow of a full calibration procedure:
      1. Check input points
      2. Initialize stimuli
      3. Collect calibration data at chosen points
      4. Compute and display results
      5. Allow user to select points for re-calibration
      6. Discard data for retried points and loop back
      7. Save calibration if requested
    """

    # Map PsychoPy key names to calibration point indices
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
        Initialize the calibration session.

        Parameters
        ----------
        win : psychopy.visual.Window
            Window for rendering stimuli and text.
        calibration_api : tobii_research.ScreenBasedCalibration
            Tobii calibration API object.
        infant_stims : list of str
            Paths to image stimuli for attention.
        shuffle : bool, optional
            Randomize stimulus order. Default True.
        audio : psychopy.sound.Sound, optional
            Sound to play on point selection. Default None.
        focus_time : float, optional
            Delay before data collection at a point. Default 0.5s.
        anim_type : {'zoom','trill'}, optional
            Animation style for stimuli. Default 'zoom'.
        """
        self.win = win
        self.calibration = calibration_api
        self.infant_stims = infant_stims
        self.shuffle = shuffle
        self.audio = audio
        self.focus_time = focus_time
        self.anim_type = anim_type

        # settings for zoom/trill animation
        self.animation_settings = {
            'animation_speed': 1.0,
            'target_min': 0.2,
        }


    def run(self, calibration_points, save_calib=False):
        """
        Run the full calibration flow.

        Parameters
        ----------
        calibration_points : list of (float, float)
            PsychoPy-normalized target positions.
        save_calib : bool, optional
            Save calibration data on success. Default False.

        Returns
        -------
        bool
            True if calibration succeeded.
        """
        # 1. Check points and initialize
        self._check_points(calibration_points)
        self._prepare_session(calibration_points)

        # enter calibration mode
        self.calibration.enter_calibration_mode()

        # retry loop
        while True:
            # 2. Data collection phase
            retries = self._collection_phase(calibration_points)

            # 3. Compute & display results
            self.result = self.calibration.compute_and_apply()
            result_img = self._show_result_image()

            # 4. Selection phase
            retries = self._selection_phase(calibration_points, result_img)
            if not retries:
                break

            # 5. Discard phase
            self._discard_phase(calibration_points, retries)

        # leave calibration mode
        self.calibration.leave_calibration_mode()

        # save if requested
        success = (self.result.status == tr.CALIBRATION_STATUS_SUCCESS)
        if success and save_calib:
            data = self.calibration.retrieve_calibration_data()
            fname = f"{datetime.now():%Y-%m-%d_%H-%M-%S}_calibration.dat"
            with open(fname, 'wb') as f:
                f.write(data)
        return success


    def _check_points(self, calibration_points):
        """
        Ensure there are between 2 and 9 calibration points.

        Raises
        ------
        ValueError
            If point count is out of allowed range.
        """
        if not (2 <= len(calibration_points) <= 9):
            raise ValueError("Calibration points must be between 2 and 9")


    def _prepare_session(self, calibration_points):
        """
        Initialize stimuli and track points to collect.
        """
        self.targets = InfantStimuli(
            self.win,
            self.infant_stims,
            shuffle=self.shuffle
        )
        self.remaining = list(range(len(calibration_points)))


    def _collection_phase(self, calibration_points):
        """
        Let user select points with number keys and collect data on Space.

        Returns
        -------
        list of int
            Indices of points not yet collected.
        """
        clock = core.Clock()
        cp_num = len(calibration_points)
        remaining = list(range(cp_num))
        point_idx = -1

        while True:
            for key in event.getKeys():
                if key in self._numkey_dict:
                    point_idx = self._numkey_dict[key]
                    if self.audio:
                        self.audio.play()
                elif key == 'space' and point_idx in remaining:
                    core.wait(self.focus_time)
                    x, y = get_tobii_pos(self.win, calibration_points[point_idx])
                    self.calibration.collect_data(x, y)
                    if self.audio:
                        self.audio.pause()
                    remaining.remove(point_idx)
                    point_idx = -1
                elif key == 'return':
                    return remaining

            if 0 <= point_idx < cp_num:
                stim = self.targets.get_stim(point_idx)
                stim.setPos(calibration_points[point_idx])
                self._animate(
                    stim, point_idx, clock,
                    self.anim_type, rotation_range=15
                )

            self.win.flip()


    def _selection_phase(self, calibration_points, result_img):
        """
        Display results and allow toggling retries.

        Returns
        -------
        list of int
            Indices chosen for re-calibration.
        """
        cp_num = len(calibration_points)
        retries = set()
        instructions = visual.TextStim(
            self.win,
            text="Number to toggle retry  Space to accept  Esc to abort",
            pos=(0, -self.win.size[1]/4),
            color='white', height=20
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

            for rp in retries:
                visual.Circle(
                    self.win, radius=10,
                    pos=calibration_points[rp],
                    lineColor='yellow'
                ).draw()
            self.win.flip()


    def _discard_phase(self, calibration_points, retries):
        """
        Discard data for specified retry points.
        """
        for idx in retries:
            x, y = get_tobii_pos(self.win, calibration_points[idx])
            self.calibration.discard_data(x, y)


    def _show_result_image(self):
        """
        Render calibration points and error lines to an ImageStim.

        Returns
        -------
        visual.SimpleImageStim
        """
        win = self.win
        img = Image.new("RGBA", tuple(win.size))
        draw = ImageDraw.Draw(img)

        if self.result.status != tr.CALIBRATION_STATUS_FAILURE:
            for pt in self.result.calibration_points:
                p = pt.position_on_display_area
                x0, y0 = p[0]*win.size[0], p[1]*win.size[1]
                draw.ellipse((x0-3, y0-3, x0+3, y0+3), outline=(0,0,0,255))
                for sample in pt.calibration_samples:
                    lp = sample.left_eye.position_on_display_area
                    rp = sample.right_eye.position_on_display_area
                    if sample.left_eye.validity == tr.VALIDITY_VALID_AND_USED:
                        lx, ly = lp[0]*win.size[0], lp[1]*win.size[1]
                        draw.line((x0, y0, lx, ly), fill=(0,255,0,255))
                    if sample.right_eye.validity == tr.VALIDITY_VALID_AND_USED:
                        rx, ry = rp[0]*win.size[0], rp[1]*win.size[1]
                        draw.line((x0, y0, rx, ry), fill=(255,0,0,255))

        stim = visual.SimpleImageStim(win, img, autoLog=False)
        stim.setImage(img)
        return stim


    def _animate(self, stim, point_idx, clock, anim_type='zoom', rotation_range=15):
        """
        Animate a stimulus with zoom or trill effect.

        Parameters
        ----------
        stim : visual.ImageStim
            Stimulus to animate.
        point_idx : int
            Calibration point index.
        clock : psychopy.core.Clock
            Elapsed time tracker.
        anim_type : {'zoom','trill'}
            Animation style.
        rotation_range : float
            Max rotation in degrees for 'trill'.
        """
        elapsed = clock.getTime() * self.animation_settings['animation_speed']

        if anim_type == 'zoom':
            orig_size = self.targets.get_stim_original_size(point_idx)
            scale = np.sin(elapsed)**2 + self.animation_settings['target_min']
            stim.setSize([scale * s for s in orig_size])
        elif anim_type == 'trill':
            angle = np.sin(elapsed) * rotation_range
            stim.setOri(angle)

        stim.draw()
