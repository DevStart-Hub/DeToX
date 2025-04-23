import tobii_research as tr
from psychopy import core, event, visual
from PIL import Image, ImageDraw
from datetime import datetime
import numpy as np

from .Utils import InfantStimuli
from .Coords import get_tobii_pos


class CalibrationSession:
    """
    Infant-friendly calibration session manager.

    Encapsulates the flow of a full calibration procedure:
      1. Validate input points
      2. Initialize stimuli
      3. Collect calibration data at chosen points
      4. Compute and display results
      5. Allow user to select points for re-calibration
      6. Discard data for retried points and loop back
      7. Save calibration if requested
    """

    def __init__(
        self,
        win,
        calibration_api,
        infant_stims,
        shuffle=True,
        audio=None,
        focus_time=0.5,
        anim_type='zoom',
        animation_settings=None,
        numkey_dict=None
    ):
        """
        Parameters
        ----------
        win : psychopy.visual.Window
            PsychoPy window for rendering stimuli and instructions.
        calibration_api : tobii_research.ScreenBasedCalibration
            Tobii's calibration interface (real or stub).
        infant_stims : list of str
            List of image file paths for attention-getting stimuli.
        shuffle : bool, optional
            Whether to randomize stimulus order each run. Default True.
        audio : psychopy.sound.Sound, optional
            Sound to play when user selects a point. Default None.
        focus_time : float, optional
            Delay (seconds) before collecting data at a point. Default 0.5.
        anim_type : str, optional
            Animation style: 'zoom' or 'trill'. Default 'zoom'.
        animation_settings : dict
            {'animation_speed':…, 'target_min':…} passed from controller.
        numkey_dict : dict
            Mapping of key strings to point indices passed from controller.
        """
        self.win = win
        self.calibration = calibration_api
        self.infant_stims = infant_stims
        self.shuffle = shuffle
        self.audio = audio
        self.focus_time = focus_time
        self.anim_type = anim_type

        # pulled in from Base.TobiiController
        self._animation_settings = animation_settings
        self._numkey_dict = numkey_dict

    def run(self, calibration_points, save_calib=False):
        """
        Execute the complete calibration loop.

        Parameters
        ----------
        calibration_points : list of (float, float)
            PsychoPy-normalized (x, y) coordinates for calibration targets.
        save_calib : bool, optional
            If True and calibration succeeds, save binary data to disk.

        Returns
        -------
        bool
            True if calibration succeeded.
        """
        # 1. Verify and prepare
        self.check_points(calibration_points)
        self._prepare_session(calibration_points)

        # Enter calibration mode
        self.calibration.enter_calibration_mode()

        # Retry loop
        while True:
            # 2. Collection phase
            retries = self._collection_phase(calibration_points)

            # 3. Compute & show
            self.calibration_result = self.calibration.compute_and_apply()
            result_img = self._show_calibration_result()

            # 4. Selection phase
            retries = self._selection_phase(calibration_points, result_img)
            if not retries:
                break

            # 5. Discard phase
            self._discard_phase(calibration_points, retries)

        # Leave calibration mode
        self.calibration.leave_calibration_mode()

        # 6. Save calibration data if requested
        success = (self.calibration_result.status == tr.CALIBRATION_STATUS_SUCCESS)
        if success and save_calib:
            data = self.calibration.retrieve_calibration_data()
            fname = f"{datetime.now():%Y-%m-%d_%H-%M-%S}_calibration.dat"
            with open(fname, 'wb') as f:
                f.write(data)
        return success


    def check_points(self, calibration_points):
        """
        Ensure number of calibration points is within allowed range.
        """
        if not (2 <= len(calibration_points) <= 9):
            raise ValueError("Calibration points must be between 2 and 9")


    def _prepare_session(self, calibration_points):
        """
        Initialize stimuli sequence and track points pending collection.
        """
        self.targets = InfantStimuli(
            self.win,
            self.infant_stims,
            shuffle=self.shuffle
        )
        self.remaining = list(range(len(calibration_points)))


    def _collection_phase(self, calibration_points):
        """
        Let user select points by number and collect data on Space.

        Returns
        -------
        list of int
            Indices of points still uncollected.
        """
        clock = core.Clock()
        cp_num = len(calibration_points)
        remaining = list(range(cp_num))
        point_idx = -1

        while True:
            # Handle keys
            for key in event.getKeys():
                if key in self._numkey_dict:
                    # select point; play audio if available
                    point_idx = self._numkey_dict[key]
                    if self.audio:
                        self.audio.play()
                elif key == 'space' and point_idx in remaining:
                    # wait, collect, pause audio
                    core.wait(self.focus_time)
                    x, y = get_tobii_pos(self.win, calibration_points[point_idx])
                    self.calibration.collect_data(x, y)
                    if self.audio:
                        self.audio.pause()
                    remaining.remove(point_idx)
                    point_idx = -1
                elif key == 'return':
                    # finish early
                    return remaining

            # Animate selected stim
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
        Show result image; toggle retry points with number keys.
        Confirm with Space; abort (retry all) with Escape.

        Returns
        -------
        list of int
            Indices chosen for re-calibration.
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

            # highlight retries
            for rp in retries:
                visual.Circle(
                    self.win,
                    radius=10,
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
        Show calibration results with lines indicating accuracy.

        Draws markers at each point and lines from each sample:
        green = left eye, red = right eye.

        Returns
        -------
        SimpleImageStim
            PsychoPy stimulus containing the result image.
        """
        # Create blank RGBA image matching window size
        img = Image.new("RGBA", tuple(self.win.size))
        img_draw = ImageDraw.Draw(img)
        # Wrap in SimpleImageStim to avoid resizing later
        result_img = visual.SimpleImageStim(self.win, img, autoLog=False)

        # Only draw if not a full failure
        if self.calibration_result.status != tr.CALIBRATION_STATUS_FAILURE:
            for point in self.calibration_result.calibration_points:
                p = point.position_on_display_area
                # Draw small circle at target
                img_draw.ellipse(
                    ((p[0]*self.win.size[0]-3, p[1]*self.win.size[1]-3),
                     (p[0]*self.win.size[0]+3, p[1]*self.win.size[1]+3)),
                    outline=(0, 0, 0, 255)
                )
                # Draw lines for each calibration sample
                for sample in point.calibration_samples:
                    lp = sample.left_eye.position_on_display_area
                    rp = sample.right_eye.position_on_display_area
                    # Left eye
                    if sample.left_eye.validity == tr.VALIDITY_VALID_AND_USED:
                        img_draw.line(
                            ((p[0]*self.win.size[0], p[1]*self.win.size[1]),
                             (lp[0]*self.win.size[0], lp[1]*self.win.size[1])),
                            fill=(0,255,0,255)
                        )
                    # Right eye
                    if sample.right_eye.validity == tr.VALIDITY_VALID_AND_USED:
                        img_draw.line(
                            ((p[0]*self.win.size[0], p[1]*self.win.size[1]),
                             (rp[0]*self.win.size[0], rp[1]*self.win.size[1])),
                            fill=(255,0,0,255)
                        )

        # Update stim's image and return
        result_img.setImage(img)
        return result_img


    def _animate(self, stim, point_idx, clock, anim_type='zoom', rotation_range=15):
        """
        Animate a stimulus with zoom or rotation ('trill').
        """
        elapsed_time = clock.getTime() * self._animation_settings['animation_speed']

        if anim_type == 'zoom':
            orig_size = self.targets.get_stim_original_size(point_idx)
            scale_factor = np.sin(elapsed_time)**2 + self._animation_settings['target_min']
            newsize = [scale_factor * s for s in orig_size]
            stim.setSize(newsize)
        elif anim_type == 'trill':
            angle = np.sin(elapsed_time) * rotation_range
            stim.setOri(angle)

        stim.draw()
