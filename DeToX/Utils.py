# Third party imports
from psychopy import visual


def NicePrint(body: str, title: str = ""):
    lines = body.splitlines() or [""]
    content_w = max(map(len, lines))
    title_s = f" {title} " if title else ""
    panel_w = max(content_w, len(title_s)) + 2
    tl, tr, bl, br, h, v = "┌", "┐", "└", "┘", "─", "│"
    if title:
        left = (panel_w - len(title_s)) // 2
        right = panel_w - len(title_s) - left
        top = f"{tl}{h*left}{title_s}{h*right}{tr}"
    else:
        top = f"{tl}{h*panel_w}{tr}"
    print(
        top,
        *(
            f"{v}{line}{' '*(panel_w-len(line))}{v}"
            for line in lines
        ),
        f"{bl}{h*panel_w}{br}",
        sep="\n"
    )



class InfantStimuli:
    """
    Stimuli for infant-friendly calibration.

    This class provides a set of animated stimuli for use in infant-friendly
    calibration procedures. It takes a list of image files and optional
    keyword arguments for the ImageStim constructor. It can be used to
    create a sequence of animated stimuli that can be used to calibrate the
    eye tracker.
    """

    def __init__(self, win, infant_stims, shuffle=True, *kwargs):
        """
        Initialize the InfantStimuli class.

        Parameters
        ----------
        win : psychopy.visual.Window
            The PsychoPy window to render the stimuli in.
        infant_stims : list of str
            List of paths to the image files to use for the stimuli.
        shuffle : bool, optional
            Whether to shuffle the order of the stimuli. Default is True.
        *kwargs : dict
            Additional keyword arguments to be passed to the ImageStim constructor.
        """
        self.win = win
        self.stims = dict((i, visual.ImageStim(self.win, image=stim, *kwargs))
                          for i, stim in enumerate(infant_stims))
        self.stim_size = dict((i, image_stim.size) for i, image_stim in self.stims.items())
        self.present_order = [*self.stims]
        if shuffle:
            np.random.shuffle(self.present_order)

    def get_stim(self, idx):
        """
        Get the stimulus by presentation order.

        Parameters
        ----------
        idx : int
            The index of the stimulus in the presentation order.

        Returns
        -------
        psychopy.visual.ImageStim
            The stimulus corresponding to the given index.
        """
        # Calculate the index using modulo to ensure it wraps around
        stim_index = self.present_order[idx % len(self.present_order)]
        
        # Retrieve and return the stimulus by its calculated index
        return self.stims[stim_index]

    def get_stim_original_size(self, idx):
        """
        Get the original size of the stimulus by presentation order.

        Parameters
        ----------
        idx : int
            The index of the stimulus in the presentation order.

        Returns
        -------
        tuple
            The original size of the stimulus as (width, height).
        """
        # Calculate the index using modulo to ensure it wraps around
        stim_index = self.present_order[idx % len(self.present_order)]
        
        # Return the original size of the stimulus
        return self.stim_size[stim_index]