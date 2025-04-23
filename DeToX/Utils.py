# Third party imports
from psychopy import visual


def NicePrint(body: str, title: str = "") -> None:
    """Print a message in a box with an optional title.

    :param body: The string to print inside the box.
    :param title: An optional title to print on the top border of the box.
    """

    # Split the body string into lines
    lines = body.splitlines() or [""]
    # Calculate the maximum width of the lines
    content_w = max(map(len, lines))

    # Calculate the panel width
    panel_w = max(content_w, len(f" {title} " if title else "")) + 2

    # Unicode characters for the corners and sides of the box
    tl, tr, bl, br, h, v = "┌", "┐", "└", "┘", "─", "│"

    # Construct the top border of the box
    if title:
        # Calculate the left and right margins for the title
        left = (panel_w - len(f" {title} ")) // 2
        right = panel_w - len(f" {title} ") - left
        # Construct the top border
        top = f"{tl}{h*left}{title} {h*right}{tr}"
    else:
        # Construct the top border if there is no title
        top = f"{tl}{h*panel_w}{tr}"

    # Print the box
    print(
        top,
        *(
            # Construct each line of the box
            f"{v}{line}{' '*(panel_w-len(line))}{v}"
            for line in lines
        ),
        # Construct the bottom border
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