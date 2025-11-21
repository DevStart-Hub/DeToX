# Third party imports
import numpy as np
from psychopy import visual



def NicePrint(body: str, title: str = "", verbose=True) -> str:
    """
    Print a message in a box with an optional title AND return the formatted text.
    
    Creates a visually appealing text box using Unicode box-drawing characters that
    displays both in the console and can be used in PsychoPy visual stimuli. This
    function is particularly useful for presenting instructions, status messages,
    and calibration information in a consistent, professional format.
    
    The box automatically adjusts its width to accommodate the longest line of text
    and centers the title if provided. The formatted output uses Unicode characters
    for smooth, connected borders that render well in both terminal and graphical
    environments.
    
    Parameters
    ----------
    body : str
        The string to print inside the box. Can contain multiple lines separated
        by newline characters. Each line will be padded to align within the box.
    title : str, optional
        A title to print on the top border of the box. The title will be centered
        within the top border. If empty string or not provided, the top border
        will be solid. Default empty string.
    verbose : bool, optional
        If True, the formatted box will be printed to the console. Default is True.
        
    Returns
    -------
    str
        The formatted text with box characters, ready for display in console or
        use with PsychoPy TextStim objects. Includes all box-drawing characters
        and proper spacing.
    """
    # --- Text Processing ---
    # Split the body string into individual lines for formatting
    lines = body.splitlines() or [""]
    
    # --- Width Calculation ---
    # Calculate the maximum width needed for content
    content_w = max(map(len, lines))
    
    # --- Panel Sizing ---
    # Calculate the panel width to accommodate both content and title
    title_space = f" {title} " if title else ""
    panel_w = max(content_w, len(title_space)) + 2
    
    # --- Box Character Definition ---
    # Unicode characters for the corners and sides of the box
    # These create smooth, connected borders in terminals that support Unicode
    tl, tr, bl, br, h, v = "┌", "┐", "└", "┘", "─", "│"
    
    # --- Top Border Construction ---
    # Construct the top border of the box with optional centered title
    if title:
        # Calculate the left and right margins for centering the title
        left = (panel_w - len(title_space)) // 2
        right = panel_w - len(title_space) - left
        # Construct the top border with embedded title
        top = f"{tl}{h * left}{title_space}{h * right}{tr}"
    else:
        # Construct solid top border without title
        top = f"{tl}{h * panel_w}{tr}"
    
    # --- Content Line Formatting ---
    # Create the middle lines with content, padding each line to panel width
    middle_lines = [
        f"{v}{line}{' ' * (panel_w - len(line))}{v}"
        for line in lines
    ]
    
    # --- Bottom Border Construction ---
    # Create the bottom border
    bottom = f"{bl}{h * panel_w}{br}"
    
    # --- Final Assembly ---
    # Combine all parts into the complete formatted text
    formatted_text = "\n".join([top] + middle_lines + [bottom])
    
    # --- Console Output ---
    # Print to console for immediate feedback
    if verbose:
        print(formatted_text)
    
    # --- Return Formatted Text ---
    # Return the formatted text for use in PsychoPy visual stimuli
    return formatted_text



class InfantStimuli:
    """
    Stimuli manager for infant-friendly calibration procedures.

    This class provides a sophisticated stimulus presentation system designed
    specifically for infant eye tracking calibration. It manages a collection
    of engaging visual stimuli (typically colorful, animated images) and handles
    their sequential presentation during calibration procedures.
    
    The class maintains the original size information for each stimulus and
    supports randomized presentation order to prevent habituation. It's designed
    to work seamlessly with both Tobii hardware calibration and mouse-based
    simulation modes.
    
    Key features include:
    - Automatic stimulus loading from image files
    - Configurable presentation order (sequential or randomized)
    - Size preservation for animation calculations
    - Modulo indexing for circular stimulus access
    - Integration with PsychoPy's visual system
    
    Attributes
    ----------
    win : psychopy.visual.Window
        The PsychoPy window used for rendering.
    stims : dict
        Dictionary mapping indices to ImageStim objects.
    stim_size : dict
        Dictionary mapping indices to original stimulus sizes.
    present_order : list
        List defining the presentation sequence of stimuli.
    """

    def __init__(self, win, infant_stims, **kwargs): # TODO: check if kwargs is needed
        """
        Initialize the InfantStimuli manager.
        
        Sets up the stimulus collection by loading images and preparing them
        for presentation. Each image is converted to a PsychoPy ImageStim object
        and its original size is preserved for animation purposes.

        Parameters
        ----------
        win : psychopy.visual.Window
            The PsychoPy window to render the stimuli in. This window's properties
            (size, units, etc.) will be used for stimulus presentation.
        infant_stims : list of str
            List of paths to the image files to use for the stimuli. These should
            be engaging images suitable for infant participants (e.g., cartoon
            characters, colorful objects, animated figures).
        """
        # --- Window Reference ---
        # Store reference to the PsychoPy window for rendering
        self.win = win
        
        # --- Stimulus Loading ---
        # Create ImageStim objects for each provided image file
        self.stims = dict((i, visual.ImageStim(self.win, image=stim, units='height', interpolate=True))
                          for i, stim in enumerate(infant_stims))
        
        # --- Size Preservation ---
        # Store original sizes for animation and scaling calculations
        self.stim_size = dict((i, image_stim.size) for i, image_stim in self.stims.items())
        
        # --- Presentation Order Setup ---
        # Create initial presentation order matching stimulus indices
        self.present_order = [*self.stims]
        

    def get_stim(self, idx):
        """
        Get the stimulus by presentation order.
        
        Retrieves a stimulus based on its position in the presentation sequence.
        Uses modulo arithmetic to wrap around when the index exceeds the number
        of available stimuli, enabling circular access patterns.

        Parameters
        ----------
        idx : int
            The index of the stimulus in the presentation order. Can be any
            non-negative integer; values beyond the stimulus count will wrap
            around using modulo operation.

        Returns
        -------
        psychopy.visual.ImageStim
            The stimulus corresponding to the given index in the presentation
            order. The returned stimulus is ready for positioning, animation,
            and drawing operations.
            
        Examples
        --------
        >>> stim_manager = InfantStimuli(win, ['img1.png', 'img2.png'])
        >>> stim = stim_manager.get_stim(0)  # Get first stimulus
        >>> stim = stim_manager.get_stim(5)  # Wraps around if only 2 stimuli
        """
        # --- Index Calculation ---
        # Calculate the index using modulo to ensure it wraps around
        stim_index = self.present_order[idx % len(self.present_order)]
        
        # --- Stimulus Retrieval ---
        # Retrieve and return the stimulus by its calculated index
        return self.stims[stim_index]

    def get_stim_original_size(self, idx):
        """
        Get the original size of the stimulus by presentation order.
        
        Returns the original dimensions of a stimulus as loaded from the image
        file. This is useful for animation calculations where you need to know
        the base size before applying scaling transformations.

        Parameters
        ----------
        idx : int
            The index of the stimulus in the presentation order. Can be any
            non-negative integer; values beyond the stimulus count will wrap
            around using modulo operation.

        Returns
        -------
        tuple
            The original size of the stimulus as (width, height) in the units
            specified during stimulus creation. These values represent the
            stimulus dimensions before any scaling or animation effects.
            
        Notes
        -----
        The original size is preserved at initialization and remains constant
        throughout the session, regardless of any size modifications made to
        the stimulus during animation.
        """
        # --- Index Calculation ---
        # Calculate the index using modulo to ensure it wraps around
        stim_index = self.present_order[idx % len(self.present_order)]
        
        # --- Size Retrieval ---
        # Return the original size of the stimulus
        return self.stim_size[stim_index]