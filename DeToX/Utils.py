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
