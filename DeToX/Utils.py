import textwrap



def NicePrint(
    body: str,
    title: str = "",
    verbose: bool = True,
    *,
    max_width: int | None = None,
    ascii_only: bool = False,
) -> str:
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
    max_width : int or None, optional
        Maximum total box width, including borders and padding. Long body lines
        are wrapped, preserving explicit line breaks and leading indentation.
        None (default) disables wrapping. Widths count characters, not Unicode
        display cells. Must be at least 6 and large enough for the title and
        indentation; otherwise ValueError is raised. Titles are never truncated.
    ascii_only : bool, optional
        Use +, -, and | instead of Unicode borders. Default is False. This only
        changes borders, not Unicode characters in the body or title.
        
    Returns
    -------
    str
        The formatted text with box characters, ready for display in console or
        use with PsychoPy TextStim objects. Includes all box-drawing characters
        and proper spacing.
    """
    lines = body.expandtabs(4).splitlines() or [""]
    title = " ".join(title.expandtabs(4).splitlines()).strip()
    title_space = f" {title} " if title else ""

    if max_width is not None:
        if isinstance(max_width, bool) or not isinstance(max_width, int):
            raise TypeError("max_width must be an integer or None")
        if max_width < 6:
            raise ValueError("max_width must be at least 6")
        if title_space and len(title_space) + 4 > max_width:
            raise ValueError("max_width is too small for the title")

        content_width = max_width - 4
        wrapped_lines = []
        for line in lines:
            if len(line) <= content_width:
                wrapped_lines.append(line)
                continue
            if not line.strip():
                wrapped_lines.append("")
                continue
            indent = line[:len(line) - len(line.lstrip(" "))]
            if len(indent) >= content_width:
                raise ValueError("max_width is too small for the body indentation")
            wrapped_lines.extend(textwrap.wrap(
                line,
                width=content_width,
                subsequent_indent=indent,
                replace_whitespace=False,
                break_on_hyphens=False,
            ))
        lines = wrapped_lines

    panel_w = max(max(map(len, lines)), len(title_space)) + 2
    tl, tr, bl, br, h, v = (
        ("+", "+", "+", "+", "-", "|") if ascii_only
        else ("┌", "┐", "└", "┘", "─", "│")
    )

    if title:
        left = (panel_w - len(title_space)) // 2
        right = panel_w - len(title_space) - left
        top = f"{tl}{h * left}{title_space}{h * right}{tr}"
    else:
        top = f"{tl}{h * panel_w}{tr}"

    middle_lines = [
        f"{v} {line.ljust(panel_w - 2)} {v}"
        for line in lines
    ]
    bottom = f"{bl}{h * panel_w}{br}"
    formatted_text = "\n".join([top] + middle_lines + [bottom])

    if verbose:
        print(formatted_text)
    return formatted_text
