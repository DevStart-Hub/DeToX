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
