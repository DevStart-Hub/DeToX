import importlib.util
import io
from contextlib import redirect_stdout
from pathlib import Path
import unittest


# Load the utility without importing the SDK/PsychoPy-dependent package entry point.
spec = importlib.util.spec_from_file_location(
    "detox_utils", Path(__file__).resolve().parents[1] / "DeToX" / "Utils.py"
)
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)
NicePrint = utils.NicePrint


class NicePrintTests(unittest.TestCase):
    def assert_aligned(self, box, max_width=None):
        widths = [len(line) for line in box.splitlines()]
        self.assertEqual(len(set(widths)), 1)
        if max_width is not None:
            self.assertLessEqual(widths[0], max_width)

    def test_balanced_padding(self):
        box = NicePrint("Hello\nOK", verbose=False)
        self.assertEqual(box, "┌───────┐\n│ Hello │\n│ OK    │\n└───────┘")
        self.assert_aligned(box)

    def test_empty_body_and_long_title(self):
        for title in ("", "A title longer than the body"):
            with self.subTest(title=title):
                box = NicePrint("", title, False)
                self.assertEqual(len(box.splitlines()), 3)
                self.assert_aligned(box)
                if title:
                    self.assertIn(f" {title} ", box.splitlines()[0])

    def test_tabs_and_multiline_title(self):
        box = NicePrint("A\tB", "Recording\nInfo", False)
        self.assertNotIn("\t", box)
        self.assertIn("A   B", box)
        self.assertIn("Recording Info", box.splitlines()[0])
        self.assertEqual(len(box.splitlines()), 3)
        self.assert_aligned(box)

    def test_wrap_preserves_paragraphs_and_indentation(self):
        box = NicePrint("  alpha beta gamma delta\n\nDone", "Info", False,
                        max_width=16)
        body_lines = [line[2:-2].rstrip() for line in box.splitlines()[1:-1]]
        self.assertEqual(body_lines, ["  alpha beta", "  gamma", "  delta", "", "Done"])
        self.assert_aligned(box, 16)

    def test_wrap_long_filename_without_losing_characters(self):
        filename = "participant_01_session_02.h5"
        box = NicePrint(filename, verbose=False, max_width=12)
        content = "".join(line[2:-2].rstrip() for line in box.splitlines()[1:-1])
        self.assertEqual(content, filename)
        self.assert_aligned(box, 12)

    def test_wrapping_is_opt_in(self):
        body = "x" * 100
        box = NicePrint(body, verbose=False)
        self.assertIn(body, box)
        self.assertEqual(len(box.splitlines()), 3)

    def test_ascii_borders(self):
        box = NicePrint("Hello", "Info", False, ascii_only=True)
        self.assertTrue(box.isascii())
        self.assertTrue(box.startswith("+"))
        self.assertIn("| Hello", box)
        self.assert_aligned(box)

    def test_verbose_controls_printing_not_return_value(self):
        for verbose in (True, False):
            with self.subTest(verbose=verbose):
                stream = io.StringIO()
                with redirect_stdout(stream):
                    box = NicePrint("Hello", "Info", verbose)
                self.assertEqual(stream.getvalue(), box + "\n" if verbose else "")
                self.assertIn("Hello", box)

    def test_invalid_widths(self):
        for width in (True, 12.5, "20"):
            with self.subTest(width=width), self.assertRaises(TypeError):
                NicePrint("Hello", verbose=False, max_width=width)
        for width in (-1, 0, 5):
            with self.subTest(width=width), self.assertRaises(ValueError):
                NicePrint("Hello", verbose=False, max_width=width)

    def test_title_and_indentation_must_fit(self):
        with self.assertRaisesRegex(ValueError, "title"):
            NicePrint("Hello", "Long title", False, max_width=10)
        with self.assertRaisesRegex(ValueError, "indentation"):
            NicePrint("      Hello", verbose=False, max_width=10)

    def test_smallest_width_and_whitespace_only_line(self):
        box = NicePrint("abc\n       \nx", verbose=False, max_width=6)
        self.assert_aligned(box, 6)
        self.assertEqual([line[2:-2].rstrip() for line in box.splitlines()[1:-1]],
                         ["ab", "c", "", "x"])


if __name__ == "__main__":
    unittest.main()
