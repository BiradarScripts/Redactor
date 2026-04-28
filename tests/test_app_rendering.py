import unittest

from app import highlighted_masked_text


class RenderingTests(unittest.TestCase):
    def test_highlighted_masked_text_escapes_untrusted_html(self):
        rendered = highlighted_masked_text("<script>alert(1)</script> [PHONE]")

        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", rendered)
        self.assertIn("<mark>[PHONE]</mark>", rendered)


if __name__ == "__main__":
    unittest.main()
