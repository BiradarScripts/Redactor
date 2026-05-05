import unittest

from fastapi.testclient import TestClient

import app as app_module
from app import highlighted_masked_text


class RenderingTests(unittest.TestCase):
    def test_highlighted_masked_text_escapes_untrusted_html(self):
        rendered = highlighted_masked_text("<script>alert(1)</script> [PHONE]")

        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", rendered)
        self.assertIn("<mark>[PHONE]</mark>", rendered)

    def test_search_route_renders_client_results(self):
        class FakeKanoonClient:
            def __init__(self):
                self.query = None

            def search_documents(self, query):
                self.query = query
                return {
                    "total": 9,
                    "docs": [
                        {
                            "tid": 321,
                            "title": "<b>Example Case</b>",
                            "headline": "Search hit",
                        }
                    ],
                }

        fake_client = FakeKanoonClient()
        original_client = app_module.client
        app_module.client = fake_client

        try:
            response = TestClient(app_module.app).post(
                "/search",
                data={"query": "  dowry   harassment  "},
            )
        finally:
            app_module.client = original_client

        self.assertEqual(response.status_code, 200)
        self.assertEqual(fake_client.query, "dowry harassment")
        self.assertIn('Search results for "dowry harassment"', response.text)
        self.assertIn("Showing 1 of 9 matching documents", response.text)
        self.assertIn("/process/321", response.text)


if __name__ == "__main__":
    unittest.main()
