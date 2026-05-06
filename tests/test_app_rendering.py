import unittest

from fastapi.testclient import TestClient

import app as app_module
from app import document_blocks, highlighted_masked_blocks, highlighted_masked_text


class RenderingTests(unittest.TestCase):
    def test_highlighted_masked_text_escapes_untrusted_html(self):
        rendered = highlighted_masked_text("<script>alert(1)</script> [PHONE]")

        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", rendered)
        self.assertIn("<mark>[PHONE]</mark>", rendered)

    def test_document_blocks_split_html_judgment_chunks(self):
        blocks = document_blocks(
            '<blockquote id="a">First block</blockquote>\n'
            '<blockquote id="b">Second [PHONE] block</blockquote>'
        )

        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0], '<blockquote id="a">First block</blockquote>')
        self.assertIn("<mark>[PHONE]</mark>", highlighted_masked_blocks(blocks[1])[0])

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

    def test_process_route_renders_synced_compare_blocks(self):
        class FakeKanoonClient:
            api_enabled = True

            def get_document(self, doc_id):
                return {
                    "title": "Example Judgment",
                    "doc": (
                        '<blockquote id="a">First block</blockquote>\n'
                        '<blockquote id="b">Second block</blockquote>'
                    ),
                }

        class FakeMasker:
            legal_model_name = "test"

            def mask_victims_and_family(self, text):
                return text.replace("Second", "[PROTECTED_PERSON_1]"), {}

        original_client = app_module.client
        original_masker = app_module.masker
        app_module.client = FakeKanoonClient()
        app_module.masker = FakeMasker()

        try:
            response = TestClient(app_module.app).get("/process/321")
        finally:
            app_module.client = original_client
            app_module.masker = original_masker

        self.assertEqual(response.status_code, 200)
        self.assertIn('data-block-index="0"', response.text)
        self.assertIn('data-block-index="1"', response.text)
        self.assertIn("function syncScroll", response.text)
        self.assertIn("<mark>[PROTECTED_PERSON_1]</mark>", response.text)


if __name__ == "__main__":
    unittest.main()
