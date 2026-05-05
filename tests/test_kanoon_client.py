import unittest
from unittest.mock import Mock, patch

import requests

from kanoon_client import KanoonClient


class KanoonClientTests(unittest.TestCase):
    def make_client(self, response):
        session = Mock()
        session.post.return_value = response

        with (
            patch("config.API_TOKEN", "test-token"),
            patch("config.BASE_URL", "https://api.indiankanoon.org"),
            patch("config.REQUEST_TIMEOUT", 20),
            patch("config.SEARCH_MAX_PAGES", 1),
        ):
            client = KanoonClient(session=session)

        return client, session

    def ok_response(self, payload):
        response = Mock()
        response.status_code = 200
        response.json.return_value = payload
        response.raise_for_status.return_value = None
        return response

    def test_search_sends_form_input_as_url_params(self):
        response = self.ok_response(
            {
                "found": 1,
                "docs": [
                    {
                        "tid": 123,
                        "title": "Example Case",
                        "headline": "Example search hit",
                        "doctype": 1000,
                    }
                ],
            }
        )
        client, session = self.make_client(response)

        results = client.search_documents("  dowry   harassment  ")

        session.post.assert_called_once_with(
            "https://api.indiankanoon.org/search/",
            headers={
                "Authorization": "Token test-token",
                "Accept": "application/json",
            },
            params={"formInput": "dowry harassment", "pagenum": 0, "maxpages": 1},
            timeout=20,
        )
        self.assertEqual(results["total"], 1)
        self.assertEqual(results["docs"][0]["tid"], 123)

    def test_search_raises_api_errors_instead_of_returning_empty_results(self):
        response = Mock()
        response.status_code = 403
        response.text = "Forbidden"
        response.json.side_effect = ValueError("not json")
        response.raise_for_status.side_effect = requests.HTTPError("403 Client Error")
        client, _ = self.make_client(response)

        with self.assertRaisesRegex(RuntimeError, "403"):
            client.search_documents("murder")

    def test_get_document_uses_document_endpoint(self):
        response = self.ok_response({"title": "Case title", "doc": "Judgment text"})
        client, session = self.make_client(response)

        data = client.get_document(987)

        session.post.assert_called_once_with(
            "https://api.indiankanoon.org/doc/987/",
            headers={
                "Authorization": "Token test-token",
                "Accept": "application/json",
            },
            params=None,
            timeout=20,
        )
        self.assertEqual(data["title"], "Case title")


if __name__ == "__main__":
    unittest.main()
