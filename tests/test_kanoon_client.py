import unittest
from unittest.mock import Mock, patch

import requests

from kanoon_client import DEFAULT_USER_AGENT, KanoonClient


class KanoonClientTests(unittest.TestCase):
    def make_client(self, response):
        session = Mock()
        session.post.return_value = response

        with (
            patch("config.API_TOKEN", "test-token"),
            patch("config.BASE_URL", "https://api.indiankanoon.org"),
            patch("config.PUBLIC_BASE_URL", "https://indiankanoon.org"),
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
                "found": "1 - 10 of 166,253",
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
                "User-Agent": DEFAULT_USER_AGENT,
            },
            params={"formInput": "dowry harassment", "pagenum": 0, "maxpages": 1},
            timeout=20,
        )
        self.assertEqual(results["total"], 166253)
        self.assertEqual(results["docs"][0]["tid"], 123)

    def test_search_falls_back_to_public_results_after_api_error(self):
        api_response = Mock()
        api_response.status_code = 403
        api_response.text = "Forbidden"
        api_response.json.side_effect = ValueError("not json")
        api_response.raise_for_status.side_effect = requests.HTTPError("403 Client Error")

        public_response = Mock()
        public_response.status_code = 200
        public_response.text = """
        <span class="results-count"><b>1 - 10 of 42</b> (0.02 seconds)</span>
        <article class="result">
            <h4 class="result_title">
                <a href="/docfragment/456/?formInput=murder">Fallback Result</a>
            </h4>
            <div class="headline">Fallback headline.</div>
        </article>
        """
        public_response.raise_for_status.return_value = None

        session = Mock()
        session.post.return_value = api_response
        session.get.return_value = public_response

        with (
            patch("config.API_TOKEN", "test-token"),
            patch("config.BASE_URL", "https://api.indiankanoon.org"),
            patch("config.PUBLIC_BASE_URL", "https://indiankanoon.org"),
            patch("config.REQUEST_TIMEOUT", 20),
            patch("config.SEARCH_MAX_PAGES", 1),
        ):
            client = KanoonClient(session=session)

        results = client.search_documents("murder")

        self.assertEqual(results["source"], "public")
        self.assertEqual(results["docs"][0]["tid"], 456)
        self.assertEqual(results["total"], 42)

    def test_get_document_uses_document_endpoint(self):
        response = self.ok_response({"title": "Case title", "doc": "Judgment text"})
        client, session = self.make_client(response)

        data = client.get_document(987)

        session.post.assert_called_once_with(
            "https://api.indiankanoon.org/doc/987/",
            headers={
                "Authorization": "Token test-token",
                "Accept": "application/json",
                "User-Agent": DEFAULT_USER_AGENT,
            },
            params=None,
            timeout=20,
        )
        self.assertEqual(data["title"], "Case title")

    def test_missing_api_token_uses_public_search_fallback(self):
        html = """
        <span class="results-count"><b>1 - 10 of 166,253</b> (0.02 seconds)</span>
        <article class="result">
            <h4 class="result_title">
                <a href="/docfragment/241889/?formInput=rape%20case">
                    Vijay Pralhad Warbuvan vs State on 23 January, 2007
                </a>
            </h4>
            <div class="headline">
                A <b>rape</b> case headline from the public search page.
            </div>
            <div class="hlbottom"><a href="/doc/241889/">Full Document</a></div>
        </article>
        """
        response = Mock()
        response.status_code = 200
        response.text = html
        response.raise_for_status.return_value = None
        session = Mock()
        session.get.return_value = response

        with (
            patch("config.API_TOKEN", ""),
            patch("config.BASE_URL", "https://api.indiankanoon.org"),
            patch("config.PUBLIC_BASE_URL", "https://indiankanoon.org"),
            patch("config.REQUEST_TIMEOUT", 20),
            patch("config.SEARCH_MAX_PAGES", 1),
        ):
            client = KanoonClient(session=session)

        results = client.search_documents("rape case")

        session.get.assert_called_once_with(
            "https://indiankanoon.org/search/",
            headers={
                "User-Agent": DEFAULT_USER_AGENT
            },
            params={"formInput": "rape case", "pagenum": 0},
            timeout=20,
        )
        self.assertEqual(results["source"], "public")
        self.assertEqual(results["total"], 166253)
        self.assertEqual(results["docs"][0]["tid"], 241889)
        self.assertIn("Vijay Pralhad", results["docs"][0]["title"])

    def test_missing_api_token_uses_public_document_fallback(self):
        html = """
        <div class="judgments">
            <h3 class="docsource_main">Bombay High Court</h3>
            <h2 class="doc_title">Vijay Pralhad Warbuvan vs State</h2>
            <pre id="pre_1">JUDGMENT
            This is the document text.</pre>
            <p id="p_1">A later paragraph.</p>
        </div>
        """
        response = Mock()
        response.status_code = 200
        response.text = html
        response.raise_for_status.return_value = None
        session = Mock()
        session.get.return_value = response

        with (
            patch("config.API_TOKEN", ""),
            patch("config.BASE_URL", "https://api.indiankanoon.org"),
            patch("config.PUBLIC_BASE_URL", "https://indiankanoon.org"),
            patch("config.REQUEST_TIMEOUT", 20),
            patch("config.SEARCH_MAX_PAGES", 1),
        ):
            client = KanoonClient(session=session)

        data = client.get_document(241889)

        session.get.assert_called_once_with(
            "https://indiankanoon.org/doc/241889/",
            headers={
                "User-Agent": DEFAULT_USER_AGENT
            },
            params=None,
            timeout=20,
        )
        self.assertEqual(data["source"], "public")
        self.assertEqual(data["title"], "Vijay Pralhad Warbuvan vs State")
        self.assertIn("This is the document text.", data["doc"])


if __name__ == "__main__":
    unittest.main()
