import config
import requests


class KanoonClient:
    def __init__(self, session=None):
        if not config.API_TOKEN:
            raise RuntimeError(
                "Missing INDIAN_KANOON_API_TOKEN. Set it in the environment before searching Indian Kanoon."
            )
        self.base_url = config.BASE_URL
        self.headers = {
            "Authorization": f"Token {config.API_TOKEN}",
            "Accept": "application/json"
        }
        self.timeout = config.REQUEST_TIMEOUT
        self.search_max_pages = config.SEARCH_MAX_PAGES
        self.session = session or requests.Session()

    @staticmethod
    def _response_detail(resp):
        try:
            payload = resp.json()
            if isinstance(payload, dict):
                return payload.get("errmsg") or payload.get("detail") or str(payload)
        except ValueError:
            pass

        return (getattr(resp, "text", "") or getattr(resp, "reason", "") or "unknown error").strip()

    @staticmethod
    def _coerce_total(value, fallback):
        try:
            return int(value)
        except (TypeError, ValueError):
            return fallback

    @staticmethod
    def _normalize_search_doc(doc):
        if not isinstance(doc, dict):
            return None

        tid = doc.get("tid") or doc.get("docid") or doc.get("id")
        if tid is None:
            return None

        return {
            **doc,
            "tid": tid,
            "title": doc.get("title") or doc.get("doc_title") or f"Document {tid}",
            "headline": doc.get("headline") or doc.get("fragment") or "",
        }

    def _request_json(self, path, params=None):
        resp = self.session.post(
            f"{self.base_url}{path}",
            headers=self.headers,
            params=params,
            timeout=self.timeout,
        )

        try:
            resp.raise_for_status()
        except requests.HTTPError as exc:
            detail = self._response_detail(resp)
            raise RuntimeError(
                f"Indian Kanoon API request failed ({resp.status_code}): {detail}"
            ) from exc

        try:
            data = resp.json()
        except ValueError as exc:
            raise RuntimeError("Indian Kanoon API returned invalid JSON") from exc

        if isinstance(data, dict) and data.get("errmsg"):
            raise RuntimeError(f"Indian Kanoon API error: {data['errmsg']}")

        return data

    def search_documents(self, query, doc_type='all', pagenum=0, maxpages=None):
        """
        Search for documents.
        doc_type: filter by type - 'judgments', 'acts', 'rules', 'all' (default is 'all')
        """
        clean_query = " ".join((query or "").split())
        if not clean_query:
            return {'docs': [], 'total': 0, 'found': 0}

        maxpages = self.search_max_pages if maxpages is None else maxpages
        data = self._request_json(
            "/search/",
            params={"formInput": clean_query, "pagenum": pagenum, "maxpages": maxpages},
        )
        if not isinstance(data, dict):
            raise RuntimeError("Indian Kanoon API returned an unexpected search response")

        raw_docs = data.get('docs') or []
        if not isinstance(raw_docs, list):
            raise RuntimeError("Indian Kanoon API returned an unexpected docs payload")

        docs = [doc for doc in (self._normalize_search_doc(doc) for doc in raw_docs) if doc]

        # Filter based on doc_type.
        if doc_type == 'judgments':
            docs = [doc for doc in docs if str(doc.get('doctype')) == "1000"]
        elif doc_type in {'acts', 'rules'}:
            docs = [doc for doc in docs if str(doc.get('doctype')) != "1000"]
        elif doc_type != 'all':
            raise ValueError(f"Unsupported document type: {doc_type}")

        return {
            'docs': docs,
            'total': self._coerce_total(data.get('found'), len(docs)),
            'found': data.get('found'),
            'pagenum': pagenum,
        }

    def get_document(self, doc_id):
        data = self._request_json(f"/doc/{int(doc_id)}/")
        if not isinstance(data, dict):
            raise RuntimeError("Indian Kanoon API returned an unexpected document response")
        return data
