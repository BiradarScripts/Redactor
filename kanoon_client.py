import config
import re
import requests
from html.parser import HTMLParser


def _has_class(attrs, class_name):
    classes = dict(attrs).get("class", "")
    return class_name in classes.split()


def _clean_text(value):
    return re.sub(r"\s+", " ", value or "").strip()


def _clean_document_text(value):
    lines = [_clean_text(line) for line in (value or "").splitlines()]
    return "\n".join(line for line in lines if line)


DEFAULT_USER_AGENT = "Redactor/1.0 (+https://github.com/BiradarScripts/Redactor)"


class PublicSearchParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.docs = []
        self.total = 0
        self.current = None
        self.in_title = False
        self.in_title_link = False
        self.in_headline = False
        self.headline_depth = 0
        self.in_results_count = False
        self.count_parts = []

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)

        if tag == "span" and _has_class(attrs, "results-count"):
            self.in_results_count = True

        if tag == "article" and _has_class(attrs, "result"):
            self.current = {"title_parts": [], "headline_parts": [], "tid": None}
            return

        if self.current is None:
            return

        if tag in {"br", "p"} and self.in_headline:
            self.current["headline_parts"].append(" ")

        if tag == "h4" and _has_class(attrs, "result_title"):
            self.in_title = True
            return

        if tag == "a" and self.in_title:
            href = attrs_dict.get("href", "")
            match = re.search(r"/(?:docfragment|doc)/(\d+)/", href)
            if match:
                self.current["tid"] = int(match.group(1))
                self.in_title_link = True
            return

        if tag == "div" and _has_class(attrs, "headline"):
            self.in_headline = True
            self.headline_depth = 1
            return

        if self.in_headline:
            self.headline_depth += 1

    def handle_endtag(self, tag):
        if tag == "span" and self.in_results_count:
            self.in_results_count = False
            count_text = _clean_text("".join(self.count_parts))
            match = re.search(r"\bof\s+([\d,]+)", count_text)
            if match:
                self.total = int(match.group(1).replace(",", ""))
            self.count_parts = []

        if self.current is None:
            return

        if tag == "a" and self.in_title_link:
            self.in_title_link = False

        if tag == "h4" and self.in_title:
            self.in_title = False

        if self.in_headline:
            self.headline_depth -= 1
            if self.headline_depth <= 0:
                self.in_headline = False

        if tag == "article":
            title = _clean_text("".join(self.current["title_parts"]))
            headline = _clean_text("".join(self.current["headline_parts"]))
            tid = self.current["tid"]

            if tid and title:
                self.docs.append({"tid": tid, "title": title, "headline": headline})

            self.current = None
            self.in_title = False
            self.in_title_link = False
            self.in_headline = False
            self.headline_depth = 0

    def handle_data(self, data):
        if self.in_results_count:
            self.count_parts.append(data)

        if self.current is None:
            return

        if self.in_title_link:
            self.current["title_parts"].append(data)
        elif self.in_headline:
            self.current["headline_parts"].append(data)


class PublicDocumentParser(HTMLParser):
    BLOCK_TAGS = {"h1", "h2", "h3", "h4", "p", "pre", "div", "br"}
    VOID_TAGS = {"br", "hr", "img", "input", "link", "meta"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.title_parts = []
        self.doc_parts = []
        self.in_title = False
        self.in_doc = False
        self.doc_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag == "h2" and _has_class(attrs, "doc_title"):
            self.in_title = True

        if tag == "div" and _has_class(attrs, "judgments"):
            self.in_doc = True
            self.doc_depth = 1
            return

        if self.in_doc:
            if tag in self.BLOCK_TAGS:
                self.doc_parts.append("\n")
            if tag not in self.VOID_TAGS:
                self.doc_depth += 1

    def handle_endtag(self, tag):
        if tag == "h2" and self.in_title:
            self.in_title = False

        if self.in_doc:
            if tag in self.BLOCK_TAGS:
                self.doc_parts.append("\n")
            self.doc_depth -= 1
            if self.doc_depth <= 0:
                self.in_doc = False

    def handle_data(self, data):
        if self.in_title:
            self.title_parts.append(data)
        if self.in_doc:
            self.doc_parts.append(data)

    @property
    def title(self):
        return _clean_text("".join(self.title_parts))

    @property
    def doc(self):
        return _clean_document_text("".join(self.doc_parts))


class KanoonClient:
    def __init__(self, session=None):
        self.api_enabled = bool(config.API_TOKEN)
        self.base_url = config.BASE_URL
        self.public_base_url = config.PUBLIC_BASE_URL
        self.headers = {
            "Accept": "application/json",
            "User-Agent": DEFAULT_USER_AGENT,
        }
        if self.api_enabled:
            self.headers["Authorization"] = f"Token {config.API_TOKEN}"
        self.public_headers = {
            "User-Agent": DEFAULT_USER_AGENT,
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
        if isinstance(value, str):
            match = re.search(r"\bof\s+([\d,]+)", value)
            if match:
                return int(match.group(1).replace(",", ""))

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
        if not self.api_enabled:
            raise RuntimeError("Indian Kanoon API token is not configured")

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

    def _request_public_html(self, path, params=None):
        resp = self.session.get(
            f"{self.public_base_url}{path}",
            headers=self.public_headers,
            params=params,
            timeout=self.timeout,
        )

        try:
            resp.raise_for_status()
        except requests.HTTPError as exc:
            detail = self._response_detail(resp)
            raise RuntimeError(
                f"Indian Kanoon public request failed ({resp.status_code}): {detail}"
            ) from exc

        return resp.text

    def _filter_docs(self, docs, doc_type):
        if doc_type == 'judgments':
            return [
                doc for doc in docs
                if 'doctype' not in doc or str(doc.get('doctype')) == "1000"
            ]
        if doc_type in {'acts', 'rules'}:
            return [
                doc for doc in docs
                if 'doctype' not in doc or str(doc.get('doctype')) != "1000"
            ]
        if doc_type != 'all':
            raise ValueError(f"Unsupported document type: {doc_type}")
        return docs

    def _search_api_documents(self, clean_query, doc_type, pagenum, maxpages):
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
        docs = self._filter_docs(docs, doc_type)

        return {
            'docs': docs,
            'total': self._coerce_total(data.get('found'), len(docs)),
            'found': data.get('found'),
            'pagenum': pagenum,
            'source': 'api',
        }

    def _search_public_documents(self, clean_query, doc_type, pagenum):
        html = self._request_public_html(
            "/search/",
            params={"formInput": clean_query, "pagenum": pagenum},
        )

        parser = PublicSearchParser()
        parser.feed(html)
        docs = self._filter_docs(parser.docs, doc_type)
        total = parser.total or len(docs)

        return {
            'docs': docs,
            'total': total,
            'found': total,
            'pagenum': pagenum,
            'source': 'public',
        }

    def search_documents(self, query, doc_type='all', pagenum=0, maxpages=None):
        """
        Search for documents.
        doc_type: filter by type - 'judgments', 'acts', 'rules', 'all' (default is 'all')
        """
        clean_query = " ".join((query or "").split())
        if not clean_query:
            return {'docs': [], 'total': 0, 'found': 0}

        maxpages = self.search_max_pages if maxpages is None else maxpages
        if self.api_enabled:
            try:
                return self._search_api_documents(clean_query, doc_type, pagenum, maxpages)
            except Exception as exc:
                print(f"Indian Kanoon API search failed; using public search fallback: {exc}")

        return self._search_public_documents(clean_query, doc_type, pagenum)

    def _get_public_document(self, doc_id):
        html = self._request_public_html(f"/doc/{int(doc_id)}/")
        parser = PublicDocumentParser()
        parser.feed(html)

        if not parser.doc:
            raise RuntimeError("Indian Kanoon public document page did not contain document text")

        return {
            'title': parser.title or f"Document {doc_id}",
            'doc': parser.doc,
            'source': 'public',
        }

    def get_document(self, doc_id):
        if self.api_enabled:
            try:
                data = self._request_json(f"/doc/{int(doc_id)}/")
                if not isinstance(data, dict):
                    raise RuntimeError("Indian Kanoon API returned an unexpected document response")
                return data
            except Exception as exc:
                print(f"Indian Kanoon API document fetch failed; using public fallback: {exc}")

        return self._get_public_document(doc_id)
