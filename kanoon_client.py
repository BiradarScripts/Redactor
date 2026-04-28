import requests
import config

class KanoonClient:
    def __init__(self):
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

    def search_documents(self, query, doc_type='all'):
        """
        Search for documents. 
        doc_type: filter by type - 'judgments', 'acts', 'rules', 'all' (default is 'all')
        """
        try:
            # Use POST request with data as per API
            resp = requests.post(
                f"{self.base_url}/search/",
                headers=self.headers,
                data={"formInput": query},
                timeout=self.timeout,
            )
            
            if resp.status_code == 200:
                data = resp.json()
                docs = data.get('docs', [])
                
                # Filter based on doc_type
                if doc_type == 'judgments':
                    docs = [doc for doc in docs if doc.get('doctype') == 1000]
                elif doc_type == 'acts':
                    docs = [doc for doc in docs if doc.get('doctype') != 1000]
                # 'all' returns everything (no filtering)
                
                return {'docs': docs, 'total': len(docs)}
            else:
                print(f"API returned status: {resp.status_code}")
                return {'docs': [], 'total': 0}
        except Exception as e:
            print(f"Search error: {e}")
            return {'docs': [], 'total': 0}

    def get_document(self, doc_id):
        try:
            resp = requests.post(f"{self.base_url}/doc/{doc_id}/", headers=self.headers, timeout=self.timeout)
            return resp.json() if resp.status_code == 200 else {}
        except Exception as e:
            print(f"Get document error: {e}")
            return {}
