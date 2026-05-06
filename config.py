import os

API_TOKEN = os.getenv("INDIAN_KANOON_API_TOKEN", "").strip()
BASE_URL = os.getenv("INDIAN_KANOON_BASE_URL", "https://api.indiankanoon.org").rstrip("/")
PUBLIC_BASE_URL = os.getenv("INDIAN_KANOON_PUBLIC_BASE_URL", "https://indiankanoon.org").rstrip("/")
REQUEST_TIMEOUT = int(os.getenv("INDIAN_KANOON_TIMEOUT", "20"))
SEARCH_MAX_PAGES = int(os.getenv("INDIAN_KANOON_SEARCH_MAX_PAGES", "1"))
