import os

API_TOKEN = os.getenv("INDIAN_KANOON_API_TOKEN", "").strip()
BASE_URL = os.getenv("INDIAN_KANOON_BASE_URL", "https://api.indiankanoon.org").rstrip("/")
REQUEST_TIMEOUT = int(os.getenv("INDIAN_KANOON_TIMEOUT", "20"))
