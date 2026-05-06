import html
import re
from pathlib import Path

from fastapi import FastAPI, Request, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import uvicorn
import traceback

# Import your modules
from kanoon_client import KanoonClient
from masking_engine import SmartMasker

app = FastAPI()

# ✅ Base directory (important for Render)
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

# ✅ Templates setup
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ✅ Global variables
client = None
masker = None

MASK_TOKEN_RE = re.compile(
    r"(\[(?:PROTECTED_PERSON_\d+|VICTIM/FAMILY_\d+|PHONE|EMAIL|LOC|ID)\])"
)
DOCUMENT_BLOCK_BOUNDARY_RE = re.compile(
    r"(?:(?<=</blockquote>)|(?<=</p>)|(?<=</pre>)|\n\s*\n+)",
    re.IGNORECASE,
)


def render_index(request: Request, **context):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"request": request, **context},
    )


def highlighted_masked_text(text: str) -> str:
    escaped = html.escape(text or "")
    return MASK_TOKEN_RE.sub(r"<mark>\1</mark>", escaped)


def document_blocks(text: str) -> list[str]:
    blocks = []
    for block in DOCUMENT_BLOCK_BOUNDARY_RE.split(text or ""):
        cleaned = block.strip()
        if cleaned:
            blocks.append(cleaned)
    return blocks or [text or ""]


def highlighted_masked_blocks(text: str) -> list[str]:
    return [highlighted_masked_text(block) for block in document_blocks(text)]


def get_kanoon_client() -> KanoonClient:
    global client
    if client is None:
        client = KanoonClient()
    return client


def get_smart_masker() -> SmartMasker:
    global masker
    if masker is None:
        masker = SmartMasker()
    return masker


@app.on_event("startup")
async def startup_event():
    global client
    try:
        print("Starting initialization...")

        client = KanoonClient()
        print("KanoonClient initialized")

    except Exception as e:
        print("KanoonClient unavailable:", e)
        traceback.print_exc()


# ✅ Home route
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return render_index(request)


# ✅ Search route
@app.post("/search", response_class=HTMLResponse)
async def search(request: Request, query: str = Form(...)):
    try:
        query = " ".join(query.split())
        if not query:
            return render_index(request, docs=[], query=query, total=0, error="Enter a search term.")

        search_client = get_kanoon_client()
        results = search_client.search_documents(query)
        docs = results.get('docs', []) if results else []
        total = results.get('total', len(docs)) if results else len(docs)

        return render_index(request, docs=docs, query=query, total=total)

    except Exception as e:
        print(f"❌ Search error: {e}")
        traceback.print_exc()

        return render_index(request, docs=[], query=query, total=0, error=str(e))


# ✅ Process document route
@app.get("/process/{doc_id}", response_class=HTMLResponse)
async def process_doc(request: Request, doc_id: int):
    try:
        search_client = get_kanoon_client()
        smart_masker = get_smart_masker()

        raw_data = search_client.get_document(doc_id)
        original_text = raw_data.get('doc', 'Error fetching document')
        title = raw_data.get('title', 'Unknown Title')

        masked_text, analysis = smart_masker.mask_victims_and_family(original_text)

        return render_index(
            request,
            doc_id=doc_id,
            title=title,
            original_text=original_text,
            masked_text=masked_text,
            masked_html=highlighted_masked_text(masked_text),
            original_blocks=document_blocks(original_text),
            masked_blocks=highlighted_masked_blocks(masked_text),
            view_mode="compare",
            analysis=analysis,
        )

    except Exception as e:
        print(f"❌ Process error: {e}")
        traceback.print_exc()

        return render_index(request, error=str(e))


# ✅ Health check route (optional but useful)
@app.get("/health")
async def health():
    search_client = get_kanoon_client()
    return {
        "status": "ok",
        "kanoon_client": client is not None,
        "search_backend": "api" if search_client.api_enabled else "public",
        "masker": True,
        "legal_model": masker.legal_model_name if masker else "rules_only_lazy",
    }


# ✅ Local run
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
