from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import traceback
import os

# Import your modules
from kanoon_client import KanoonClient
from masking_engine import SmartMasker

app = FastAPI()

# ✅ Base directory (important for Render)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ✅ Templates & Static setup
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

# ✅ Global variables (initialized later)
client = None
masker = None


# ✅ Startup event (prevents crash on deploy)
@app.on_event("startup")
async def startup_event():
    global client, masker
    try:
        print("🚀 Starting initialization...")

        client = KanoonClient()
        print("✅ KanoonClient initialized")

        masker = SmartMasker()
        print("✅ SmartMasker initialized")

    except Exception as e:
        print("❌ Initialization failed:", e)
        traceback.print_exc()


# ✅ Home route
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ✅ Search route
@app.post("/search", response_class=HTMLResponse)
async def search(request: Request, query: str = Form(...)):
    try:
        if client is None:
            raise Exception("Client not initialized")

        results = client.search_documents(query)
        docs = results.get('docs', []) if results else []

        return templates.TemplateResponse("index.html", {
            "request": request,
            "docs": docs,
            "query": query
        })

    except Exception as e:
        print(f"❌ Search error: {e}")
        traceback.print_exc()

        return templates.TemplateResponse("index.html", {
            "request": request,
            "docs": [],
            "query": query,
            "error": str(e)
        })


# ✅ Process document route
@app.get("/process/{doc_id}", response_class=HTMLResponse)
async def process_doc(request: Request, doc_id: int):
    try:
        if client is None or masker is None:
            raise Exception("Services not initialized")

        # Fetch document
        raw_data = client.get_document(doc_id)
        original_text = raw_data.get('doc', 'Error fetching document')
        title = raw_data.get('title', 'Unknown Title')

        # Mask data
        masked_text, analysis = masker.mask_victims_and_family(original_text)

        return templates.TemplateResponse("index.html", {
            "request": request,
            "doc_id": doc_id,
            "title": title,
            "original_text": original_text,
            "masked_text": masked_text,
            "view_mode": "compare",
            "analysis": analysis
        })

    except Exception as e:
        print(f"❌ Process error: {e}")
        traceback.print_exc()

        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": str(e)
        })


# ✅ Health check route (VERY IMPORTANT for Render debugging)
@app.get("/health")
async def health():
    return {"status": "ok"}


# ✅ Local run (not used in Render, but safe)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
