from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from kanoon_client import KanoonClient
from masking_engine import SmartMasker
import uvicorn
import traceback
import os

app = FastAPI()

# ✅ Use absolute path (important for Render)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# ✅ Fix static folder (don't use ".")
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

# Initialize Logic
client = KanoonClient()
masker = SmartMasker()


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/search", response_class=HTMLResponse)
async def search(request: Request, query: str = Form(...)):
    try:
        results = client.search_documents(query)
        docs = results.get('docs', []) if results else []

        return templates.TemplateResponse("index.html", {
            "request": request,
            "docs": docs,
            "query": query
        })

    except Exception as e:
        print(f"Search error: {e}")
        traceback.print_exc()

        return templates.TemplateResponse("index.html", {
            "request": request,
            "docs": [],
            "query": query,
            "error": str(e)
        })


@app.get("/process/{doc_id}", response_class=HTMLResponse)
async def process_doc(request: Request, doc_id: int):
    try:
        raw_data = client.get_document(doc_id)

        original_text = raw_data.get('doc', 'Error fetching document')
        title = raw_data.get('title', 'Unknown Title')

        # Masking
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
        print(f"Process error: {e}")
        traceback.print_exc()

        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": str(e)
        })


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
