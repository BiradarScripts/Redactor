# Indian Kanoon Legal Privacy Redactor

A FastAPI web app for searching Indian Kanoon judgments and producing a privacy-preserving version of the text.

The masking engine is designed around Indian court judgment structure. It uses memory-safe legal-domain rules by default, and can optionally use spaCy/Presidio/OpenNyAI models on larger instances.

## What It Protects

- Protected people mentioned in sensitive contexts, including victims, survivors, prosecutrix references, witnesses, complainants, informants, minors, and family members.
- Indian judgment relationship patterns such as `S/o`, `D/o`, `W/o`, witness labels like `P.W.7 Name`, and minor/victim phrases like `victim girl Name`.
- Repeated mentions and longer aliases of a protected name across the same judgment.
- Phone numbers, email addresses, PAN/Aadhaar/Voter-style identifiers, address phrases, and locations.

## What It Preserves

- Judges, lawyers, courts, statutes, provisions, precedents, and case numbers where a legal model or legal rules identify them.
- General party names unless there is sensitive context linking them to a protected person category.

## Legal NER Basis

This project incorporates the design lessons from "Named Entity Recognition in Indian Court Judgments" by Kalamkar et al. (NLLP 2022):

- Indian judgments should be split into preamble and judgment body because party, judge, lawyer, court, and date mentions behave differently in each section.
- Legal entities are more specific than generic NER labels: `COURT`, `PETITIONER`, `RESPONDENT`, `JUDGE`, `LAWYER`, `DATE`, `ORG`, `GPE`, `STATUTE`, `PROVISION`, `PRECEDENT`, `CASE_NUMBER`, `WITNESS`, and `OTHER_PERSON`.
- Sentence-level inference plus document-level reconciliation is more accurate for judgment text.

The OpenNyAI/InLegalNER model is preferred when available. The app remains usable without it.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create local environment config:

```bash
cp .env.example .env
```

Set `INDIAN_KANOON_API_TOKEN` in your shell or deployment environment. If you use `.env`, export it before running the app.

Optional heavy NLP mode for larger instances:

```bash
pip install presidio-analyzer presidio-anonymizer spacy
python -m spacy download en_core_web_sm
pip install https://huggingface.co/opennyaiorg/en_legal_ner_sm/resolve/main/en_legal_ner_sm-any-py3-none-any.whl
export ENABLE_HEAVY_NLP=1
export LEGAL_NER_MODEL=en_legal_ner_sm
```

The transformer model `en_legal_ner_trf` is more accurate but is tied to older spaCy 3.2.x packaging. Use it only in a compatible environment.

## Run

```bash
uvicorn app:app --reload
```

Open `http://127.0.0.1:8000`.

Health check:

```bash
curl http://127.0.0.1:8000/health
```

The health response reports whether the Indian Kanoon client and the legal NER model are available.

## Project Structure

```text
app.py              FastAPI routes and safe rendering helpers
kanoon_client.py    Indian Kanoon API client
masking_engine.py   Legal-aware redaction engine
config.py           Environment-based configuration
templates/          Web UI
static/             UI assets
```

## Notes

Automated redaction should be reviewed before publication. The app is optimized for Indian legal text, but no NER model or rule set can guarantee perfect anonymization.
