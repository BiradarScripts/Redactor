# Indian Kanoon Legal Privacy Redactor

A FastAPI web app for searching Indian Kanoon judgments and producing a privacy-preserving version of the text.

The masking engine is designed around Indian court judgment structure. It can use OpenNyAI/InLegalNER when a legal spaCy model is installed, and otherwise falls back to Presidio, spaCy, and legal-domain rules for provisions, statutes, precedents, and sensitive-person context.

## What It Protects

- Protected people mentioned in sensitive contexts, including victims, survivors, prosecutrix references, witnesses, complainants, informants, minors, and family members.
- Repeated mentions of a protected name across the same judgment.
- Phone numbers, email addresses, common Indian identity numbers where Presidio supports them, and locations.

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
python -m spacy download en_core_web_sm
```

Create local environment config:

```bash
cp .env.example .env
```

Set `INDIAN_KANOON_API_TOKEN` in your shell or deployment environment. If you use `.env`, export it before running the app.

Optional legal NER model:

```bash
pip install https://huggingface.co/opennyaiorg/en_legal_ner_sm/resolve/main/en_legal_ner_sm-any-py3-none-any.whl
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
mcn/                MCN research prototype
docs/mcn.md         MCN prototype notes and commands
```

## MCN Research Prototype

The repository also contains a standalone Morphogenic Computation Network prototype based on the MCN/CGB research direction. It is separate from the redactor web app and uses optional ML dependencies.

```bash
pip install -r requirements-mcn.txt
python scripts/train_mcn_toy.py --epochs 2 --n-cells 16 --n-seed-cells 4
python scripts/train_mcn_toy.py --epochs 2 --use-cgb
python scripts/train_mcn_toy.py --model transformer --epochs 2
python scripts/train_mcn_toy.py --model universal --epochs 2
python scripts/train_mcn_toy.py --disable-pruning --epochs 2
python scripts/run_mcn_config.py configs/mcn_scan.yaml --epochs 2 --d-model 64
python scripts/run_mcn_suite.py --epochs 30 --eval-items 0 --include-baselines --include-ablations --run-name final-suite
python scripts/run_mcn_suite.py --split random --epochs 30 --eval-items 0 --run-name sanity-suite
python scripts/run_mcn_roadmap.py --epochs 12 --run-name final-roadmap
```

Single runs write reproducible artifacts to `runs/mcn_toy/`; suite runs write comparable artifacts to `runs/mcn_suite/`; the roadmap suite writes a full local-scale coverage report to `runs/mcn_roadmap/`. Each MCN run includes `config.json`, `config.yaml`, CSV curves, `history.jsonl`, `predictions.jsonl`, `compute_profile.json`, `summary.json`, `checkpoint_best.pt`, `checkpoint_last.pt`, Graphviz DOT exports, and PNG visualizations under `graphs/` and `plots/`. See `docs/mcn.md` for the architecture notes, exposed metrics, baselines, ablations, and roadmap benchmark details.

## Notes

Automated redaction should be reviewed before publication. The app is optimized for Indian legal text, but no NER model or rule set can guarantee perfect anonymization.
