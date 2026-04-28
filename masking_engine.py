import logging
import os
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import spacy
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import SpacyNlpEngine


LOGGER = logging.getLogger(__name__)

DEFAULT_SPACY_MODEL = os.getenv("SPACY_MODEL", "en_core_web_sm")
LEGAL_MODEL_CANDIDATES = (
    os.getenv("LEGAL_NER_MODEL", "").strip(),
    "en_legal_ner_trf",
    "en_legal_ner_sm",
)

LEGAL_ENTITY_LABELS = {
    "COURT",
    "PETITIONER",
    "RESPONDENT",
    "JUDGE",
    "LAWYER",
    "DATE",
    "ORG",
    "GPE",
    "STATUTE",
    "PROVISION",
    "PRECEDENT",
    "CASE_NUMBER",
    "WITNESS",
    "OTHER_PERSON",
}

PUBLIC_LEGAL_LABELS = {
    "COURT",
    "JUDGE",
    "LAWYER",
    "STATUTE",
    "PROVISION",
    "PRECEDENT",
    "CASE_NUMBER",
    "DATE",
}

PERSON_LEGAL_LABELS = {
    "PETITIONER",
    "RESPONDENT",
    "WITNESS",
    "OTHER_PERSON",
}

DIRECT_IDENTIFIER_LABELS = {
    "EMAIL_ADDRESS": "[EMAIL]",
    "PHONE_NUMBER": "[PHONE]",
    "IN_AADHAAR": "[ID]",
    "IN_PAN": "[ID]",
    "IN_PASSPORT": "[ID]",
    "IN_VOTER": "[ID]",
    "CREDIT_CARD": "[ID]",
    "IBAN_CODE": "[ID]",
}

STATUTE_SHORTFORMS = {
    "ipc": "Indian Penal Code",
    "i.p.c": "Indian Penal Code",
    "i.p.c.": "Indian Penal Code",
    "penal code": "Indian Penal Code",
    "crpc": "Criminal Procedure Code",
    "cr.p.c": "Criminal Procedure Code",
    "cr.p.c.": "Criminal Procedure Code",
    "code of criminal procedure": "Criminal Procedure Code",
    "cpc": "Code of Civil Procedure",
    "c.p.c": "Code of Civil Procedure",
    "c.p.c.": "Code of Civil Procedure",
    "constitution": "Constitution of India",
    "mv act": "Motor Vehicles Act",
    "motor vehicles act": "Motor Vehicles Act",
    "sarfaesi": "SARFAESI Act",
    "ndps": "Narcotic Drugs and Psychotropic Substances Act",
    "pocso": "Protection of Children from Sexual Offences Act",
}

PREAMBLE_BOUNDARY_RE = re.compile(
    r"(?im)^\s*(J\s*U\s*D\s*G\s*M\s*E\s*N\s*T|JUDG(?:E)?MENT|"
    r"O\s*R\s*D\s*E\s*R|ORDER|ORAL\s+JUDGMENT|COMMON\s+ORDER)\s*[:.-]*\s*$"
)

NUMBERED_PARAGRAPH_RE = re.compile(r"(?m)^\s*(?:\d+\.|\(\d+\))\s+[A-Z]")

SENSITIVE_PERSON_RE = re.compile(
    r"\b(?:victim|deceased|prosecutrix|survivor|minor|child|girl|boy|"
    r"complainant|informant|witness|pw[-\s]?\d+|cw[-\s]?\d+|"
    r"wife|husband|son|daughter|mother|father|brother|sister|"
    r"guardian|parent|relative|family|widow)\b",
    re.IGNORECASE,
)

PUBLIC_ROLE_RE = re.compile(
    r"\b(?:justice|hon'?ble|judge|bench|advocate|counsel|solicitor|"
    r"public prosecutor|amicus|registrar|magistrate)\b",
    re.IGNORECASE,
)

PERSON_AFTER_SENSITIVE_ROLE_RE = re.compile(
    r"\b(?:victim|deceased|prosecutrix|survivor|complainant|informant|witness|"
    r"pw[-\s]?\d+|cw[-\s]?\d+)\s*(?:namely|named|name(?:d)?\s+as|:|-|,)?\s*"
    r"([A-Z][A-Za-z.'-]+(?:[ \t]+[A-Z][A-Za-z.'-]+){0,4})"
)

PERSON_BEFORE_SENSITIVE_ROLE_RE = re.compile(
    r"([A-Z][A-Za-z.'-]+(?:[ \t]+[A-Z][A-Za-z.'-]+){0,4})"
    r"\s*(?:,|-|\()?\s*(?:victim|deceased|prosecutrix|survivor|minor|"
    r"complainant|informant|witness)\b",
    re.IGNORECASE,
)

FAMILY_MEMBER_NAME_RE = re.compile(
    r"\b(?:his|her|the)?\s*(?:wife|husband|son|daughter|mother|father|brother|sister|guardian)"
    r"\s+(?:of\s+)?([A-Z][A-Za-z.'-]+(?:[ \t]+[A-Z][A-Za-z.'-]+){0,4})"
)

STATUTE_RE = re.compile(
    r"\b(?:I\.?\s*P\.?\s*C\.?|Cr\.?\s*P\.?\s*C\.?|C\.?\s*P\.?\s*C\.?|"
    r"Constitution(?:\s+of\s+India)?|"
    r"[A-Z][A-Za-z&()'-]+(?:\s+[A-Z0-9][A-Za-z0-9&()'-]+){0,8}\s+Act(?:,?\s*\d{4})?)\b"
)

PROVISION_RE = re.compile(
    r"\b(?:section|sections|article|articles|order|rule|rules)\s+"
    r"\d+[A-Za-z0-9()/-]*(?:\s*(?:,|and|or|to|-)\s*\d+[A-Za-z0-9()/-]*)*",
    re.IGNORECASE,
)

CASE_NUMBER_RE = re.compile(
    r"\b(?:(?:criminal|civil|writ|special\s+leave|tax|company)\s+)?"
    r"(?:appeal|petition|application|suit|case|no\.?)\s+"
    r"(?:no\.?\s*)?[\w./()-]+(?:\s+of\s+\d{4})?",
    re.IGNORECASE,
)

PRECEDENT_RE = re.compile(
    r"\b[A-Z][A-Za-z.'@&-]+(?:[ \t]+[A-Z][A-Za-z.'@&-]+){0,5}"
    r"\s+(?:v\.?|vs\.?|versus)\s+"
    r"[A-Z][A-Za-z.'@&-]+(?:[ \t]+[A-Z][A-Za-z.'@&-]+){0,6}"
)


@dataclass(frozen=True)
class EntitySpan:
    start: int
    end: int
    label: str
    text: str
    source: str
    score: float = 1.0

    @property
    def length(self) -> int:
        return self.end - self.start


class LoadedSpacyNlpEngine(SpacyNlpEngine):
    def __init__(self, loaded_spacy_model):
        super().__init__()
        self.nlp = {"en": loaded_spacy_model}


def _download_spacy_model(model_name: str) -> None:
    python = os.path.join(os.path.dirname(sys.executable), "python")
    subprocess.run([python, "-m", "spacy", "download", model_name], check=True)


def _load_spacy_model(model_name: str):
    try:
        return spacy.load(model_name)
    except OSError:
        if os.getenv("AUTO_DOWNLOAD_SPACY_MODEL", "1") != "1":
            raise
        LOGGER.info("Downloading missing spaCy model %s", model_name)
        _download_spacy_model(model_name)
        return spacy.load(model_name)


def _load_optional_legal_model():
    for model_name in LEGAL_MODEL_CANDIDATES:
        if not model_name:
            continue
        try:
            model = spacy.load(model_name)
            LOGGER.info("Loaded legal NER model %s", model_name)
            return model, model_name
        except Exception as exc:
            LOGGER.info("Legal NER model %s unavailable: %s", model_name, exc)
    return None, "presidio_spacy_fallback"


def normalize_name(value: str) -> str:
    cleaned = re.sub(r"\b(?:mr|mrs|ms|dr|smt|shri|kumari|master)\.?\s+", "", value, flags=re.I)
    cleaned = re.sub(r"[^a-z0-9]+", " ", cleaned.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def normalize_statute(value: str) -> str:
    key = re.sub(r"\s+", " ", value.lower().replace(".", ".")).strip()
    compact_key = key.replace(".", "")
    return STATUTE_SHORTFORMS.get(key) or STATUTE_SHORTFORMS.get(compact_key) or value.strip()


def split_preamble_and_body(text: str) -> tuple[str, str, int]:
    boundary = PREAMBLE_BOUNDARY_RE.search(text)
    if boundary:
        return text[: boundary.start()], text[boundary.start() :], boundary.start()

    paragraph = NUMBERED_PARAGRAPH_RE.search(text)
    if paragraph and paragraph.start() > 250:
        return text[: paragraph.start()], text[paragraph.start() :], paragraph.start()

    fallback = min(len(text), 2500)
    return text[:fallback], text[fallback:], fallback


class SmartMasker:
    def __init__(self):
        self.generic_nlp = _load_spacy_model(DEFAULT_SPACY_MODEL)
        self.legal_nlp, self.legal_model_name = _load_optional_legal_model()
        self.analyzer = AnalyzerEngine(nlp_engine=LoadedSpacyNlpEngine(self.generic_nlp))
        self.name_mapping: dict[str, str] = {}

    def reset_mapping(self):
        self.name_mapping = {}

    def _context(
        self,
        text: str,
        start: int,
        end: int,
        window: int = 120,
        right_boundary: int | None = None,
    ) -> str:
        right = min(len(text), end + window)
        if right_boundary is not None and end <= right_boundary:
            right = min(right, right_boundary)
        return text[max(0, start - window) : right]

    def _is_sensitive_context(self, text: str, start: int, end: int, body_start: int) -> bool:
        return bool(SENSITIVE_PERSON_RE.search(self._context(text, start, end, right_boundary=body_start)))

    def _is_public_context(self, text: str, start: int, end: int) -> bool:
        before = text[max(0, start - 55) : start]
        after = text[end : min(len(text), end + 30)]
        return bool(PUBLIC_ROLE_RE.search(before) or PUBLIC_ROLE_RE.match(after.strip()))

    def _supported_presidio_entities(self) -> list[str]:
        requested = {
            "PERSON",
            "LOCATION",
            *DIRECT_IDENTIFIER_LABELS.keys(),
        }
        supported = set(self.analyzer.get_supported_entities(language="en"))
        return sorted(requested & supported)

    def _presidio_spans(self, text: str) -> list[EntitySpan]:
        spans: list[EntitySpan] = []
        for result in self.analyzer.analyze(
            text=text,
            entities=self._supported_presidio_entities(),
            language="en",
        ):
            spans.append(
                EntitySpan(
                    start=result.start,
                    end=result.end,
                    label=result.entity_type,
                    text=text[result.start : result.end],
                    source="presidio",
                    score=result.score,
                )
            )
        return spans

    def _legal_spans_for_doc(self, text: str, offset: int = 0) -> list[EntitySpan]:
        if not text.strip() or self.legal_nlp is None:
            return []

        doc = self.legal_nlp(text)
        return [
            EntitySpan(
                start=offset + ent.start_char,
                end=offset + ent.end_char,
                label=ent.label_,
                text=ent.text,
                source="legal_ner",
            )
            for ent in doc.ents
            if ent.label_ in LEGAL_ENTITY_LABELS
        ]

    def _sentence_level_legal_spans(self, body: str, offset: int) -> list[EntitySpan]:
        if not body.strip() or self.legal_nlp is None:
            return []

        spans: list[EntitySpan] = []
        doc = self.generic_nlp(body)
        for sent in doc.sents:
            sentence = sent.text.strip()
            if not sentence:
                continue
            spans.extend(self._legal_spans_for_doc(sentence, offset + sent.start_char))
        return spans

    def _legal_spans(self, text: str) -> tuple[list[EntitySpan], int]:
        preamble, body, body_start = split_preamble_and_body(text)
        if self.legal_nlp is None:
            return self._filter_overlaps(self._rule_legal_spans(text)), body_start

        spans = self._legal_spans_for_doc(preamble, 0)
        if os.getenv("LEGAL_NER_SENTENCE_LEVEL", "1") == "1":
            spans.extend(self._sentence_level_legal_spans(body, body_start))
        else:
            spans.extend(self._legal_spans_for_doc(body, body_start))
        return spans, body_start

    def _rule_legal_spans(self, text: str) -> list[EntitySpan]:
        patterns = (
            ("STATUTE", STATUTE_RE),
            ("PROVISION", PROVISION_RE),
            ("CASE_NUMBER", CASE_NUMBER_RE),
            ("PRECEDENT", PRECEDENT_RE),
        )
        spans: list[EntitySpan] = []
        for label, regex in patterns:
            for match in regex.finditer(text):
                value = match.group().strip()
                if value:
                    spans.append(EntitySpan(match.start(), match.end(), label, value, "legal_rules"))
        return spans

    def _rule_spans(self, text: str) -> list[EntitySpan]:
        spans: list[EntitySpan] = []
        for regex in (PERSON_AFTER_SENSITIVE_ROLE_RE, PERSON_BEFORE_SENSITIVE_ROLE_RE, FAMILY_MEMBER_NAME_RE):
            for match in regex.finditer(text):
                start, end = match.span(1)
                value = text[start:end].strip(" .,:;()")
                normalized = normalize_name(value)
                if len(normalized) >= 3 and normalized not in {"the", "this", "that", "said", "court"}:
                    spans.append(EntitySpan(start, end, "SENSITIVE_PERSON", value, "rules"))
        return spans

    def _overlapping_legal_labels(self, span: EntitySpan, legal_spans: Iterable[EntitySpan]) -> set[str]:
        return {
            legal.label
            for legal in legal_spans
            if legal.start < span.end and span.start < legal.end
        }

    def _should_mask_person(self, span: EntitySpan, legal_labels: set[str], text: str, body_start: int) -> bool:
        if legal_labels & PUBLIC_LEGAL_LABELS:
            return False
        if "WITNESS" in legal_labels or span.label == "SENSITIVE_PERSON":
            return True
        if self._is_public_context(text, span.start, span.end):
            return False
        return self._is_sensitive_context(text, span.start, span.end, body_start)

    def _add_document_level_name_mentions(self, text: str, spans: list[EntitySpan]) -> list[EntitySpan]:
        protected_names = {
            span.text
            for span in spans
            if span.label in {"PERSON", "PETITIONER", "RESPONDENT", "WITNESS", "OTHER_PERSON", "SENSITIVE_PERSON"}
            and len(normalize_name(span.text)) >= 3
        }
        expanded = list(spans)

        for name in protected_names:
            normalized = normalize_name(name)
            if not normalized or normalized in {"state", "union", "court"}:
                continue
            tokens = re.findall(r"[A-Za-z0-9]+", name)
            if not tokens or (len(tokens) == 1 and len(tokens[0]) < 4):
                continue
            pattern = re.compile(r"\b" + r"[\s.,'@-]+".join(map(re.escape, tokens)) + r"\b", re.IGNORECASE)
            for match in pattern.finditer(text):
                expanded.append(
                    EntitySpan(match.start(), match.end(), "PERSON", text[match.start() : match.end()], "document_coref")
                )
        return expanded

    def _filter_overlaps(self, spans: Iterable[EntitySpan]) -> list[EntitySpan]:
        ordered = sorted(spans, key=lambda span: (span.start, -span.length, span.label))
        selected: list[EntitySpan] = []
        occupied: list[tuple[int, int]] = []

        for span in ordered:
            if span.start < 0 or span.end <= span.start:
                continue
            if any(start < span.end and span.start < end for start, end in occupied):
                continue
            selected.append(span)
            occupied.append((span.start, span.end))

        return sorted(selected, key=lambda span: span.start)

    def _replacement_for(self, span: EntitySpan) -> str:
        if span.label in DIRECT_IDENTIFIER_LABELS:
            return DIRECT_IDENTIFIER_LABELS[span.label]
        if span.label in {"LOCATION", "GPE"}:
            return "[LOC]"

        normalized = normalize_name(span.text)
        if normalized not in self.name_mapping:
            self.name_mapping[normalized] = f"[PROTECTED_PERSON_{len(self.name_mapping) + 1}]"
        return self.name_mapping[normalized]

    def _apply_replacements(self, text: str, spans: list[EntitySpan]) -> str:
        output = []
        cursor = 0
        for span in spans:
            output.append(text[cursor : span.start])
            output.append(self._replacement_for(span))
            cursor = span.end
        output.append(text[cursor:])
        return "".join(output)

    def _analysis(self, text: str, masked_text: str, spans: list[EntitySpan], legal_spans: list[EntitySpan], body_start: int):
        pii_counts = Counter(span.label for span in spans)
        legal_counts = Counter(span.label for span in legal_spans)
        normalized_statutes = sorted(
            {
                normalize_statute(span.text)
                for span in legal_spans
                if span.label == "STATUTE" and normalize_statute(span.text)
            }
        )

        protected_people = sum(
            1
            for span in spans
            if span.label in {"PERSON", "PETITIONER", "RESPONDENT", "WITNESS", "OTHER_PERSON", "SENSITIVE_PERSON"}
        )

        return {
            "total_masked": len(spans),
            "protected_person_count": protected_people,
            "victim_family_count": protected_people,
            "phone_count": pii_counts["PHONE_NUMBER"],
            "email_count": pii_counts["EMAIL_ADDRESS"],
            "location_count": pii_counts["LOCATION"] + pii_counts["GPE"],
            "id_count": sum(pii_counts[label] for label in DIRECT_IDENTIFIER_LABELS if label not in {"PHONE_NUMBER", "EMAIL_ADDRESS"}),
            "legal_entity_count": len(legal_spans),
            "statute_count": legal_counts["STATUTE"],
            "provision_count": legal_counts["PROVISION"],
            "precedent_count": legal_counts["PRECEDENT"],
            "legal_entity_counts": dict(legal_counts),
            "normalized_statutes": normalized_statutes[:12],
            "original_length": len(text),
            "masked_length": len(masked_text),
            "reduction_percentage": int(round(((len(masked_text) / len(text)) - 1) * 100)) if text else 0,
            "legal_model": self.legal_model_name,
            "legal_model_available": self.legal_nlp is not None,
            "preamble_length": body_start,
        }

    def mask_victims_and_family(self, text):
        if not text:
            return "", {}

        self.reset_mapping()

        legal_spans, body_start = self._legal_spans(text)
        presidio_spans = self._presidio_spans(text)
        rule_spans = self._rule_spans(text)

        spans_to_mask: list[EntitySpan] = []
        for span in [*presidio_spans, *legal_spans, *rule_spans]:
            legal_labels = self._overlapping_legal_labels(span, legal_spans)

            if span.label in DIRECT_IDENTIFIER_LABELS:
                spans_to_mask.append(span)
            elif span.label in {"LOCATION", "GPE"}:
                spans_to_mask.append(span)
            elif span.label in PERSON_LEGAL_LABELS or span.label in {"PERSON", "SENSITIVE_PERSON"}:
                if self._should_mask_person(span, legal_labels or {span.label}, text, body_start):
                    spans_to_mask.append(span)

        spans_to_mask = self._add_document_level_name_mentions(text, spans_to_mask)
        spans_to_mask = self._filter_overlaps(spans_to_mask)
        masked_text = self._apply_replacements(text, spans_to_mask)

        return masked_text, self._analysis(text, masked_text, spans_to_mask, legal_spans, body_start)
