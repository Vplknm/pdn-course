"""Гибридный распознаватель ПДн: модель NER, правила и валидаторы."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterable

from .person_ner import PersonModelRecognizer
from .validators import inn_valid, luhn_valid, ogrnip_valid, snils_valid


@dataclass(frozen=True, order=True)
class Entity:
    start: int
    end: int
    label: str
    text: str

    def as_list(self) -> list[object]:
        return [self.label, self.start, self.end]


@dataclass(frozen=True)
class Rule:
    label: str
    pattern: re.Pattern[str]
    validator: Callable[[str], bool] | None = None
    priority: int = 50


FLAGS = re.IGNORECASE | re.UNICODE
RULES = (
    Rule("EMAIL", re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+", FLAGS), priority=100),
    Rule("URL", re.compile(r"(?<!\w)(?:https?://|www\.)[^\s<>()]+", FLAGS), priority=99),
    Rule("IP", re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)"),
         lambda value: all(0 <= int(part) <= 255 for part in value.split(".")), 98),
    Rule("SNILS", re.compile(r"(?<!\d)\d{3}[- ]?\d{3}[- ]?\d{3}[ -]?\d{2}(?!\d)"), snils_valid, 97),
    Rule("INN", re.compile(r"(?<!\d)(?:\d{12}|\d{10})(?!\d)"), inn_valid, 96),
    Rule("OGRNIP", re.compile(r"(?<!\d)\d{15}(?!\d)"), ogrnip_valid, 95),
    Rule("BANK_CARD", re.compile(r"(?<!\d)(?:\d[ -]?){15}\d(?!\d)"), luhn_valid, 94),
    Rule("PHONE", re.compile(r"(?<!\d)(?:\+7|8)[\s-]?\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}(?!\d)"), priority=93),
    Rule("PASSPORT", re.compile(r"(?<!\d)\d{2}[ ]?\d{2}[ ]?\d{6}(?!\d)"), priority=92),
    Rule("CADASTRAL", re.compile(r"(?<!\d)\d{2}:\d{2}:\d{6,7}:\d+(?!\d)"), priority=91),
    Rule("KLADR", re.compile(r"(?<!\d)\d{13}(?:\d{4})?(?!\d)"), priority=90),
    Rule("MARRIAGE_CERT", re.compile(r"(?i)(?<=о браке: )[IVXА-ЯA-Z-]{1,8}[ ]?№?[ ]?\d{5,8}"), priority=89),
    Rule("BIRTH_CERT", re.compile(r"(?i)(?<=о рождении: )[IVXА-ЯA-Z-]{1,8}[ ]?№?[ ]?\d{5,8}"), priority=89),
)

WORD_RE = re.compile(r"[А-ЯЁа-яёA-Za-z]+(?:-[А-ЯЁа-яёA-Za-z]+)?|\d+|[^\w\s]", re.UNICODE)


class Recognizer:
    """Распознаёт сущности и разрешает пересечения по приоритету и длине."""

    def __init__(self, person_model: PersonModelRecognizer | None = None):
        self.person_model = person_model or PersonModelRecognizer()

    def recognize(self, text: str) -> list[Entity]:
        candidates: list[tuple[int, Entity]] = []
        for rule in RULES:
            for match in rule.pattern.finditer(text):
                raw = match.group(0).rstrip(".,;:!?")
                if not raw:
                    continue
                if rule.validator and not rule.validator(raw):
                    continue
                entity = Entity(match.start(), match.start() + len(raw), rule.label, raw)
                candidates.append((rule.priority, entity))

        for part in self.person_model.recognize(text):
            candidates.append((80, Entity(part.start, part.end, part.label, part.text)))

        accepted: list[Entity] = []
        for _priority, entity in sorted(candidates, key=lambda item: (-item[0], -(item[1].end-item[1].start), item[1].start)):
            if any(entity.start < other.end and other.start < entity.end for other in accepted):
                continue
            accepted.append(entity)
        return sorted(accepted)

    def as_lists(self, text: str) -> list[list[object]]:
        return [entity.as_list() for entity in self.recognize(text)]

    def bio_tags(self, text: str, entities: Iterable[Entity] | None = None) -> list[tuple[str, str]]:
        entities = list(self.recognize(text) if entities is None else entities)
        result: list[tuple[str, str]] = []
        for token_match in WORD_RE.finditer(text):
            token = token_match.group(0)
            tag = "O"
            for entity in entities:
                if token_match.start() >= entity.start and token_match.end() <= entity.end:
                    prefix = "B" if token_match.start() == entity.start else "I"
                    tag = f"{prefix}-{entity.label}"
                    break
            result.append((token, tag))
        return result
