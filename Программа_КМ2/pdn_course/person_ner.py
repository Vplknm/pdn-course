"""Модельное распознавание ФИО с помощью Natasha/Slovnet."""

from __future__ import annotations

import re
from dataclasses import dataclass

try:
    import pymorphy3
    from natasha import Doc, NewsEmbedding, NewsNERTagger, Segmenter
except ImportError as error:  # понятная ошибка вместо скрытого возврата к словарям
    raise RuntimeError(
        "Для распознавания ФИО установите зависимости: "
        "python -m pip install -r requirements.txt"
    ) from error


WORD_RE = re.compile(r"[А-ЯЁа-яё]+(?:-[А-ЯЁа-яё]+)?", re.UNICODE)
PATRONYMIC_ENDING_RE = re.compile(
    r"(?:"
    r"(?:ов|ев|ь)ич(?:а|у|ем|е)?|"
    r"(?:ов|ев)н(?:а|ой|ую|е|ы)|"
    r"иничн(?:а|ой|ую|е|ы)"
    r")$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PersonPart:
    start: int
    end: int
    label: str
    text: str


class PersonModelRecognizer:
    """Находит PER моделью NER и делит ФИО на компоненты морфологией."""

    def __init__(self) -> None:
        self.segmenter = Segmenter()
        self.embedding = NewsEmbedding()
        self.ner_tagger = NewsNERTagger(self.embedding)
        self.morph = pymorphy3.MorphAnalyzer()

    def recognize(self, text: str) -> list[PersonPart]:
        doc = Doc(text)
        doc.segment(self.segmenter)
        doc.tag_ner(self.ner_tagger)

        result: list[PersonPart] = []
        for span in doc.spans:
            if span.type != "PER":
                continue
            span_start, span_end = self._expanded_bounds(text, span.start, span.stop)
            span_text = text[span_start:span_end]
            words = list(WORD_RE.finditer(span_text))
            labels = [self._component_label(match.group(0)) for match in words]
            labels = self._resolve_unknown_labels(labels)
            for match, label in zip(words, labels):
                if label is None:
                    continue
                start = span_start + match.start()
                end = span_start + match.end()
                result.append(PersonPart(start, end, label, text[start:end]))
        return result

    def _expanded_bounds(self, text: str, start: int, end: int) -> tuple[int, int]:
        """Добавляет соседние части ФИО, которые модель оставила рядом с PER."""

        words = list(WORD_RE.finditer(text))
        for match in reversed(words):
            if match.end() > start:
                continue
            separator = text[match.end():start]
            if separator.isspace() and self._component_label(match.group(0)) in {"SURNAME", "NAME", "PATRONYMIC"}:
                start = match.start()
            break
        for match in words:
            if match.start() < end:
                continue
            separator = text[end:match.start()]
            if separator.isspace() and self._component_label(match.group(0)) in {"SURNAME", "NAME", "PATRONYMIC"}:
                end = match.end()
            break
        return start, end

    def _component_label(self, word: str) -> str | None:
        scores = {"NAME": 0.0, "SURNAME": 0.0, "PATRONYMIC": 0.0}
        for parse in self.morph.parse(word):
            grammemes = parse.tag.grammemes
            if "Name" in grammemes:
                scores["NAME"] = max(scores["NAME"], parse.score)
            if "Surn" in grammemes:
                scores["SURNAME"] = max(scores["SURNAME"], parse.score)
            if "Patr" in grammemes:
                scores["PATRONYMIC"] = max(scores["PATRONYMIC"], parse.score)
        # Формы «Александровичем/Александровиче» могут иметь более вероятный
        # словарный разбор как фамилия. Окончание уточняет компонент только
        # внутри PER-спана, который уже найден NER-моделью.
        if scores["PATRONYMIC"] > 0 and PATRONYMIC_ENDING_RE.search(word):
            return "PATRONYMIC"
        label, score = max(scores.items(), key=lambda item: item[1])
        return label if score > 0 else None

    @staticmethod
    def _resolve_unknown_labels(labels: list[str | None]) -> list[str | None]:
        """Разрешает редкие неизвестные формы внутри уже найденного моделью PER-спана."""

        resolved = list(labels)
        if len(resolved) == 1 and resolved[0] is None:
            resolved[0] = "NAME"
            return resolved

        for index, label in enumerate(resolved):
            if label is not None:
                continue
            if index == len(resolved) - 1 and "PATRONYMIC" not in resolved:
                resolved[index] = "PATRONYMIC"
            elif "NAME" not in resolved:
                resolved[index] = "NAME"
            else:
                resolved[index] = "SURNAME"
        return resolved
