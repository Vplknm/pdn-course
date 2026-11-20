"""Генерация размеченного контрольного набора для лабораторной работы."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pymorphy3

from .recognizer import Entity
from .validators import luhn_check_digit, make_inn, make_ogrnip, make_snils


MALE_NAMES = ["Александр", "Алексей", "Андрей", "Артём", "Виктор", "Дмитрий", "Иван", "Кирилл", "Максим", "Михаил", "Николай", "Павел", "Пётр", "Роман", "Сергей"]
FEMALE_NAMES = ["Александра", "Алёна", "Анастасия", "Анна", "Валерия", "Виктория", "Дарья", "Евгения", "Екатерина", "Елена", "Ирина", "Ксения", "Мария", "Наталья", "Ольга", "Светлана", "София", "Татьяна", "Юлия"]
MALE_SURNAMES = ["Алексеев", "Андреев", "Васильев", "Волков", "Воробьёв", "Егоров", "Иванов", "Козлов", "Кузнецов", "Лебедев", "Морозов", "Николаев", "Новиков", "Орлов", "Павлов", "Петров", "Попов", "Семёнов", "Смирнов", "Соколов", "Фёдоров"]
FEMALE_SURNAMES = [name + "а" for name in MALE_SURNAMES]
MALE_PATRONYMICS = [
    "Александрович", "Алексеевич", "Андреевич", "Викторович", "Дмитриевич",
    "Евгеньевич", "Иванович", "Михайлович", "Николаевич", "Павлович", "Петрович", "Сергеевич",
]
FEMALE_PATRONYMICS = [
    "Александровна", "Алексеевна", "Андреевна", "Викторовна", "Дмитриевна",
    "Евгеньевна", "Ивановна", "Михайловна", "Николаевна", "Павловна", "Петровна", "Сергеевна",
]

MORPH = pymorphy3.MorphAnalyzer()
PERSON_GRAMMEMES = {"NAME": "Name", "SURNAME": "Surn", "PATRONYMIC": "Patr"}
PERSON_TEMPLATES = (
    ("nomn", ("SURNAME", "NAME", "PATRONYMIC"), "Заявитель ", "."),
    ("gent", ("SURNAME", "NAME", "PATRONYMIC"), "Получены документы от ", "."),
    ("datv", ("SURNAME", "NAME", "PATRONYMIC"), "Уведомление направлено ", "."),
    ("accs", ("SURNAME", "NAME", "PATRONYMIC"), "Система зарегистрировала ", "."),
    ("ablt", ("SURNAME", "NAME", "PATRONYMIC"), "Договор подписан ", "."),
    ("loct", ("SURNAME", "NAME", "PATRONYMIC"), "В заявлении о ", " обнаружена запись."),
    ("nomn", ("NAME", "PATRONYMIC", "SURNAME"), "Контактное лицо: ", "."),
)


def _add(parts: list[str], entities: list[Entity], text: str, label: str | None = None) -> None:
    start = sum(len(part) for part in parts)
    parts.append(text)
    if label:
        entities.append(Entity(start, start + len(text), label, text))


def _inflect_person(word: str, label: str, case: str, gender: str) -> str:
    """Склоняет часть ФИО, выбирая разбор нужного типа."""

    desired = PERSON_GRAMMEMES[label]
    parses = MORPH.parse(word)
    typed = [item for item in parses if desired in item.tag.grammemes]
    matching_gender = [item for item in typed if item.tag.gender == gender]
    parsed = matching_gender[0] if matching_gender else (typed[0] if typed else parses[0])
    inflected = parsed.inflect({case})
    value = inflected.word if inflected is not None else word
    return value.title()


def _format_card(number: str, style: int) -> str:
    if style == 0:
        return " ".join(number[pos:pos + 4] for pos in range(0, 16, 4))
    if style == 1:
        return "-".join(number[pos:pos + 4] for pos in range(0, 16, 4))
    return number


def _format_phone(index: int, rng: random.Random) -> str:
    operator = f"9{index % 10:02d}"
    tail = f"{rng.randrange(100, 1000):03d}{rng.randrange(10, 100):02d}{rng.randrange(10, 100):02d}"
    if index % 3 == 0:
        return f"+7 ({operator}) {tail[:3]}-{tail[3:5]}-{tail[5:]}"
    if index % 3 == 1:
        return f"8 {operator} {tail[:3]} {tail[3:5]} {tail[5:]}"
    return f"+7-{operator}-{tail[:3]}-{tail[3:5]}-{tail[5:]}"


def _add_person(
    parts: list[str],
    entities: list[Entity],
    values: dict[str, str],
    index: int,
    female: bool,
) -> None:
    case, order, prefix, suffix = PERSON_TEMPLATES[index % len(PERSON_TEMPLATES)]
    _add(parts, entities, prefix)
    for position, label in enumerate(order):
        if position:
            _add(parts, entities, " ")
        gender = "femn" if female else "masc"
        _add(parts, entities, _inflect_person(values[label], label, case, gender), label)
    _add(parts, entities, suffix)


def generate_record(index: int, rng: random.Random) -> dict[str, object]:
    female = index % 2 == 1
    name_pool = FEMALE_NAMES if female else MALE_NAMES
    surname_pool = FEMALE_SURNAMES if female else MALE_SURNAMES
    patronymic_pool = FEMALE_PATRONYMICS if female else MALE_PATRONYMICS
    name = name_pool[index % len(name_pool)]
    surname = surname_pool[index % len(surname_pool)]
    patronymic = patronymic_pool[index % len(patronymic_pool)]
    inn = make_inn(f"77{index % 100:02d}", rng, 12 if index % 3 else 10)
    passport = f"{45 + index % 5:02d} {10 + index % 90:02d} {rng.randrange(100000, 999999)}"
    card_base = "220220" + "".join(str(rng.randrange(10)) for _ in range(9))
    card = _format_card(card_base + luhn_check_digit(card_base), index % 3)
    snils_digits = make_snils(rng)
    if index % 2:
        snils = f"{snils_digits[:3]} {snils_digits[3:6]} {snils_digits[6:9]} {snils_digits[9:]}"
    else:
        snils = f"{snils_digits[:3]}-{snils_digits[3:6]}-{snils_digits[6:9]} {snils_digits[9:]}"
    ogrnip = make_ogrnip(f"3{20 + index % 6:02d}{77 + index % 10:02d}", rng)
    url_prefix = ("https://", "http://", "www.")[index % 3]
    url = f"{url_prefix}profile{index}.example.ru/account/{1000+index}"
    ip = f"10.{index % 250}.{(index*7) % 250}.{(index*13) % 250 or 1}"
    phone = _format_phone(index, rng)
    email = f"person.{index}+lab@mail{index % 7}.example"
    marriage = f"III-АБ № {rng.randrange(100000, 999999)}"
    birth = f"IV-ВГ № {rng.randrange(100000, 999999)}"
    cadastral = f"77:{index % 20:02d}:{rng.randrange(100000,999999)}:{rng.randrange(1,999)}"
    kladr = f"77{index % 1000:03d}{rng.randrange(0, 10**8):08d}"

    parts: list[str] = []
    entities: list[Entity] = []
    _add_person(
        parts,
        entities,
        {"SURNAME": surname, "NAME": name, "PATRONYMIC": patronymic},
        index,
        female,
    )

    layouts = (
        [
            (" ИНН ", None), (inn, "INN"), (", паспорт ", None), (passport, "PASSPORT"),
            (", карта ", None), (card, "BANK_CARD"), (", СНИЛС ", None), (snils, "SNILS"),
            (", ОГРНИП ", None), (ogrnip, "OGRNIP"), (". Профиль ", None), (url, "URL"),
            (", IP ", None), (ip, "IP"), (", телефон ", None), (phone, "PHONE"),
            (", e-mail ", None), (email, "EMAIL"),
        ],
        [
            (" Для связи: ", None), (phone, "PHONE"), (", ", None), (email, "EMAIL"),
            (". Реквизиты: ИНН ", None), (inn, "INN"), (", СНИЛС ", None), (snils, "SNILS"),
            (", паспорт ", None), (passport, "PASSPORT"), (", карта ", None), (card, "BANK_CARD"),
            (". Сетевые данные: ", None), (ip, "IP"), (", ", None), (url, "URL"),
            (". ОГРНИП ", None), (ogrnip, "OGRNIP"),
        ],
        [
            (" Электронная почта: ", None), (email, "EMAIL"), ("; телефон: ", None), (phone, "PHONE"),
            (". Карта ", None), (card, "BANK_CARD"), (", ИНН ", None), (inn, "INN"),
            (", ОГРНИП ", None), (ogrnip, "OGRNIP"), (", СНИЛС ", None), (snils, "SNILS"),
            (". Паспорт ", None), (passport, "PASSPORT"), ("; адрес профиля ", None), (url, "URL"),
            ("; IP-адрес ", None), (ip, "IP"),
        ],
    )
    for text, label in layouts[index % len(layouts)]:
        _add(parts, entities, text, label)

    for text, label in [
        (". Свидетельство о браке: ", None), (marriage, "MARRIAGE_CERT"),
        (", свидетельство о рождении: ", None), (birth, "BIRTH_CERT"),
        (". Кадастровый номер ", None), (cadastral, "CADASTRAL"),
        (", КЛАДР ", None), (kladr, "KLADR"), (".", None),
    ]:
        _add(parts, entities, text, label)
    return {"id": index + 1, "text": "".join(parts), "entities": [entity.__dict__ for entity in entities]}


def generate_dataset(count: int = 120, seed: int = 20261009) -> list[dict[str, object]]:
    rng = random.Random(seed)
    return [generate_record(index, rng) for index in range(count)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--output", type=Path, default=Path("data/test_dataset.jsonl"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for record in generate_dataset(args.count, args.seed):
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Сохранено текстов: {args.count}; файл: {args.output}")


if __name__ == "__main__":
    main()
