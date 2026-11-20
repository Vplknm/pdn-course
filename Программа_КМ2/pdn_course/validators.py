"""Проверка контрольных разрядов и генерация валидных идентификаторов."""

from __future__ import annotations

import random


def digits(value: str) -> str:
    return "".join(ch for ch in value if ch.isdigit())


def luhn_valid(value: str) -> bool:
    number = digits(value)
    if len(number) < 12:
        return False
    total = 0
    parity = len(number) % 2
    for index, char in enumerate(number):
        digit = int(char)
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def luhn_check_digit(prefix: str) -> str:
    for digit in "0123456789":
        if luhn_valid(prefix + digit):
            return digit
    raise AssertionError("контрольная цифра Luhn не найдена")


def inn_valid(value: str) -> bool:
    number = digits(value)
    if len(number) == 10:
        coeffs = (2, 4, 10, 3, 5, 9, 4, 6, 8)
        return int(number[-1]) == sum(int(x) * k for x, k in zip(number, coeffs)) % 11 % 10
    if len(number) == 12:
        c11 = (7, 2, 4, 10, 3, 5, 9, 4, 6, 8)
        c12 = (3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8)
        d11 = sum(int(x) * k for x, k in zip(number[:10], c11)) % 11 % 10
        d12 = sum(int(x) * k for x, k in zip(number[:11], c12)) % 11 % 10
        return number[-2:] == f"{d11}{d12}"
    return False


def make_inn(prefix4: str, rng: random.Random, length: int = 12) -> str:
    prefix4 = digits(prefix4).ljust(4, "0")[:4]
    if length == 10:
        base = prefix4 + "".join(str(rng.randrange(10)) for _ in range(5))
        coeffs = (2, 4, 10, 3, 5, 9, 4, 6, 8)
        return base + str(sum(int(x) * k for x, k in zip(base, coeffs)) % 11 % 10)
    base = prefix4 + "".join(str(rng.randrange(10)) for _ in range(6))
    c11 = (7, 2, 4, 10, 3, 5, 9, 4, 6, 8)
    d11 = sum(int(x) * k for x, k in zip(base, c11)) % 11 % 10
    first11 = base + str(d11)
    c12 = (3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8)
    d12 = sum(int(x) * k for x, k in zip(first11, c12)) % 11 % 10
    return first11 + str(d12)


def snils_valid(value: str) -> bool:
    number = digits(value)
    if len(number) != 11:
        return False
    base, check = number[:9], int(number[9:])
    total = sum(int(d) * weight for d, weight in zip(base, range(9, 0, -1)))
    calculated = total if total < 100 else (0 if total in (100, 101) else total % 101)
    if calculated == 100:
        calculated = 0
    return check == calculated


def make_snils(rng: random.Random) -> str:
    while True:
        base = "".join(str(rng.randrange(10)) for _ in range(9))
        total = sum(int(d) * weight for d, weight in zip(base, range(9, 0, -1)))
        check = total if total < 100 else (0 if total in (100, 101) else total % 101)
        if check == 100:
            check = 0
        value = base + f"{check:02d}"
        if snils_valid(value):
            return value


def ogrnip_valid(value: str) -> bool:
    number = digits(value)
    return len(number) == 15 and int(number[-1]) == int(number[:14]) % 13 % 10


def make_ogrnip(prefix5: str, rng: random.Random) -> str:
    base = digits(prefix5).ljust(5, "0")[:5] + "".join(str(rng.randrange(10)) for _ in range(9))
    return base + str(int(base) % 13 % 10)
