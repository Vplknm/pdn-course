"""Лабораторная работа по распознаванию персональных данных."""

from .person_ner import PersonModelRecognizer
from .recognizer import Entity, Recognizer

__all__ = ["Entity", "PersonModelRecognizer", "Recognizer"]
