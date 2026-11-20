"""Демонстрация распознавания для КМ2: свой текст, файл или запись корпуса."""
import argparse
import json
from pathlib import Path
import sys
from .recognizer import Recognizer

ROOT=Path(__file__).resolve().parents[1]
DEFAULT='Документы переданы Иванову Ивану Петровичу, ИНН 7707083893, e-mail ivan@example.ru.'

def main():
    parser=argparse.ArgumentParser(description='Распознавание ПДн для КМ2')
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--text')
    group.add_argument('--file',type=Path)
    group.add_argument('--dataset-record',type=int)
    parser.add_argument('--bio',action='store_true')
    args=parser.parse_args()
    if args.file:
        text=args.file.read_text(encoding='utf8')
    elif args.dataset_record is not None:
        rows=[json.loads(l) for l in (ROOT/'data/test_dataset.jsonl').read_text(encoding='utf8').splitlines()]
        if not 1<=args.dataset_record<=len(rows):parser.error('Запись должна быть от 1 до 100')
        text=rows[args.dataset_record-1]['text']
    elif args.text is not None:text=args.text
    elif sys.stdin.isatty():text=input('Введите текст или Enter для примера: ').strip() or DEFAULT
    else:text=sys.stdin.read().strip() or DEFAULT
    model=Recognizer();entities=model.recognize(text)
    print('Исходный текст:\n'+text+'\n\nНайденные атрибуты:')
    for e in entities:print(e.as_list(),repr(e.text))
    if args.bio:
        print('\nBIO-разметка:')
        for token,tag in model.bio_tags(text,entities):print(f'{token:24} {tag}')
if __name__=='__main__':main()
