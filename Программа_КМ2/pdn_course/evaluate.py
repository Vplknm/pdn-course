"""Воспроизводимая токенная BIO-оценка и строгая оценка границ сущностей."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
from .recognizer import Entity, Recognizer, WORD_RE

REQUIRED = {'SURNAME','NAME','INN','PASSPORT','BANK_CARD','OGRNIP','IP','EMAIL','BIRTH_CERT','KLADR'}

def load_dataset(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding='utf8').splitlines() if line.strip()]

def validate_dataset(dataset: list[dict]) -> None:
    if not dataset:
        raise ValueError('Пустой датасет')
    ids = set()
    for record in dataset:
        if record['id'] in ids:
            raise ValueError('Повтор id')
        ids.add(record['id'])
        text = record['text']
        entities = sorted(record['entities'], key=lambda e: e['start'])
        for entity in entities:
            if not 0 <= entity['start'] < entity['end'] <= len(text):
                raise ValueError('Границы вне текста')
            if text[entity['start']:entity['end']] != entity['text']:
                raise ValueError('Сущность не совпадает с текстом')
        for a,b in zip(entities, entities[1:]):
            if a['end'] > b['start']:
                raise ValueError('Пересечение эталонных сущностей')

def tags_for(text: str, entities: list[Entity]) -> list[str]:
    tags=[]
    for token in WORD_RE.finditer(text):
        tag='O'
        for entity in entities:
            if token.start() >= entity.start and token.end() <= entity.end:
                tag=('B-' if token.start()==entity.start else 'I-')+entity.label
                break
        tags.append(tag)
    return tags

def scores(tp: int, fp: int, fn: int) -> dict:
    precision=tp/(tp+fp) if tp+fp else 0.0
    recall=tp/(tp+fn) if tp+fn else 0.0
    return dict(tp=tp,fp=fp,fn=fn,support=tp+fn,precision=precision,recall=recall,
                f1=2*precision*recall/(precision+recall) if precision+recall else 0.0)

def class_scores(gold:list[str], pred:list[str], labels:list[str]) -> dict:
    return {label:scores(sum(a==label and b==label for a,b in zip(gold,pred)),
                         sum(a!=label and b==label for a,b in zip(gold,pred)),
                         sum(a==label and b!=label for a,b in zip(gold,pred))) for label in labels}

def average(per_class:dict) -> dict:
    rows=list(per_class.values())
    support=sum(row['support'] for row in rows)
    micro=scores(sum(r['tp'] for r in rows),sum(r['fp'] for r in rows),sum(r['fn'] for r in rows))
    return {'macro_f1':sum(r['f1'] for r in rows)/len(rows) if rows else 0.0,
            'weighted_f1':sum(r['f1']*r['support'] for r in rows)/support if support else 0.0,
            'micro':micro}

def calculate(dataset:list[dict], recognizer:Recognizer|None=None) -> tuple[dict,list]:
    validate_dataset(dataset)
    recognizer=recognizer or Recognizer()
    gold_all=[];pred_all=[];counts=Counter();gold_spans=Counter();pred_spans=Counter();errors=[]
    for record in dataset:
        text=record['text'];gold=[Entity(**e) for e in record['entities']]
        pred=recognizer.recognize(text)
        gold_all.extend(tags_for(text,gold));pred_all.extend(tags_for(text,pred))
        counts.update(e.label for e in gold)
        gs={(e.label,e.start,e.end) for e in gold};ps={(e.label,e.start,e.end) for e in pred}
        for label in counts.keys() | {e.label for e in pred}:
            gold_spans[label]+=sum(s[0]==label for s in gs)
            pred_spans[label]+=sum(s[0]==label for s in ps)
        for kind,items in [('missed',gs-ps),('extra_or_wrong',ps-gs)]:
            for label,start,end in sorted(items):
                errors.append({'record_id':record['id'],'kind':kind,'label':label,'start':start,'end':end,
                               'text':text[start:end],'context':text[max(0,start-45):min(len(text),end+45)]})
    labels=sorted((set(gold_all)|set(pred_all))-{'O'})
    per_bio=class_scores(gold_all,pred_all,labels)
    collapse=lambda tags:[t[2:] if t!='O' else 'O' for t in tags]
    ga,pa=collapse(gold_all),collapse(pred_all)
    attrs=sorted((set(ga)|set(pa))-{'O'})
    per_attribute=class_scores(ga,pa,attrs)
    exact={}
    # Strict span metric: a match requires label AND both character boundaries.
    fn=Counter(e['label'] for e in errors if e['kind']=='missed')
    fp=Counter(e['label'] for e in errors if e['kind']=='extra_or_wrong')
    for label in attrs:
        exact[label]=scores(gold_spans[label]-fn[label],fp[label],fn[label])
    result={'texts':len(dataset),'tokens':len(gold_all),'entity_instances':dict(sorted(counts.items())),
            'accuracy_including_O':sum(a==b for a,b in zip(gold_all,pred_all))/len(gold_all),
            'bio_excluding_O':average(per_bio),'per_bio_class':per_bio,
            'attributes_excluding_O':average(per_attribute),'per_attribute':per_attribute,
            'required_variant4_excluding_O':average({k:v for k,v in per_attribute.items() if k in REQUIRED}),
            'exact_span':{'average':average(exact),'per_attribute':exact},
            'definitions':{'main':'BIO token classification; macro/weighted averaged over B/I classes, O excluded',
                           'attribute':'B/I collapsed only for per-attribute table; boundary errors remain in main BIO metric',
                           'required':'10 attributes of even variant; prediction of other labels treated as other for this diagnostic',
                           'no_finetuning':True}}
    return result,errors

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',type=Path,default=Path('data/test_dataset.jsonl'))
    parser.add_argument('--output',type=Path,default=Path('results/recognition_metrics.json'))
    args=parser.parse_args()
    result,errors=calculate(load_dataset(args.dataset))
    result['dataset_sha256']=hashlib.sha256(args.dataset.read_bytes()).hexdigest()
    result['environment']={'python':platform.python_version(),**{p:version(p) for p in ['natasha','slovnet','navec','pymorphy3','numpy']}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    error_file=args.output.with_name(args.output.stem+'_errors.json')
    error_file.write_text(json.dumps(errors,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:result[k] for k in ['texts','tokens','bio_excluding_O','attributes_excluding_O','required_variant4_excluding_O']},ensure_ascii=False,indent=2))
    print('Полные результаты:',args.output,'Ошибки:',error_file)
if __name__=='__main__':main()
