"""Восстановление корпуса из включённых исходных .txt/.spans без сети."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from .generate_dataset import generate_dataset

ROOT=Path(__file__).resolve().parents[1]

def build(root:Path=ROOT):
    metadata=json.loads((root/'data/provenance.json').read_text(encoding='utf8'))
    synthetic=generate_dataset(120,metadata['augmentation_seed'])
    source=root/'data/source/factRuEval2016'
    records=[]
    for item,extra in zip(metadata['records'],synthetic):
        path=source/item['file'];spans=path.with_suffix('.spans')
        for p,key in [(path,'text_sha256'),(spans,'spans_sha256')]:
            if hashlib.sha256(p.read_bytes()).hexdigest()!=item[key]:
                raise ValueError(f'Нарушена целостность исходника: {p.name}')
        text=path.read_text(encoding='utf8');entities={}
        for line in spans.read_text(encoding='utf8').splitlines():
            columns=line.split()
            if len(columns)<4 or columns[1] not in metadata['mapping']:continue
            label=metadata['mapping'][columns[1]];start=int(columns[2]);end=start+int(columns[3])
            entities[start,end,label]=dict(start=start,end=end,label=label,text=text[start:end])
        gold=sorted(entities.values(),key=lambda e:(e['start'],e['end']))
        sep='\n\nСинтетическая карточка для проверки реквизитов. '
        offset=len(text)+len(sep)
        gold.extend(dict(e,start=e['start']+offset,end=e['end']+offset) for e in extra['entities'])
        records.append(dict(id=item['id'],source=f"factRuEval2016/testset/{path.name}",
                            augmentation='synthetic profile; no relation to people in article',
                            original_length=len(text),text=text+sep+extra['text'],entities=gold))
    return records,synthetic

def main():
    records,synthetic=build()
    for name,rows in [('test_dataset.jsonl',records),('synthetic.jsonl',synthetic)]:
        (ROOT/'data'/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf8')
        print(name,len(rows),'текстов')
if __name__=='__main__':main()
