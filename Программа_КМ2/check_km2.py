"""Проверка КМ2 без запуска обезличивания и без загрузки BERTScore."""
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parent

def main():
    if sys.version_info<(3,10):
        raise SystemExit('Требуется Python 3.10 или новее')
    steps=[['-m','unittest','discover','-s','tests','-v'],
           ['-m','pdn_course.build_dataset'],
           ['-m','pdn_course.evaluate','--dataset','data/test_dataset.jsonl','--output','results/recognition_metrics.json'],
           ['-m','pdn_course.evaluate','--dataset','data/synthetic.jsonl','--output','results/synthetic_metrics.json']]
    for step in steps:
        print('\nЗапуск:',sys.executable,*step,flush=True)
        subprocess.run([sys.executable,*step],cwd=ROOT,check=True)
    from pdn_course.evaluate import load_dataset,validate_dataset
    import json
    results=json.loads((ROOT/'results/recognition_metrics.json').read_text(encoding='utf8'))
    if results['texts']<100 or min(results['entity_instances'].values())<50:
        raise SystemExit('Не выполнены требования к набору данных')
    metrics=results['bio_excluding_O']
    if metrics['macro_f1']<.8 or metrics['weighted_f1']<.85:
        raise SystemExit('Качество ниже порога задания')
    print('\nКМ2: тесты пройдены, объём данных и пороги F1 соблюдены.')
if __name__=='__main__':main()
