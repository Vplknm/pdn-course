import copy
import unittest
from pathlib import Path
from pdn_course.evaluate import validate_dataset,class_scores,average,tags_for,load_dataset
from pdn_course.recognizer import Entity
from pdn_course.validators import snils_valid
from pdn_course.build_dataset import build,ROOT

class EvaluationTests(unittest.TestCase):
    def test_metrics_with_known_confusion(self):
        result=class_scores(['B-NAME','B-NAME','O'],['B-NAME','O','B-NAME'],['B-NAME'])
        self.assertEqual(result['B-NAME']['tp'],1)
        self.assertEqual(result['B-NAME']['fp'],1)
        self.assertEqual(result['B-NAME']['fn'],1)
        self.assertEqual(average(result)['weighted_f1'],.5)

    def test_bio_distinguishes_boundary(self):
        gold=tags_for('Иван Петров',[Entity(0,11,'NAME','Иван Петров')])
        pred=tags_for('Иван Петров',[Entity(0,4,'NAME','Иван'),Entity(5,11,'NAME','Петров')])
        self.assertEqual(gold,['B-NAME','I-NAME'])
        self.assertEqual(pred,['B-NAME','B-NAME'])

    def test_bad_gold_offsets_fail(self):
        with self.assertRaises(ValueError):
            validate_dataset([{'id':1,'text':'Иван','entities':[dict(start=0,end=3,label='NAME',text='Иван')]}])

    def test_corpus_is_reproducible_and_minimums_hold(self):
        built,_=build()
        self.assertEqual(built,load_dataset(ROOT/'data/test_dataset.jsonl'))
        validate_dataset(built)
        self.assertEqual(len(built),100)
        from collections import Counter
        counts=Counter(e['label'] for r in built for e in r['entities'])
        self.assertEqual(len(counts),16)
        self.assertGreaterEqual(min(counts.values()),50)

    def test_snils_known_sum_below_100(self):
        self.assertTrue(snils_valid('112-233-445 95'))
        self.assertFalse(snils_valid('112-233-445 00'))
if __name__=='__main__':unittest.main()
