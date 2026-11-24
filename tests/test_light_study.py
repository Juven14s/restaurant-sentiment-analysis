import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from restaurant_sentiment.analysis import diagnose, single_fence_prediction, format_ablation
from restaurant_sentiment.baselines import run_baseline
from restaurant_sentiment.benchmark import freeze_challenge, freeze_validation_probe, category_scores
from restaurant_sentiment.data import ASPECTS, LABELS, prepare_data, load_jsonl


class LightStudyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        source = self.root/'source.tsv'
        with source.open('w',newline='',encoding='utf-8') as f:
            writer = csv.writer(f,delimiter='\t')
            writer.writerow(['Avis',*ASPECTS])
            for i in range(30):
                writer.writerow([f'Original test review number {i} for deterministic selection during unit checks',*[LABELS[i%4]]*3])
        self.data = self.root/'data'
        prepare_data([source],self.data)

    def test_validation_baseline_never_needs_test(self):
        (self.data/'test.jsonl').unlink()
        result = run_baseline(self.data,self.root/'baseline',evaluation_split='validation')
        self.assertEqual(result['n_samples'],3)
        self.assertEqual(result['evaluation_split'],'validation')
        self.assertNotIn('test',result['data_files_sha256'])

    def test_paraphrases_select_training_sources_only(self):
        script=Path(__file__).resolve().parents[1]/'scripts'/'audit_paraphrases.py'
        spec=importlib.util.spec_from_file_location('audit_paraphrases',script)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        (self.data/'test.jsonl').unlink()
        (self.data/'validation.jsonl').unlink()
        sources,_=module.select_sources(self.data/'train.jsonl')
        self.assertEqual(len(sources),12)
        self.assertEqual(len({r['id'] for r in sources}),12)
        for label in LABELS:
            self.assertEqual(sum(r['labels']['Cuisine']==label for r in sources),3)
        with self.assertRaises(ValueError):
            module.select_sources(self.data/'validation.jsonl')

    def test_probe_is_deterministic_frozen_and_label_independent(self):
        a,b = self.root/'a',self.root/'b'
        freeze_validation_probe(self.data/'validation.jsonl',a,size=2,seed=7)
        freeze_validation_probe(self.data/'validation.jsonl',a,size=2,seed=7)
        selected = {r['id'] for r in load_jsonl(a/'probe.jsonl')}
        rows=load_jsonl(self.data/'validation.jsonl')
        for row in rows:
            row['labels']={a:LABELS[(LABELS.index(row['labels'][a])+1)%4] for a in ASPECTS}
        path=self.data/'validation.jsonl'
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        manifest=json.loads((self.data/'manifest.json').read_text())
        manifest['files']['validation']['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        (self.data/'manifest.json').write_text(json.dumps(manifest))
        freeze_validation_probe(path,b,size=2,seed=7)
        self.assertEqual(selected,{r['id'] for r in load_jsonl(b/'probe.jsonl')})
        with self.assertRaises(FileExistsError):
            freeze_validation_probe(path,a,size=2,seed=7)
        with self.assertRaises(ValueError):
            freeze_validation_probe(self.data/'test.jsonl',self.root/'bad',size=1)

    def test_external_challenge_scoring_and_category_alignment(self):
        cases=[dict(case_id='one',category='absence',text='Le bâtiment est bleu.',
                    labels={a:'NE' for a in ASPECTS},rationale='No opinion.')]
        source=self.root/'cases.json';source.write_text(json.dumps(cases))
        out=self.root/'challenge'
        freeze_challenge(source,out)
        report=run_baseline(self.data,self.root/'baseline',evaluation_path=out/'challenge.jsonl')
        self.assertEqual(report['evaluation_split'],'external')
        self.assertEqual(report['n_samples'],1)
        slices=category_scores(out/'challenge.jsonl',self.root/'baseline'/'predictions.jsonl',out/'annotations.json')
        self.assertEqual(slices['absence']['n_samples'],1)
        annotations=json.loads((out/'annotations.json').read_text());annotations.append(annotations[0])
        (out/'annotations.json').write_text(json.dumps(annotations))
        with self.assertRaises(ValueError):
            category_scores(out/'challenge.jsonl',self.root/'baseline'/'predictions.jsonl',out/'annotations.json')

    def test_fence_normalization_does_not_repair_labels_or_hide_text(self):
        valid=json.dumps({a:'NE' for a in ASPECTS})
        self.assertEqual(single_fence_prediction('```json\n'+valid+'\n```'),{a:'NE' for a in ASPECTS})
        self.assertIsNone(single_fence_prediction('Explanation\n```json\n'+valid+'\n```'))
        self.assertIsNone(single_fence_prediction('```json\n'+valid+'\n```\nOther text'))
        self.assertIsNone(single_fence_prediction('```json\n'+valid.replace('NE','Excellent')+'\n```'))

    def test_ablation_keeps_invalid_rows_in_denominator(self):
        references=load_jsonl(self.data/'validation.jsonl')
        predictions=[]
        for i,row in enumerate(references):
            raw='```json\n'+json.dumps(row['labels'])+'\n```' if i else 'not JSON'
            predictions.append(dict(id=row['id'],prediction=None,raw_output=raw))
        path=self.root/'predictions.jsonl';path.write_text(''.join(json.dumps(p)+'\n' for p in predictions))
        report=format_ablation(self.data/'validation.jsonl',path,self.root/'ablation')
        self.assertEqual(report['strict']['macro_accuracy'],0)
        self.assertAlmostEqual(report['single_fence']['macro_accuracy'],2/3)
        self.assertEqual(report['recovered_by_single_fence'],2)

    def test_diagnostics_preserve_reviewed_worksheet_and_reject_test(self):
        rows=load_jsonl(self.data/'validation.jsonl')
        path=self.root/'predictions.jsonl'
        path.write_text(''.join(json.dumps(dict(id=r['id'],prediction=None))+'\n' for r in rows))
        report=diagnose(self.data/'validation.jsonl',path,self.root/'analysis',sample_size=2)
        self.assertEqual(report['error_reviews'],3)
        self.assertEqual(report['review_sample_size'],2)
        worksheet=self.root/'analysis'/'error_review.csv'
        with worksheet.open() as f: records=list(csv.DictReader(f))
        records[0]['review_category']='reviewed'
        with worksheet.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=records[0]);w.writeheader();w.writerows(records)
        before=worksheet.read_bytes()
        with self.assertRaises(FileExistsError):
            diagnose(self.data/'validation.jsonl',path,self.root/'analysis')
        self.assertEqual(before,worksheet.read_bytes())
        with self.assertRaises(ValueError):
            diagnose(self.data/'test.jsonl',path,self.root/'test-analysis')

    def test_published_challenge_is_frozen_original_and_complete(self):
        directory=Path(__file__).resolve().parents[1]/'benchmarks'/'fr_aspect_challenge_v1'
        rows=load_jsonl(directory/'challenge.jsonl')
        self.assertEqual(len(rows),80)
        with tempfile.TemporaryDirectory() as tmp:
            freeze_challenge(directory/'cases.json',Path(tmp))
            for name in ('challenge.jsonl','annotations.json','manifest.json'):
                self.assertEqual((directory/name).read_bytes(),(Path(tmp)/name).read_bytes())
