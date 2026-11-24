import json
from pathlib import Path
import tempfile
import unittest

from restaurant_sentiment.reporting import summarize


class ReportingTests(unittest.TestCase):
    def test_multiple_seeds_and_different_test_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = []
            for seed, value in ((42, 0.5), (43, 0.75)):
                p = root / f'{seed}.json'
                p.write_text(json.dumps(dict(method='test', seed=seed,
                    reference_sha256='abc', data_fingerprint='def', experiment_signature='same',
                    macro_f1=value, macro_accuracy=value,
                    joint_exact_match=value, json_valid_rate=1.0)))
                paths.append(p)
            report = summarize(paths, root/'summary.json')
            self.assertEqual(report['results'][0]['macro_f1']['mean'], 0.625)
            self.assertIsNotNone(report['results'][0]['macro_f1']['sample_std'])
            data = json.loads(paths[-1].read_text())
            data['reference_sha256'] = 'other'
            paths[-1].write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                summarize(paths, root/'summary.json')

    def test_rejects_config_mixing_and_nonfinite_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = []
            for seed in (42, 43):
                p = root / f'{seed}.json'
                p.write_text(json.dumps(dict(method='lora', seed=seed,
                    reference_sha256='abc', data_fingerprint='def', experiment_signature=str(seed),
                    macro_f1=.5, macro_accuracy=.5, joint_exact_match=.5, json_valid_rate=1.)))
                paths.append(p)
            with self.assertRaises(ValueError):
                summarize(paths, root/'summary.json')
            data = json.loads(paths[0].read_text())
            data['macro_f1'] = float('nan')
            paths[0].write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                summarize([paths[0]], root/'summary.json')

    def test_single_seed_does_not_invent_uncertainty(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p = root/'metrics.json'
            p.write_text(json.dumps(dict(method='test', seed=42,
                reference_sha256='abc', data_fingerprint='def', macro_f1=.5,
                macro_accuracy=.5, joint_exact_match=.5, json_valid_rate=1.)))
            report = summarize([p], root/'summary.json')
            self.assertIsNone(report['results'][0]['macro_f1']['sample_std'])
