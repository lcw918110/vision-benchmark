"""无需 GPU 和外部数据的评测协议回归测试。"""
import copy
import json
import argparse
import tempfile
import unittest
from pathlib import Path

import benchmark_v1 as b


class BenchmarkTests(unittest.TestCase):
    def record(self):
        return {'schema': b.SCHEMA, 'status': 'success',
                'model': {'weights_sha256': 'weight'}, 'metrics': {'mAP50': 0.5, 'mAP50_95': 0.3},
                'dataset': {'sha256': 'a', 'images': 1, 'names': {'0': 'person'}},
                'protocol': {'task': 'detect', 'adapter': 'ultralytics-yolo', 'imgsz': 640,
                             'batch': 1, 'precision': 'fp32', 'confidence': 0.001,
                             'nms_iou': 0.7, 'max_det': 300, 'split': 'val', 'device': '0',
                             'metric_version': 'ultralytics-box-val-v1'},
                'environment': {'python': '3', 'ultralytics': '8', 'torch': '2', 'cuda': '12', 'gpu': 'x'},
                'provenance': {'evaluator_sha256': 'a'}}

    def test_compatible_and_fail_closed(self):
        a = self.record()
        self.assertEqual([], b.comparable(a, copy.deepcopy(a)))
        for section, key in [('dataset', 'sha256'), ('protocol', 'task'), ('protocol', 'imgsz'),
                             ('environment', 'gpu'), ('provenance', 'evaluator_sha256')]:
            other = copy.deepcopy(a)
            other[section][key] = 'different'
            self.assertTrue(b.comparable(a, other), (section, key))
        other = copy.deepcopy(a)
        del other['protocol']['confidence']
        self.assertTrue(b.comparable(a, other))
        other = copy.deepcopy(a)
        other['status'] = 'failed'
        self.assertTrue(b.comparable(a, other))

    def test_comparison_saved_and_does_not_overwrite_input(self):
        with tempfile.TemporaryDirectory(dir='/output' if Path('/configs/benchmark_v1.py').exists() else Path(b.__file__).resolve().parent.parent) as tmp:
            root = Path(tmp)
            record = self.record()
            record['run_id'] = 'run1'
            first, second, output = (root / name for name in ('a.json', 'b.json', 'comparison.json'))
            first.write_text(json.dumps(record))
            record['run_id'] = 'run2'
            second.write_text(json.dumps(record))
            args = argparse.Namespace(results=[str(first), str(second)], out=str(output))
            self.assertEqual(0, b.compare(args))
            self.assertTrue(json.loads(output.read_text())['comparable'])
            args.out = str(first)
            with self.assertRaises(ValueError):
                b.compare(args)

    def test_malformed_comparison_fails_closed(self):
        a = self.record()
        bad = copy.deepcopy(a)
        bad['protocol'] = None
        self.assertTrue(b.comparable(a, bad))

    def test_output_boundary(self):
        with self.assertRaises(ValueError):
            b.assert_output_inside_project('/root/CoreYoloU/result.json')

    def test_nonfinite_and_missing_metrics_rejected(self):
        a = self.record()
        for value in (float('nan'), float('inf'), -1, 2, None, True):
            other = copy.deepcopy(a)
            other['metrics']['mAP50'] = value
            self.assertTrue(b.comparable(a, other))

    def test_names_strict(self):
        b.check_names(['a', 'b'], {0: 'a', 1: 'b'})
        with self.assertRaises(ValueError):
            b.check_names(['a', 'b'], {0: 'b', 1: 'a'})

    def test_dataset_hash_and_missing_label(self):
        with tempfile.TemporaryDirectory(dir='/output' if Path('/configs/benchmark_v1.py').exists() else Path(b.__file__).resolve().parent.parent) as tmp:
            root = Path(tmp)
            (root / 'images').mkdir()
            (root / 'labels').mkdir()
            (root / 'images' / 'a.jpg').write_bytes(b'image')
            with self.assertRaises(ValueError):
                b.fingerprint_dataset(root)
            label = root / 'labels' / 'a.txt'
            label.write_text('')
            first = b.fingerprint_dataset(root)
            self.assertEqual(1, first['images'])
            label.write_text('0 0.5 0.5 0.2 0.2')
            self.assertNotEqual(first['sha256'], b.fingerprint_dataset(root)['sha256'])


if __name__ == '__main__':
    unittest.main()
