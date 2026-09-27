"""Verify retained curves, thresholds and the hardware-based concurrency decision."""
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import evaluate_geometry_learning_curves as curves


def training_curve(successes):
    return [{'revision': revision, 'results': [{'seed': seed, 'success': seed in selected} for seed in range(5)]}
            for revision, selected in enumerate(successes)]


class CurveTests(unittest.TestCase):
    def test_retention_is_training_only_even_when_test_scores_improve(self):
        training = training_curve([set(), {0,1}, {1,2,3}, {0,1,4}])
        result = curves.curves(training,[0,.6,.95,.7],[0,100,250,400],[.6,.8])
        self.assertEqual(result['retained_revisions'],[0,1,1,3])
        self.assertEqual(result['retained'],[0,.6,.6,.7])
        self.assertEqual(result['first_raw_threshold']['0.8'],2)
        self.assertIsNone(result['first_retained_threshold']['0.8'])
        self.assertEqual(result['cumulative_generation_tokens'],[0,100,250,400])

    def test_hardware_gate_keeps_16_only_when_all_samples_fit(self):
        healthy={'memory':{'MemAvailable':20*1024**3},'gpu_query_returncode':0,'gpu_csv':'RTX 3090, 24576, 6000, 70'}
        self.assertEqual(curves.pilot_workers([healthy],16),16)
        low_ram=dict(healthy,memory={'MemAvailable':5*1024**3})
        high_gpu=dict(healthy,gpu_csv='RTX 3090, 24576, 24000, 100')
        self.assertEqual(curves.pilot_workers([healthy,low_ram],16),8)
        self.assertEqual(curves.pilot_workers([high_gpu],16),8)
        self.assertEqual(curves.pilot_workers([],16),8)

    def test_plot_exports_all_replicas_and_both_curves(self):
        training=training_curve([set(),{0},{0,1},{0,1,2}])
        row=curves.curves(training,[0,.4,.6,.7],[0,100,250,400],[.8])
        report={'replicas':{'replica_01':{'basic':row,'symbolic':row}},
                'mean_curves':{'basic':row,'symbolic':row}}
        with TemporaryDirectory() as temp:
            output=Path(temp)
            curves.plot(output,report)
            self.assertTrue((output/'learning_curves.png').is_file())
            self.assertTrue((output/'learning_curves.pdf').is_file())
            self.assertEqual(len((output/'learning_curves.csv').read_text().splitlines()),9)


if __name__ == '__main__':
    unittest.main()
