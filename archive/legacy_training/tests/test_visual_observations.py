"""No-network verification that rendered observations reach the model command."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from PIL import Image
from aspire_so100.agent_loop import ask_llm, select_visual_observations


class VisualObservationTests(unittest.TestCase):
    def test_cli_attaches_image_and_records_manifest(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            photo = root / 'observation.png'
            Image.new('RGB', (12,12), 'red').save(photo)
            def fake_process(command, **kwargs):
                self.assertEqual(command[command.index('--image') + 1], str(photo))
                response = Path(command[command.index('--output-last-message') + 1])
                response.write_text(json.dumps({'code':'def run(api): pass'}))
            with patch('aspire_so100.agent_loop.resolve_codex', return_value=Path('/mock/codex')), \
                 patch('aspire_so100.agent_loop.subprocess.run', side_effect=fake_process) as process:
                response = ask_llm('look at the image', root/'proposal', ('code',), image_paths=[str(photo)])
            self.assertIn('code', response)
            self.assertEqual(process.call_count, 1)
            manifest = json.loads((root/'proposal/images.json').read_text())
            self.assertEqual(manifest['attachments'], [str(photo)])

    def test_best_and_latest_images_bounded_and_labeled(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            entries = []
            for i in range(3):
                directory = root/f'round_{i:03d}'
                directory.mkdir()
                images = {}
                for phase in ('initial','outcome'):
                    path = directory/f'{phase}.png'
                    Image.new('RGB',(2,2)).save(path)
                    images[phase] = str(path)
                entries.append({'candidate':str(directory/'candidate.py'), 'successes':0,
                                'mean_score': .5 if i == 0 else 0,
                                'results':[{'seed':51,'visual_observations':images}]})
            paths, labels = select_visual_observations(entries)
            self.assertEqual(len(paths), 4)
            self.assertEqual([l['policy_round'] for l in labels], ['round_000']*2+['round_002']*2)
            self.assertEqual([l['phase'] for l in labels], ['initial','outcome']*2)
            self.assertEqual(len(select_visual_observations(entries[:1])[0]), 2)
            self.assertEqual(select_visual_observations([]), ([], []))
