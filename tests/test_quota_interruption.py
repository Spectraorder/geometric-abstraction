"""Only pre-response quota failures can be archived for an infrastructure retry."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from archive_quota_interruption import quota_failure


class QuotaInterruptionTests(unittest.TestCase):
    def fixture(self, path, extra=(), message='You have hit your usage limit.'):
        events=[{'type':'error','message':message},{'type':'turn.failed','error':{'message':message}},*extra]
        (path/'model.jsonl').write_text('\n'.join(json.dumps(e) for e in events)+'\n')

    def test_pre_response_quota_failure_is_explicit(self):
        with TemporaryDirectory() as tmp:
            path=Path(tmp);self.fixture(path)
            self.assertEqual(len(quota_failure(path)),2)

    def test_completed_or_emitted_response_cannot_be_retried(self):
        for event in ({'type':'turn.completed'}, {'type':'item.completed','item':{'type':'agent_message','text':'candidate'}}):
            with TemporaryDirectory() as tmp:
                path=Path(tmp);self.fixture(path,[event])
                with self.assertRaisesRegex(ValueError,'completed or emitted'):quota_failure(path)

    def test_generated_candidate_cannot_be_archived(self):
        with TemporaryDirectory() as tmp:
            path=Path(tmp);self.fixture(path);(path/'candidate.py').write_text('def run(api): pass\n')
            with self.assertRaisesRegex(ValueError,'generated candidate'):quota_failure(path)

    def test_other_errors_require_a_separate_diagnosis(self):
        with TemporaryDirectory() as tmp:
            path=Path(tmp);self.fixture(path,message='Invalid generated response')
            with self.assertRaisesRegex(ValueError,'Only explicit'):quota_failure(path)


if __name__=='__main__':unittest.main()
