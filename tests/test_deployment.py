"""Failure isolation, queue limits and restart semantics without loading weights."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pyannote_dual.service import JobService
from pyannote_dual.pipeline import Community1Diarizer, DiarizationBundle, Segment
from pyannote_dual.batch import run_batch


class FakeDiarizer:
    def process(self, audio, **kwargs):
        if Path(audio).read_bytes() == b'bad':
            raise ValueError('bad audio')
        return DiarizationBundle(Path(audio), 1.0, [Segment(0, 1, 'SPEAKER_00')],
                                 [Segment(0, 1, 'SPEAKER_00')], 'test', 'cpu')


class DeploymentTest(unittest.TestCase):
    def test_interrupted_upload_recovers_as_failed_and_queue_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = JobService(FakeDiarizer(), tmp, max_pending=1)
            job_id, audio = service.reserve('a.wav', {})
            with self.assertRaises(OverflowError):
                service.reserve('b.wav', {})
            service.close()
            restored = JobService(FakeDiarizer(), tmp, max_pending=1)
            self.assertEqual(restored.snapshot(job_id)['status'], 'failed')
            restored.reserve('new.wav', {})
            restored.close()

    def test_failed_job_does_not_stop_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = JobService(FakeDiarizer(), tmp)
            ids = []
            for data in (b'bad', b'good'):
                job_id, path = service.reserve('a.wav', {})
                path.write_bytes(data)
                service.enqueue(job_id)
                ids.append(job_id)
            service.queue.join()
            self.assertEqual([service.snapshot(i)['status'] for i in ids], ['failed', 'succeeded'])
            service.close()

    def test_batch_preserves_same_stem_extensions_and_nested_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, target = Path(tmp) / 'in', Path(tmp) / 'out'
            (source / 'nested').mkdir(parents=True)
            for name in ('same.wav', 'same.mp3', 'nested/same.wav'):
                (source / name).write_bytes(b'good')
            records = run_batch(FakeDiarizer(), source, target)
            self.assertEqual(len(records), 3)
            self.assertEqual(len({r['summary']['output_dir'] for r in records}), 3)
            with self.assertRaises(ValueError):
                run_batch(FakeDiarizer(), source, source / 'outputs')

    def test_invalid_speaker_arguments_rejected_before_pipeline(self):
        diarizer = Community1Diarizer(None, 'test', 'cpu')
        for kwargs in ({'num_speakers': 0}, {'num_speakers': 1, 'max_speakers': 2},
                       {'min_speakers': 3, 'max_speakers': 2}, {'num_speakers': 1.5}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                diarizer.process(__file__, **kwargs)


if __name__ == '__main__':
    unittest.main()
