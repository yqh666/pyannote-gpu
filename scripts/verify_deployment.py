"""Real CUDA integration check: HTTP upload, failure recovery, output regression, batch."""
import json
import argparse
import shutil
import sys
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pyannote_dual import Community1Diarizer
from pyannote_dual.service import JobService, handler_for
from pyannote_dual.batch import run_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['torch', 'shared-frames', 'tensorrt'], default='torch')
    parser.add_argument('--engine', default=None)
    args = parser.parse_args()
    target = ROOT / 'outputs' / ('deployment_verify_' + args.backend + '_' + time.strftime('%Y%m%d_%H%M%S'))
    target.mkdir(parents=True)
    audio = ROOT / 'example/huawei_test.wav'
    diarizer = Community1Diarizer.from_pretrained(device='cuda', offline=True,
                                               backend=args.backend, engine_path=args.engine)
    service = JobService(diarizer, target / 'service')
    server = ThreadingHTTPServer(('127.0.0.1', 0), handler_for(service, 64 * 1024**2))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{server.server_port}'
    def call(path, data=None):
        request = Request(base + path, data=data, headers={'Content-Type': 'application/octet-stream'})
        try:
            with urlopen(request, timeout=30) as response:
                return response.status, response.read()
        except HTTPError as error:
            return error.code, error.read()
    def wait_job(job_id):
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            status, data = call('/jobs/' + job_id)
            assert status == 200
            job = json.loads(data)
            if job['status'] in {'succeeded', 'failed'}:
                return job
            time.sleep(0.5)
        raise TimeoutError(job_id)
    try:
        status, health = call('/health')
        health = json.loads(health)
        assert status == 200 and health['device'] == 'cuda' and health['backend'] == args.backend
        assert call('/jobs?num_speakers=0', b'x')[0] == 400
        assert call('/jobs?filename=../escape.wav', b'x')[0] == 400
        status, data = call('/jobs?filename=broken.wav', b'not an audio file')
        assert status == 202
        bad = wait_job(json.loads(data)['id'])
        assert bad['status'] == 'failed'
        status, data = call('/jobs?filename=huawei_test.wav&num_speakers=3', audio.read_bytes())
        assert status == 202
        job_id = json.loads(data)['id']
        good = wait_job(job_id)
        assert good['status'] == 'succeeded', good
        status, data = call('/jobs/' + job_id + '/result')
        result = json.loads(data)
        assert status == 200
        for mode in ('complete', 'non_conflict'):
            baseline = ROOT / 'outputs/verify_run' / mode
            expected = json.loads((baseline / 'segments.json').read_text(encoding='utf-8'))
            assert result[mode]['segments'] == expected['segments']
            for name in ('diarization.txt', 'diarization.rttm', 'segments.json'):
                status, body = call(f'/jobs/{job_id}/artifacts/{mode}/{name}')
                assert status == 200
                if name != 'segments.json':
                    # RTTM URI follows the uploaded input filename input.wav.
                    expected_bytes = (baseline / name).read_bytes()
                    if name.endswith('.rttm'):
                        expected_bytes = expected_bytes.replace(b'huawei_test', b'input')
                    assert body == expected_bytes
        assert result['non_conflict']['overlap_seconds'] == 0
        assert result['complete']['speech_seconds'] == result['non_conflict']['speech_seconds']
        assert call(f'/jobs/{job_id}/artifacts/../input.wav')[0] == 404
        assert call('/jobs/not-a-job')[0] == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
        service.close()
    # A real second entry point reuses the loaded model. A bad file must not end the batch.
    inputs = target / 'batch_inputs'
    (inputs / 'nested').mkdir(parents=True)
    (inputs / 'broken.wav').write_bytes(b'not an audio file')
    shutil.copyfile(audio, inputs / 'nested/huawei_test.wav')
    records = run_batch(diarizer, inputs, target / 'batch_results', num_speakers=3)
    assert [row['status'] for row in records] == ['failed', 'succeeded']
    for mode in ('complete', 'non_conflict'):
        actual = json.loads((target / 'batch_results/nested/huawei_test.wav' / mode / 'segments.json').read_text(encoding='utf-8'))
        baseline = json.loads((ROOT / 'outputs/verify_run' / mode / 'segments.json').read_text(encoding='utf-8'))
        assert actual['segments'] == baseline['segments']
    summary = {'verified': True, 'device': diarizer.device, 'backend': diarizer.backend,
               'http_health': health, 'http_job_id': job_id,
               'bad_audio_status': bad['status'], 'subsequent_audio_status': good['status'],
               'http_summary': result['summary'], 'batch_statuses': [r['status'] for r in records],
               'segment_regression': 'Exact start/end/speaker equality for complete and non_conflict in both HTTP and batch',
               'output_dir': str(target)}
    (target / 'verification.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
