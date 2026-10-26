"""Local HTTP upload -> queued GPU inference -> dual-output artifacts.

Uses only the Python standard library for HTTP; one resident model worker
serializes inference to avoid concurrent access to a pyannote pipeline.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import queue
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from .pipeline import Community1Diarizer, MODEL_ID
from .output import write_bundle
from .batch import AUDIO_SUFFIXES

ARTIFACTS = {f'{mode}/{name}' for mode in ('complete', 'non_conflict')
             for name in ('segments.json', 'diarization.txt', 'diarization.rttm')} | {'summary.json'}
TERMINAL = {'succeeded', 'failed'}


def write_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temporary, path)


class JobService:
    def __init__(self, diarizer, data_dir, max_pending=8):
        self.diarizer = diarizer
        self.root = Path(data_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_pending = max_pending
        self.jobs = {}
        self.lock = threading.RLock()
        self.queue = queue.Queue()
        # Do not silently replay interrupted requests after a restart.
        for path in self.root.glob('*/job.json'):
            job = json.loads(path.read_text(encoding='utf-8'))
            if job['status'] not in TERMINAL:
                job.update(status='failed', error='Service stopped before this job completed')
                write_json(path, job)
            self.jobs[job['id']] = job
        self.worker = threading.Thread(target=self._work, name='diarization-worker', daemon=True)
        self.worker.start()

    def reserve(self, filename, options):
        with self.lock:
            active = sum(job['status'] not in TERMINAL for job in self.jobs.values())
            if active >= self.max_pending:
                raise OverflowError('Job queue is full; retry after an active job finishes')
            job_id = uuid.uuid4().hex
            folder = self.root / job_id
            folder.mkdir()
            job = {'id': job_id, 'status': 'receiving', 'filename': filename,
                   'options': options, 'created_at_unix': time.time()}
            self.jobs[job_id] = job
            self.update(job_id)
            return job_id, folder / ('input' + Path(filename).suffix.lower())

    def update(self, job_id, **fields):
        with self.lock:
            self.jobs[job_id].update(fields)
            write_json(self.root / job_id / 'job.json', self.jobs[job_id])

    def snapshot(self, job_id):
        with self.lock:
            job = self.jobs.get(job_id)
            return json.loads(json.dumps(job)) if job else None

    def enqueue(self, job_id):
        self.update(job_id, status='queued')
        self.queue.put(job_id)

    def _work(self):
        while True:
            job_id = self.queue.get()
            if job_id is None:
                self.queue.task_done()
                return
            try:
                job = self.snapshot(job_id)
                self.update(job_id, status='running', started_at_unix=time.time())
                audio = self.root / job_id / ('input' + Path(job['filename']).suffix.lower())
                started = time.perf_counter()
                bundle = self.diarizer.process(audio, **job['options'])
                elapsed = time.perf_counter() - started
                result = write_bundle(bundle, self.root / job_id / 'result', elapsed)
                self.update(job_id, status='succeeded', completed_at_unix=time.time(), summary=result)
            except Exception as exc:
                self.update(job_id, status='failed', completed_at_unix=time.time(),
                            error=f'{type(exc).__name__}: {exc}')
            finally:
                self.queue.task_done()

    def close(self):
        self.queue.put(None)
        self.worker.join()


def handler_for(service, max_upload_bytes):
    class Handler(BaseHTTPRequestHandler):
        # HTTP/1.0 closes each connection, including early upload rejections.
        def setup(self):
            super().setup()
            self.connection.settimeout(60)

        def reply(self, status, value):
            body = json.dumps(value, ensure_ascii=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == '/health':
                with service.lock:
                    active = sum(j['status'] not in TERMINAL for j in service.jobs.values())
                return self.reply(200, {'status': 'ready', 'device': service.diarizer.device,
                                        'backend': service.diarizer.backend, 'active_jobs': active,
                                        'model': service.diarizer.model})
            parts = path.strip('/').split('/')
            if len(parts) < 2 or parts[0] != 'jobs':
                return self.reply(404, {'error': 'Unknown endpoint'})
            job = service.snapshot(parts[1])
            if job is None:
                return self.reply(404, {'error': 'Unknown job'})
            if len(parts) == 2:
                return self.reply(200, job)
            if job['status'] != 'succeeded':
                return self.reply(409, {'error': 'Result is not available', 'status': job['status']})
            root = service.root / job['id'] / 'result'
            if len(parts) == 3 and parts[2] == 'result':
                result = {'summary': job['summary']}
                for mode in ('complete', 'non_conflict'):
                    result[mode] = json.loads((root / mode / 'segments.json').read_text(encoding='utf-8'))
                return self.reply(200, result)
            if len(parts) >= 4 and parts[2] == 'artifacts':
                artifact = '/'.join(parts[3:])
                if artifact in ARTIFACTS:
                    data = (root / artifact).read_bytes()
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json' if artifact.endswith('.json') else 'text/plain; charset=utf-8')
                    self.send_header('Content-Length', str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
            self.reply(404, {'error': 'Unknown artifact or endpoint'})

        def do_POST(self):
            url = urlsplit(self.path)
            if url.path != '/jobs':
                return self.reply(404, {'error': 'Unknown endpoint'})
            job_id = None
            try:
                query = parse_qs(url.query, keep_blank_values=True)
                allowed = {'filename', 'num_speakers', 'min_speakers', 'max_speakers', 'streaming'}
                if set(query) - allowed or any(len(v) != 1 for v in query.values()):
                    raise ValueError('Unknown or repeated query parameter')
                filename = query.get('filename', ['audio.wav'])[0]
                if not filename or '/' in filename or '\\' in filename or Path(filename).suffix.lower() not in AUDIO_SUFFIXES:
                    raise ValueError('filename must be a supported audio/video basename')
                if self.headers.get('Transfer-Encoding'):
                    raise ValueError('Use Content-Length, not Transfer-Encoding')
                length = int(self.headers.get('Content-Length', '0'))
                if length < 1:
                    raise ValueError('A positive Content-Length is required')
                if length > max_upload_bytes:
                    return self.reply(413, {'error': 'Upload exceeds configured size limit'})
                options = {}
                for key in ('num_speakers', 'min_speakers', 'max_speakers'):
                    if key in query:
                        options[key] = int(query[key][0])
                        if options[key] < 1:
                            raise ValueError('Speaker counts must be positive integers')
                if 'num_speakers' in options and len(options) > 1:
                    raise ValueError('num_speakers cannot be combined with bounds')
                if options.get('min_speakers', 1) > options.get('max_speakers', math.inf):
                    raise ValueError('min_speakers must not exceed max_speakers')
                streaming = query.get('streaming', ['false'])[0]
                if streaming not in ('true', 'false'):
                    raise ValueError('streaming must be true or false')
                options['streaming'] = streaming == 'true'
                job_id, audio = service.reserve(filename, options)
                with audio.open('wb') as stream:
                    remaining = length
                    while remaining:
                        block = self.rfile.read(min(1024 * 1024, remaining))
                        if not block:
                            raise ValueError('Incomplete upload')
                        stream.write(block)
                        remaining -= len(block)
                service.enqueue(job_id)
            except OverflowError as exc:
                return self.reply(429, {'error': str(exc)})
            except (ValueError, TimeoutError) as exc:
                if job_id:
                    service.update(job_id, status='failed', error=str(exc))
                return self.reply(400, {'error': str(exc)})
            except OSError as exc:
                if job_id:
                    service.update(job_id, status='failed', error=f'Upload failed: {exc}')
                return self.reply(500, {'error': 'Upload could not be stored'})
            # A client disconnect after acceptance must not cancel an enqueued job.
            self.reply(202, {'id': job_id, 'status_url': f'/jobs/{job_id}',
                             'result_url': f'/jobs/{job_id}/result'})
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1', choices=['127.0.0.1', 'localhost'])
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--data-dir', default='outputs/service')
    parser.add_argument('--model', default=MODEL_ID)
    parser.add_argument('--device', choices=['cuda', 'cpu', 'auto'], default='cuda')
    parser.add_argument('--backend', choices=['torch', 'shared-frames', 'tensorrt'], default='torch')
    parser.add_argument('--engine', default=None)
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--max-pending', type=int, default=8)
    parser.add_argument('--max-upload-mb', type=int, default=512)
    args = parser.parse_args()
    if args.max_pending < 1 or args.max_upload_mb < 1:
        parser.error('Queue and upload limits must be positive')
    diarizer = Community1Diarizer.from_pretrained(model=args.model, device=args.device,
                                               offline=args.offline, backend=args.backend,
                                               engine_path=args.engine)
    service = JobService(diarizer, args.data_dir, args.max_pending)
    server = ThreadingHTTPServer((args.host, args.port), handler_for(service, args.max_upload_mb * 1024**2))
    print(json.dumps({'listening': f'http://{args.host}:{server.server_port}',
                      'device': diarizer.device, 'backend': diarizer.backend}), flush=True)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        service.close()


if __name__ == '__main__':
    main()
