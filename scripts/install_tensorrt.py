"""Install reviewed Windows CP311 TensorRT wheels into this project only."""
import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true', help='Download missing wheels (about 1.540 GB total)')
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 11) or platform.system() != 'Windows' or platform.machine().lower() not in {'amd64', 'x86_64'}:
        parser.error('This pinned dependency manifest is for Windows x64 Python 3.11')
    manifest = json.loads((ROOT / 'vendor/tensorrt/dependencies.json').read_text(encoding='utf-8-sig'))
    total = sum(row['Size'] for row in manifest)
    if total > 1_550_000_000:
        raise RuntimeError('Manifest exceeds the reviewed download size')
    wheels = ROOT / 'tools/gpu-wheels'
    wheels.mkdir(parents=True, exist_ok=True)
    for row in manifest:
        path = wheels / row['Name']
        size = path.stat().st_size if path.exists() else 0
        if size != row['Size']:
            if not args.download:
                raise RuntimeError(f'Missing/incomplete {path.name}. Rerun with --download to fetch the reviewed wheels.')
            if size > row['Size']:
                raise RuntimeError(f'Unexpected size for {path.name}; file preserved')
            request = urllib.request.Request(row['Url'], headers={'Range': f'bytes={size}-'} if size else {})
            with urllib.request.urlopen(request, timeout=60) as response:
                if size and response.status != 206:
                    raise RuntimeError('Server did not honor resume request')
                if size + int(response.headers['Content-Length']) != row['Size']:
                    raise RuntimeError('File size differs from reviewed manifest')
                last = time.monotonic()
                with path.open('ab') as stream:
                    while block := response.read(1024**2):
                        if size + len(block) > row['Size']:
                            raise RuntimeError('Download exceeds reviewed size; stopping')
                        stream.write(block)
                        size += len(block)
                        if time.monotonic() - last > 5:
                            print(f'{path.name}: {size / row["Size"]:.1%}', flush=True)
                            last = time.monotonic()
            if path.stat().st_size != row['Size']:
                raise RuntimeError(f'Incomplete wheel: {path.name}')
        if row.get('SHA256'):
            with path.open('rb') as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != row['SHA256']:
                    raise RuntimeError(f'SHA256 mismatch: {path.name}')
    target = ROOT / 'tools/gpu-packages'
    command = [sys.executable, '-m', 'pip', 'install', '--no-deps', '--no-index', '--no-compile',
               '--disable-pip-version-check', '--no-warn-script-location', '--upgrade',
               '--target', str(target), *[str(wheels / row['Name']) for row in manifest]]
    subprocess.run(command, check=True)
    sys.path.insert(0, str(ROOT / 'src'))
    from pyannote_dual.trt_backend import import_tensorrt
    trt = import_tensorrt()
    import onnx
    print(json.dumps({'tensorrt': trt.__version__, 'onnx': onnx.__version__,
                      'packages': str(target), 'download_manifest_bytes': total}), flush=True)


if __name__ == '__main__':
    main()
