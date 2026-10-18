"""Lossless checkpoint transport against immutable public safetensors shards.

The archive stores the exact target header and compressed bytewise XOR payloads.
No weight is quantized or omitted. Restoration is verified by the SHA256 of the
complete original safetensors file, independently of model loading.
"""

import argparse
import base64
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import time
import zipfile

import numpy as np


CHUNK = 8 * 1024 * 1024


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(CHUNK), b''):
            h.update(block)
    return h.hexdigest()


def header(path):
    with Path(path).open('rb') as stream:
        first = stream.read(8)
        length, = struct.unpack('<Q', first)
        if length > 100 * 1024 * 1024:
            raise ValueError('Invalid safetensors header length')
        raw = first + stream.read(length)
    return raw, json.loads(raw[8:])


def base_tensors(directory):
    result, files = {}, {}
    for path in sorted(Path(directory).glob('*.safetensors')):
        raw, info = header(path)
        files[path.name] = {'sha256': digest(path), 'bytes': path.stat().st_size}
        for name, value in info.items():
            if name == '__metadata__':
                continue
            if name in result:
                raise ValueError('Duplicate base tensor: ' + name)
            result[name] = {'file': path.name, 'offset': len(raw) + value['data_offsets'][0],
                            'dtype': value['dtype'], 'shape': value['shape'],
                            'bytes': value['data_offsets'][1] - value['data_offsets'][0]}
    return result, files


def safe_base(directory, name):
    path = (Path(directory) / name).resolve()
    path.relative_to(Path(directory).resolve())
    return path


def restore(archive, base, output):
    output = Path(output)
    temporary = output.with_suffix(output.suffix + '.part')
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        manifest = json.loads(source.read('manifest.json'))
        for name, expected in manifest['base_files'].items():
            path = safe_base(base, name)
            if path.stat().st_size != expected['bytes'] or digest(path) != expected['sha256']:
                raise ValueError('Public base shard differs: ' + name)
        with temporary.open('wb') as target:
            target.write(base64.b64decode(manifest['target_header_base64']))
            for tensor in manifest['tensors']:
                if target.tell() != tensor['target_offset']:
                    raise ValueError('Noncontiguous target tensor offsets')
                origin = None
                if tensor['encoding'] == 'xor':
                    origin = safe_base(base, tensor['base_file']).open('rb')
                    origin.seek(tensor['base_offset'])
                try:
                    with source.open(tensor['payload']) as payload:
                        left = tensor['bytes']
                        while left:
                            block = payload.read(min(CHUNK, left))
                            if not block:
                                raise ValueError('Incomplete tensor payload')
                            if origin is not None:
                                reference = origin.read(len(block))
                                if len(reference) != len(block):
                                    raise ValueError('Incomplete base shard')
                                block = np.bitwise_xor(np.frombuffer(block, np.uint8),
                                                       np.frombuffer(reference, np.uint8)).tobytes()
                            target.write(block)
                            left -= len(block)
                finally:
                    if origin is not None:
                        origin.close()
    if temporary.stat().st_size != manifest['target_bytes'] or digest(temporary) != manifest['target_sha256']:
        raise ValueError('Reconstructed checkpoint SHA256 differs')
    temporary.replace(output)
    return manifest


def pack(checkpoint, base, output):
    checkpoint, base, output = Path(checkpoint), Path(base), Path(output)
    started = time.time()
    references, files = base_tensors(base)
    raw, info = header(checkpoint)
    ordered = sorted(((name, value) for name, value in info.items() if name != '__metadata__'),
                     key=lambda item: item[1]['data_offsets'][0])
    manifest = {'format': 'safetensors-byte-xor-v1', 'lossless': True,
                'target_sha256': digest(checkpoint), 'target_bytes': checkpoint.stat().st_size,
                'target_header_base64': base64.b64encode(raw).decode(),
                'base_files': files, 'tensors': []}
    temporary = output.with_suffix(output.suffix + '.part')
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED,
                         compresslevel=3, allowZip64=True) as archive, checkpoint.open('rb') as target:
        for index, (name, value) in enumerate(ordered):
            begin, end = value['data_offsets']
            reference = references.get(name)
            same_layout = (reference is not None and reference['dtype'] == value['dtype']
                           and reference['shape'] == value['shape'] and reference['bytes'] == end - begin)
            tensor = {'name': name, 'payload': f'tensors/{index:05d}', 'bytes': end - begin,
                      'target_offset': len(raw) + begin, 'encoding': 'xor' if same_layout else 'raw'}
            origin = None
            if same_layout:
                tensor.update(base_file=reference['file'], base_offset=reference['offset'])
                origin = (base / reference['file']).open('rb')
                origin.seek(reference['offset'])
            target.seek(len(raw) + begin)
            try:
                with archive.open(tensor['payload'], 'w', force_zip64=True) as payload:
                    left = end - begin
                    while left:
                        block = target.read(min(CHUNK, left))
                        if not block:
                            raise ValueError('Incomplete target checkpoint')
                        if origin is not None:
                            source = origin.read(len(block))
                            if len(source) != len(block):
                                raise ValueError('Incomplete base tensor')
                            block = np.bitwise_xor(np.frombuffer(block, np.uint8),
                                                   np.frombuffer(source, np.uint8)).tobytes()
                        payload.write(block)
                        left -= len(block)
            finally:
                if origin is not None:
                    origin.close()
            manifest['tensors'].append(tensor)
            if (index + 1) % 100 == 0:
                print(json.dumps({'stage': 'lossless_checkpoint_pack', 'tensors': index + 1,
                                  'total': len(ordered), 'seconds': time.time() - started}), flush=True)
        archive.writestr('manifest.json', json.dumps(manifest, separators=(',', ':')))
    # Verify a full reconstruction before publishing the transport artifact.
    with tempfile.TemporaryDirectory(prefix='delta-check-', dir=output.parent) as directory:
        restore(temporary, base, Path(directory) / 'model.safetensors')
    temporary.replace(output)
    result = {'archive': output.name, 'archive_bytes': output.stat().st_size,
              'archive_sha256': digest(output), 'target_bytes': manifest['target_bytes'],
              'target_sha256': manifest['target_sha256'], 'lossless_roundtrip_verified': True,
              'base_files': files, 'seconds': time.time() - started}
    output.with_suffix(output.suffix + '.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['pack', 'restore'])
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'pack':
        pack(args.input, args.base, args.output)
    else:
        result = restore(args.input, args.base, args.output)
        print(json.dumps({'restored': str(args.output), 'sha256': result['target_sha256']}))


if __name__ == '__main__':
    main()
