"""Small, dependency-free artifact/provenance utilities for the public CLI."""
import hashlib
import json
import os
from pathlib import Path
import struct


def matrix_shape(path):
    path = Path(path)
    with path.open('rb') as handle:
        raw = handle.read(8)
    if len(raw) != 8:
        raise ValueError(f'Truncated matrix header: {path}')
    rows, dim = struct.unpack('<II', raw)
    if not rows or not dim or path.stat().st_size != 8 + 4 * rows * dim:
        raise ValueError(f'Invalid fbin/ibin dimensions or size: {path}')
    return rows, dim


def identity(path):
    """Full hash for small files; stat + distributed samples for large inputs.

    This is a restart guard, not a claimed full checksum of a billion-row file.
    """
    path = Path(path).resolve(strict=True)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        if before.st_size <= 4 * 1024**2:
            digest.update(handle.read())
            mode = 'full-sha256'
        else:
            for fraction in (0, .25, .5, .75, 1):
                offset = int(max(0, before.st_size - 65536) * fraction)
                handle.seek(offset)
                digest.update(struct.pack('<Q', offset))
                digest.update(handle.read(65536))
            mode = 'stat-and-sampled-sha256'
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f'Input changed while fingerprinting: {path}')
    return dict(path=str(path), bytes=before.st_size, mtime_ns=before.st_mtime_ns,
                sha256=digest.hexdigest(), checksum_scope=mode)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def sliced_matrix(source, output, rows, batch=65536):
    """Copy a prefix without loading a full query collection into RAM."""
    count, dim = matrix_shape(source)
    if not 0 < rows <= count:
        raise ValueError('Requested build-query count is outside the input file')
    output = Path(output)
    if output.exists():
        if matrix_shape(output) != (rows, dim):
            raise ValueError(f'Existing query slice has the wrong dimensions: {output}')
        return
    temporary = output.with_name(output.name + '.tmp')
    with Path(source).open('rb') as inp, temporary.open('wb') as out:
        inp.seek(8)
        out.write(struct.pack('<II', rows, dim))
        remaining = rows * dim * 4
        while remaining:
            data = inp.read(min(remaining, batch * dim * 4))
            if not data:
                raise ValueError('Input query file was truncated during slicing')
            out.write(data)
            remaining -= len(data)
        out.flush()
        os.fsync(out.fileno())
    os.replace(temporary, output)
