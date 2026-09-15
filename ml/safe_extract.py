"""Guarded archive extraction.

Every package this project touches is untrusted -- roughly half of them are real
malware. Nothing here executes package code; archives are only unpacked to disk
so their source can be parsed as text. The guards below exist because archives
themselves are an attack surface:

* path traversal ("../../.ssh/authorized_keys") and absolute member paths
* symlinks / hardlinks pointing outside the destination
* zip bombs (small archive, enormous expansion)
* archives with an absurd number of members

Anything that trips a guard raises `UnsafeArchive` and the package is skipped.
"""

from __future__ import annotations

import tarfile
import zipfile
from pathlib import Path

from config import (
    MAX_ARCHIVE_BYTES,
    MAX_FILES_PER_PACKAGE,
    MAX_UNPACKED_BYTES,
)


class UnsafeArchive(Exception):
    """Raised when an archive violates an extraction guard."""


def _resolve_within(dest: Path, member_name: str) -> Path:
    """Resolve *member_name* under *dest*, refusing anything that escapes it."""
    if member_name.startswith("/") or member_name.startswith("\\"):
        raise UnsafeArchive(f"absolute member path: {member_name!r}")
    target = (dest / member_name).resolve()
    if target != dest and dest not in target.parents:
        raise UnsafeArchive(f"path traversal: {member_name!r}")
    return target


def extract_zip(archive: Path, dest: Path, password: bytes | None = None) -> int:
    """Extract a zip into *dest*. Returns the number of files written."""
    _check_archive_size(archive)
    dest.mkdir(parents=True, exist_ok=True)
    written = 0
    unpacked = 0

    with zipfile.ZipFile(archive) as zf:
        infos = zf.infolist()
        if len(infos) > MAX_FILES_PER_PACKAGE:
            raise UnsafeArchive(f"{len(infos)} members exceeds cap")

        for info in infos:
            if info.is_dir():
                continue
            unpacked += info.file_size
            if unpacked > MAX_UNPACKED_BYTES:
                raise UnsafeArchive("unpacked size cap exceeded (zip bomb?)")

            target = _resolve_within(dest, info.filename)
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info, pwd=password) as src, open(target, "wb") as out:
                out.write(src.read())
            written += 1

    return written


def extract_tar(archive: Path, dest: Path) -> int:
    """Extract a tarball into *dest*. Returns the number of files written."""
    _check_archive_size(archive)
    dest.mkdir(parents=True, exist_ok=True)
    written = 0
    unpacked = 0

    with tarfile.open(archive, "r:*") as tf:
        for member in tf:
            # Only ever materialise regular files. Symlinks, hardlinks, devices
            # and FIFOs have no place in a source distribution we intend to read.
            if not member.isfile():
                continue

            unpacked += member.size
            if unpacked > MAX_UNPACKED_BYTES:
                raise UnsafeArchive("unpacked size cap exceeded (tar bomb?)")
            written += 1
            if written > MAX_FILES_PER_PACKAGE:
                raise UnsafeArchive("member count cap exceeded")

            target = _resolve_within(dest, member.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            src = tf.extractfile(member)
            if src is None:
                continue
            with src, open(target, "wb") as out:
                out.write(src.read())

    return written


def extract_any(archive: Path, dest: Path, password: bytes | None = None) -> int:
    """Dispatch to the right extractor based on the archive's magic bytes."""
    if zipfile.is_zipfile(archive):
        return extract_zip(archive, dest, password=password)
    if tarfile.is_tarfile(archive):
        return extract_tar(archive, dest)
    raise UnsafeArchive(f"unrecognised archive format: {archive.name}")


def _check_archive_size(archive: Path) -> None:
    size = archive.stat().st_size
    if size > MAX_ARCHIVE_BYTES:
        raise UnsafeArchive(f"archive is {size / 1e6:.0f}MB, over the cap")
