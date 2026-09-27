"""Resolve a package name to an unpacked source tree.

Given something a user typed into the chat -- `requests`, `requests==2.31.0`,
`Django 4.2` -- resolve it against the PyPI JSON API, download the source
distribution, and unpack it into a temporary directory.

The package is never installed and never executed. It is downloaded, unpacked
under the guards in `ml/safe_extract.py`, read, and deleted.
"""

from __future__ import annotations

import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))

from config import (  # noqa: E402
    DOWNLOAD_TIMEOUT,
    HTTP_USER_AGENT,
    MAX_ARCHIVE_BYTES,
    PYPI_JSON_API,
)
from safe_extract import UnsafeArchive, extract_any  # noqa: E402

SDIST_SUFFIXES = (".tar.gz", ".tgz", ".zip", ".tar.bz2")

# `name`, `name==1.2.3`, `name@1.2.3`, `name 1.2.3`
SPEC_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:[=@ ]=?\s*v?([\w.!+-]+))?\s*$")


class PackageNotFound(Exception):
    """The package, or a usable source distribution for it, does not exist."""


@dataclass
class FetchedPackage:
    name: str
    version: str
    root: Path            # temp dir holding the unpacked source
    summary: str
    author: str
    home_page: str
    requires_dist: list[str]
    upload_time: str
    n_releases: int
    sdist_bytes: int

    def cleanup(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)


def parse_spec(text: str) -> tuple[str, str | None]:
    """Pull a package name and optional version out of free-form chat text."""
    m = SPEC_RE.match(text.strip())
    if not m:
        raise PackageNotFound(f"could not read a package name from {text!r}")
    return m.group(1), m.group(2)


def _pick_sdist(files: list[dict]) -> dict | None:
    for f in files:
        if f.get("packagetype") == "sdist" and f["filename"].endswith(SDIST_SUFFIXES):
            return f
    return None


async def fetch_package(spec: str) -> FetchedPackage:
    """Download and unpack a package's source distribution."""
    name, version = parse_spec(spec)

    headers = {"User-Agent": HTTP_USER_AGENT}
    async with httpx.AsyncClient(timeout=DOWNLOAD_TIMEOUT, headers=headers,
                                 follow_redirects=True) as client:
        r = await client.get(PYPI_JSON_API.format(name=name))
        if r.status_code == 404:
            raise PackageNotFound(f"'{name}' is not on PyPI")
        r.raise_for_status()
        data = r.json()

        info = data["info"]
        resolved = version or info["version"]

        if version and version not in data["releases"]:
            raise PackageNotFound(f"'{name}' has no version {version}")

        files = data["releases"][resolved] if version else data["urls"]
        sdist = _pick_sdist(files)
        if sdist is None:
            raise PackageNotFound(
                f"'{name}=={resolved}' publishes no source distribution -- only wheels. "
                "This analyser reads sdists because that is where setup.py, and so "
                "install-time code execution, lives."
            )
        if sdist["size"] > MAX_ARCHIVE_BYTES:
            raise PackageNotFound(
                f"'{name}=={resolved}' is {sdist['size'] / 1e6:.0f}MB, over the size cap"
            )

        tmp = Path(tempfile.mkdtemp(prefix="pkgscan-"))
        archive = tmp / sdist["filename"]
        try:
            async with client.stream("GET", sdist["url"]) as resp:
                resp.raise_for_status()
                with open(archive, "wb") as out:
                    async for chunk in resp.aiter_bytes(1 << 16):
                        out.write(chunk)

            root = tmp / "src"
            extract_any(archive, root)
        except UnsafeArchive as exc:
            shutil.rmtree(tmp, ignore_errors=True)
            raise PackageNotFound(f"'{name}=={resolved}' failed extraction guards: {exc}")
        except Exception:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        finally:
            archive.unlink(missing_ok=True)

    return FetchedPackage(
        name=info["name"],
        version=resolved,
        root=root,
        summary=info.get("summary") or "",
        author=info.get("author") or info.get("author_email") or "",
        home_page=info.get("home_page") or info.get("project_url") or "",
        requires_dist=info.get("requires_dist") or [],
        upload_time=(files[0].get("upload_time_iso_8601") or "") if files else "",
        n_releases=len(data.get("releases", {})),
        sdist_bytes=sdist["size"],
    )
