"""Fetch signed NASA CDN assets in bounded, verified byte ranges."""
import re
import shutil
from urllib.parse import urlsplit
from uuid import uuid4

import requests

from experiments.e12_earthdata_assets import MAGIC, filename, usable

CDN_HOSTS = {"d1nklfio7vscoe.cloudfront.net", "d13j1jds5ybppo.cloudfront.net"}
BLOCK = 4 * 1024 * 1024


def signed_url(session, source, token):
    with session.get(source, headers={"Authorization": f"Bearer {token}", "Range": "bytes=0-15"},
                     allow_redirects=False, stream=True, timeout=(12, 20)) as response:
        target = response.headers.get("Location", "")
        parsed = urlsplit(target)
        if response.status_code != 303 or parsed.scheme != "https" or parsed.hostname not in CDN_HOSTS:
            raise ValueError(f"unexpected NASA redirect ({response.status_code})")
        return target


def byte_range(session, target, start, end):
    with session.get(target, headers={"Range": f"bytes={start}-{end}"}, allow_redirects=False,
                     timeout=(12, 25)) as response:
        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
        if response.status_code != 206 or not match:
            raise ValueError(f"unexpected CDN range response ({response.status_code})")
        lower, upper, total = map(int, match.groups())
        data = response.content
        if lower != start or upper != min(end, total - 1) or len(data) != upper - lower + 1:
            raise ValueError("inconsistent CDN byte range")
        return data, total


def verified_prefix(session, signed, target):
    if not target.name.startswith("MYD03."):
        return None, 0, None
    partials = sorted(target.parent.glob(f"{target.stem}.*.part"),
                      key=lambda file: file.stat().st_size, reverse=True)
    for partial in partials:
        size = partial.stat().st_size
        if size < 1024 * 1024:
            continue
        with partial.open("rb") as handle:
            first = handle.read(16)
            if not first.startswith(MAGIC[target.suffix]):
                continue
            handle.seek(size - 16)
            last = handle.read(16)
        try:
            remote_first, total = byte_range(session, signed, 0, 15)
            if size > total or remote_first != first:
                continue
            remote_last, end_total = byte_range(session, signed, size - 16, size - 1)
            if end_total == total and remote_last == last:
                return partial, size, total
        except (requests.RequestException, ValueError):
            return None, 0, None
    return None, 0, None

def fetch_ranges(url, directory, token):
    target = directory / filename(url)
    if usable(target):
        return dict(asset=target.name, status="cached", bytes=target.stat().st_size)
    temp = target.with_suffix(f".{uuid4().hex}.part")
    status = "range_failed"
    try:
        with requests.Session() as session, temp.open("wb") as file:
            signed = signed_url(session, url, token)
            prefix, start, total = verified_prefix(session, signed, target)
            if prefix is not None:
                with prefix.open("rb") as existing:
                    shutil.copyfileobj(existing, file)
            block = BLOCK
            while total is None or start < total:
                end = start + block - 1
                try:
                    data, reported_total = byte_range(session, signed, start, end)
                except (requests.RequestException, ValueError):
                    signed = signed_url(session, url, token)
                    block = min(block, 512 * 1024)
                    data, reported_total = byte_range(session, signed, start, start + block - 1)
                if total is not None and reported_total != total:
                    raise ValueError("inconsistent asset size")
                total = reported_total
                if start == 0 and not data.startswith(MAGIC[target.suffix]):
                    raise ValueError("asset signature did not match")
                file.write(data)
                start += len(data)
        if temp.stat().st_size != total:
            raise ValueError("incomplete file on disk")
        temp.replace(target)
        return dict(asset=target.name, status="downloaded_range", bytes=target.stat().st_size)
    except (requests.RequestException, ValueError, OSError) as error:
        status = f"range_failed:{type(error).__name__}"
    finally:
        if temp.is_file() and (not target.name.startswith("MYD03.") or temp.stat().st_size < 1024 * 1024):
            temp.unlink()
    return dict(asset=target.name, status=status, bytes=0)
