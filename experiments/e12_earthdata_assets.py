"""Catalog and cache the NASA assets for the final E11 pilot sample."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re
from urllib.parse import urlsplit
from uuid import uuid4

from dotenv import dotenv_values
import pandas as pd
import requests

from experiments.e12_catalog import catalog

NASA_HOSTS = {
    "data.lpdaac.earthdatacloud.nasa.gov", "data.laadsdaac.earthdatacloud.nasa.gov",
    "ladsweb.modaps.eosdis.nasa.gov",
}
MAGIC = {".hdf": bytes.fromhex("0e031301"), ".nc": bytes.fromhex("894844460d0a1a0a")}


def read_token(path):
    raw = path.read_text(encoding="utf-8-sig").strip()
    if not raw:
        raise ValueError("Earthdata token file is empty")
    # Accept a raw token or a dotenv assignment. Never report values from this file.
    values = dotenv_values(path, encoding="utf-8-sig", interpolate=False)
    matches = [v.strip() for k, v in values.items()
               if v and any(label in k.upper() for label in ("TOKEN", "EARTHDATA", "EDL"))]
    token = matches[0] if len(matches) == 1 else raw
    token = re.sub(r"^Bearer\s+", "", token, flags=re.IGNORECASE).strip().strip("\"'")
    if not re.fullmatch(r"[A-Za-z0-9._+=/-]{24,4096}", token):
        raise ValueError("Unsupported Earthdata token file format")
    return token


def filename(url):
    parsed = urlsplit(url)
    name = parsed.path.rsplit("/", 1)[-1]
    if (parsed.scheme != "https" or parsed.hostname not in NASA_HOSTS
            or not re.fullmatch(r"(?:MYD14|MYD03|VNP14IMG|VNP03IMG)\.[A-Za-z0-9.]+\.(?:hdf|nc)", name)):
        raise ValueError("Unrecognized NASA data URL")
    return name


def archive_url(name):
    product, date, *_ = name.split(".")
    collection = "61" if product.startswith("MYD") else "5200"
    return (f"https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/{collection}/"
            f"{product}/{date[1:5]}/{date[5:8]}/{name}")


def usable(path):
    if not path.is_file() or path.stat().st_size < 4096:
        return False
    with path.open("rb") as handle:
        return handle.read(len(MAGIC[path.suffix])) == MAGIC[path.suffix]


def download_one(url, directory, token):
    name = filename(url)
    target = directory / name
    if usable(target):
        return dict(asset=name, status="cached", bytes=target.stat().st_size)
    temporary = target.with_suffix(f"{target.suffix}.{uuid4().hex}.part")
    headers = {"Authorization": f"Bearer {token}", "Accept-Encoding": "identity"}
    status = "unavailable"
    for endpoint in dict.fromkeys([url, archive_url(name)]):
        try:
            # requests strips Authorization when a redirect changes hostname.
            with requests.get(endpoint, headers=headers, stream=True, timeout=(12, 45)) as response:
                status = f"http_{response.status_code}"
                if response.status_code != 200:
                    continue
                chunks = response.iter_content(1 << 20)
                first = next(chunks, b"")
                if not first.startswith(MAGIC[target.suffix]):
                    status = "login_or_non_data"
                    continue
                with temporary.open("wb") as sink:
                    sink.write(first)
                    for chunk in chunks:
                        sink.write(chunk)
                expected = response.headers.get("Content-Length")
                if not usable_with_suffix(temporary, target.suffix) or (
                    expected and temporary.stat().st_size != int(expected)
                ):
                    status = "truncated"
                    continue
                temporary.replace(target)
                return dict(asset=name, status="downloaded", bytes=target.stat().st_size)
        except requests.RequestException as exc:
            status = type(exc).__name__
        finally:
            temporary.unlink(missing_ok=True)
    return dict(asset=name, status=status, bytes=0)


def usable_with_suffix(path, suffix):
    if not path.is_file() or path.stat().st_size < 4096:
        return False
    with path.open("rb") as handle:
        return handle.read(len(MAGIC[suffix])) == MAGIC[suffix]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token-file", type=Path)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    catalog_path = args.output / "candidate_swaths.csv"
    if not catalog_path.is_file():
        samples = pd.read_csv(args.samples, parse_dates=["acq_date", "timestamp"])
        catalog(samples).to_csv(catalog_path, index=False)
    if args.token_file is None:
        return
    try:
        token = read_token(args.token_file)
    except (ValueError, OSError) as exc:
        parser.exit(2, "Unable to read a nonempty Earthdata token from the specified file.\n")
    candidates = pd.read_csv(catalog_path)
    urls = sorted(set(candidates.mask_url.dropna()) | set(candidates.geo_url.dropna()))
    files = args.output / "swaths"
    files.mkdir(exist_ok=True)
    if not urls:
        parser.exit(2, "No NASA asset URLs in the candidate catalog.\n")
    # Probe one asset from each distribution host before sending the full batch.
    seeds = list({urlsplit(url).hostname: url for url in urls}.values())
    probes = [download_one(url, files, token) for url in seeds]
    print("Earthdata probes:", [(p["asset"].split(".")[0], p["status"]) for p in probes], flush=True)
    if not any(p["bytes"] for p in probes):
        pd.DataFrame(probes).to_csv(args.output / "download_status.csv", index=False)
        parser.exit(2, "NASA did not return any binary granules for the probes.\n")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = []
        for result in pool.map(lambda url: download_one(url, files, token), urls):
            results.append(result)
            if len(results) % 10 == 0:
                print(f"Processed {len(results)}/{len(urls)} assets", flush=True)
    audit = pd.DataFrame(results)
    audit.to_csv(args.output / "download_status.csv", index=False)
    print(audit.groupby("status").size().to_string(), flush=True)


if __name__ == "__main__":
    main()
