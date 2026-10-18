"""Download BBS (USGS ScienceBase) and WorldClim 2.1; optionally eBird Status for the cross-check."""
from __future__ import annotations

import argparse
import json
import os
import urllib.request
import zipfile
from pathlib import Path

SCIENCEBASE_ITEM = "6a0b0b0ab66b0188da36aedd"
BBS_FILES = ["Routes.csv", "Weather.csv", "States.zip", "SpeciesList.csv"]

WORLDCLIM = "https://geodata.ucdavis.edu/climate/worldclim/2_1"
BIO_VARS = [1, 4, 12, 15]
MONTHLY_VARS = ["tmin", "tmax", "prec"]
YEARS = list(range(1995, 2001)) + list(range(2006, 2012)) + list(range(2017, 2023))
DECADES = ["1990-1999", "2000-2009", "2010-2019", "2020-2024"]

EBIRD_API = "https://st-download.ebird.org/v1"
EBIRD_VERSION = "2023"
EBIRD_SPECIES = ["woothr", "logshr", "rebwoo", "carwre", "easmea"]


def fetch(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"have {dest.name}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}")
    urllib.request.urlretrieve(url, dest)


def download_bbs(out: Path) -> None:
    url = f"https://www.sciencebase.gov/catalog/item/{SCIENCEBASE_ITEM}?format=json&fields=files"
    with urllib.request.urlopen(url, timeout=60) as resp:
        files = {f["name"]: f["url"] for f in json.loads(resp.read())["files"]}
    for name in BBS_FILES:
        fetch(files[name], out / "bbs" / name)


def extract(zip_path: Path, dest: Path, wanted) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            if wanted(Path(member).name):
                (dest / Path(member).name).write_bytes(zf.read(member))


def download_worldclim(out: Path) -> None:
    base = out / "worldclim"
    tmp = out / "tmp"
    names = {f"wc2.1_10m_bio_{b}.tif" for b in BIO_VARS} | {"wc2.1_10m_elev.tif"}
    if not all((base / n).exists() for n in names):
        for kind in ("bio", "elev"):
            zip_path = tmp / f"wc2.1_10m_{kind}.zip"
            fetch(f"{WORLDCLIM}/base/wc2.1_10m_{kind}.zip", zip_path)
            base.mkdir(parents=True, exist_ok=True)
            extract(zip_path, base, lambda n: n in names)
            zip_path.unlink()

    monthly = out / "worldclim_monthly"
    monthly.mkdir(parents=True, exist_ok=True)
    for var in MONTHLY_VARS:
        for decade in DECADES:
            start, end = (int(y) for y in decade.split("-"))
            years = [y for y in YEARS if start <= y <= end]
            if not years:
                continue
            tags = tuple(f"_{var}_{y}-" for y in years)
            if all((monthly / f"wc2.1_cruts4.09_10m_{var}_{y}-{m:02d}.tif").exists() for y in years for m in range(1, 13)):
                continue
            zip_path = tmp / f"{var}_{decade}.zip"
            fetch(f"{WORLDCLIM}/hist/cts4.09/wc2.1_cruts4.09_10m_{var}_{decade}.zip", zip_path)
            extract(zip_path, monthly, lambda n: any(t in n for t in tags))
            zip_path.unlink()


def download_ebird(out: Path) -> None:
    key = os.environ.get("EBIRD_API_KEY")
    if not key:
        raise SystemExit("EBIRD_API_KEY is not set")
    for code in EBIRD_SPECIES:
        listing = f"{EBIRD_API}/list-obj/{EBIRD_VERSION}/{code}?key={key}"
        with urllib.request.urlopen(listing, timeout=30) as resp:
            files = json.loads(resp.read())
        obj_key = next(f for f in files if "27km" in f and "abundance_seasonal_max" in f)
        fetch(f"{EBIRD_API}/fetch?objKey={obj_key}&key={key}",
              out / "ebird" / f"{code}_abundance_seasonal_max_27km_{EBIRD_VERSION}.tif")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--skip-bbs", action="store_true")
    parser.add_argument("--skip-climate", action="store_true")
    parser.add_argument("--ebird", action="store_true", help="also download eBird Status (needs EBIRD_API_KEY)")
    args = parser.parse_args()

    if not args.skip_bbs:
        download_bbs(args.data)
    if not args.skip_climate:
        download_worldclim(args.data)
    if args.ebird:
        download_ebird(args.data)


if __name__ == "__main__":
    main()
