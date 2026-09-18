from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path


DATASET_URL = (
    "https://zenodo.org/records/1188976/files/"
    "Audio_Speech_Actors_01-24.zip?download=1"
)


def safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    destination = destination.resolve()
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        if destination not in target.parents and target != destination:
            raise ValueError(f"Unsafe archive member: {member.filename}")
    archive.extractall(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the official RAVDESS speech subset.")
    parser.add_argument("--output", type=Path, default=Path("data/ravdess"))
    parser.add_argument("--accept-license", action="store_true")
    parser.add_argument("--keep-archive", action="store_true")
    args = parser.parse_args()

    if not args.accept_license:
        parser.error(
            "Review https://zenodo.org/records/1188976 and rerun with "
            "--accept-license to acknowledge CC BY-NC-SA 4.0."
        )

    args.output.mkdir(parents=True, exist_ok=True)
    archive_path = args.output.parent / "Audio_Speech_Actors_01-24.zip"
    print(f"Downloading official archive to {archive_path} ...")
    with urllib.request.urlopen(DATASET_URL) as response, archive_path.open("wb") as output:
        shutil.copyfileobj(response, output)

    print(f"Extracting to {args.output} ...")
    with zipfile.ZipFile(archive_path) as archive:
        safe_extract(archive, args.output)

    if not args.keep_archive:
        archive_path.unlink()
    print("RAVDESS audio-only speech subset is ready.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
