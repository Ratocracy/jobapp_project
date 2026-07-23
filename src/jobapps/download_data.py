"""Download and verify the private S3 source datasets.

Credentials are resolved by boto3 from a named AWS profile or its normal
credential chain. No credentials are read from project configuration files.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from typing import Any

import boto3
from botocore import UNSIGNED
from botocore.client import Config
from botocore.exceptions import BotoCoreError, ClientError, ProfileNotFound
import yaml


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    remote_key: str
    local_name: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class DataManifest:
    bucket: str
    region: str
    access: str
    datasets: dict[str, DatasetSpec]


def load_manifest(path: str | Path) -> DataManifest:
    """Load and validate the versioned dataset manifest."""

    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Data manifest not found: {manifest_path}")
    with manifest_path.open(encoding="utf-8") as stream:
        raw: Any = yaml.safe_load(stream)
    if not isinstance(raw, dict) or raw.get("version") != 1:
        raise ValueError("Data manifest must be a version 1 YAML mapping")

    try:
        source = raw["source"]
        raw_datasets = raw["datasets"]
        datasets = {
            name: DatasetSpec(
                name=name,
                remote_key=str(values["remote_key"]),
                local_name=str(values["local_name"]),
                size_bytes=int(values["size_bytes"]),
                sha256=str(values["sha256"]).lower(),
            )
            for name, values in raw_datasets.items()
        }
        manifest = DataManifest(
            bucket=str(source["bucket"]),
            region=str(source["region"]),
            access=str(source["access"]),
            datasets=datasets,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid data manifest: {manifest_path}") from exc

    for spec in manifest.datasets.values():
        if Path(spec.local_name).name != spec.local_name:
            raise ValueError(f"local_name must be a filename: {spec.local_name}")
        if spec.size_bytes < 1:
            raise ValueError(f"size_bytes must be positive for {spec.name}")
        if len(spec.sha256) != 64 or any(
            character not in "0123456789abcdef" for character in spec.sha256
        ):
            raise ValueError(f"Invalid SHA-256 for {spec.name}")
    return manifest


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Hash a file without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, spec: DatasetSpec) -> tuple[bool, str]:
    """Verify one local file against its expected size and SHA-256."""

    if not path.is_file():
        return False, "missing"
    actual_size = path.stat().st_size
    if actual_size != spec.size_bytes:
        return False, f"size mismatch ({actual_size} != {spec.size_bytes})"
    actual_hash = sha256_file(path)
    if actual_hash != spec.sha256:
        return False, "SHA-256 mismatch"
    return True, "verified"


def download_dataset(
    s3_client: Any,
    bucket: str,
    spec: DatasetSpec,
    output_dir: Path,
    force: bool = False,
) -> str:
    """Safely download one dataset via a temporary file and verify it."""

    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / spec.local_name
    valid, reason = verify_file(destination, spec)
    if valid:
        return "already verified"
    if destination.exists() and not force:
        raise FileExistsError(
            f"{destination} exists but failed verification: {reason}. "
            "Rerun with --force to replace it."
        )

    temporary = destination.with_name(f"{destination.name}.part")
    if temporary.exists():
        temporary.unlink()
    try:
        s3_client.download_file(bucket, spec.remote_key, str(temporary))
        valid, reason = verify_file(temporary, spec)
        if not valid:
            raise ValueError(f"Downloaded {spec.name} failed verification: {reason}")
        temporary.replace(destination)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return "downloaded and verified"


def create_s3_client(
    region: str,
    profile: str | None = None,
    anonymous: bool = False,
):
    """Create an anonymous, profile-backed, or default-chain S3 client."""

    if anonymous:
        return boto3.client(
            "s3",
            region_name=region,
            config=Config(signature_version=UNSIGNED),
        )
    session = (
        boto3.Session(profile_name=profile, region_name=region)
        if profile
        else boto3.Session(region_name=region)
    )
    return session.client("s3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="config/data_manifest.yaml")
    parser.add_argument("--output-dir", default="raw_data")
    authentication = parser.add_mutually_exclusive_group()
    authentication.add_argument(
        "--profile",
        default=os.environ.get("AWS_PROFILE"),
        help="Use a named AWS profile instead of anonymous access",
    )
    authentication.add_argument(
        "--use-default-chain",
        action="store_true",
        help="Use the standard boto3 credential chain",
    )
    authentication.add_argument(
        "--anonymous",
        action="store_true",
        help="Force unsigned anonymous S3 access",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        help="Dataset name to download; repeat as needed. Default: all datasets",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify existing local files without contacting S3",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace a local file only when it fails manifest verification",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    requested = args.dataset or list(manifest.datasets)
    unknown = sorted(set(requested) - set(manifest.datasets))
    if unknown:
        raise ValueError(f"Unknown datasets: {', '.join(unknown)}")

    output_dir = Path(args.output_dir)
    if args.verify_only:
        failures = 0
        for name in requested:
            spec = manifest.datasets[name]
            valid, reason = verify_file(output_dir / spec.local_name, spec)
            print(f"{name}: {reason}")
            failures += int(not valid)
        if failures:
            raise SystemExit(1)
        return

    use_anonymous = args.anonymous or (
        manifest.access == "public"
        and not args.profile
        and not args.use_default_chain
    )
    profile = args.profile if not use_anonymous else None
    try:
        client = create_s3_client(
            manifest.region,
            profile=profile,
            anonymous=use_anonymous,
        )
        for name in requested:
            spec = manifest.datasets[name]
            status = download_dataset(
                client,
                manifest.bucket,
                spec,
                output_dir,
                force=args.force,
            )
            print(f"{name}: {status}")
    except ProfileNotFound as exc:
        raise SystemExit(
            f"AWS profile {profile!r} was not found. Configure it with "
            "'aws configure'/'aws configure sso', use anonymous access for "
            "public data, or pass --use-default-chain."
        ) from exc
    except (BotoCoreError, ClientError) as exc:
        raise SystemExit(
            "S3 access failed. Confirm the bucket policy permits public "
            "s3:GetObject access, or provide an authorized AWS profile. "
            f"Bucket: s3://{manifest.bucket}/. "
            f"Original error: {exc}"
        ) from exc


if __name__ == "__main__":
    main()
