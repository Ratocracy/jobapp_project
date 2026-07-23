import hashlib
from pathlib import Path

import pytest

from jobapps.download_data import (
    DatasetSpec,
    download_dataset,
    load_manifest,
    verify_file,
)


class FakeS3Client:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.calls: list[tuple[str, str, str]] = []

    def download_file(self, bucket: str, key: str, destination: str) -> None:
        self.calls.append((bucket, key, destination))
        Path(destination).write_bytes(self.payload)


def _spec(payload: bytes) -> DatasetSpec:
    return DatasetSpec(
        name="example",
        remote_key="example.parquet",
        local_name="example.parquet",
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def test_project_manifest_loads_expected_datasets() -> None:
    path = Path(__file__).resolve().parents[1] / "config" / "data_manifest.yaml"
    manifest = load_manifest(path)

    assert manifest.bucket == "s3ds5110jobspark"
    assert manifest.region == "us-east-2"
    assert manifest.access == "public"
    assert set(manifest.datasets) == {
        "linkedin_job_postings",
        "job_summary",
        "job_skills",
        "resume_screening",
    }


def test_download_uses_temporary_file_and_verifies(tmp_path: Path) -> None:
    payload = b"parquet fixture"
    spec = _spec(payload)
    client = FakeS3Client(payload)

    status = download_dataset(client, "example-bucket", spec, tmp_path)

    assert status == "downloaded and verified"
    assert (tmp_path / spec.local_name).read_bytes() == payload
    assert not (tmp_path / f"{spec.local_name}.part").exists()
    assert len(client.calls) == 1


def test_valid_existing_file_is_not_downloaded(tmp_path: Path) -> None:
    payload = b"existing parquet"
    spec = _spec(payload)
    (tmp_path / spec.local_name).write_bytes(payload)
    client = FakeS3Client(b"different")

    status = download_dataset(client, "example-bucket", spec, tmp_path)

    assert status == "already verified"
    assert client.calls == []


def test_invalid_existing_file_requires_force(tmp_path: Path) -> None:
    payload = b"correct"
    spec = _spec(payload)
    destination = tmp_path / spec.local_name
    destination.write_bytes(b"incorrect")

    with pytest.raises(FileExistsError, match="--force"):
        download_dataset(FakeS3Client(payload), "example-bucket", spec, tmp_path)

    status = download_dataset(
        FakeS3Client(payload), "example-bucket", spec, tmp_path, force=True
    )
    assert status == "downloaded and verified"
    assert verify_file(destination, spec) == (True, "verified")
