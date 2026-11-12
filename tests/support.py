"""Shared fixtures for deterministic media, HTTP, and storage-failure tests.

The test harness never waits for media processing to finish by sleeping.  It
uses barriers and events so two requests can be placed at exact repository
boundaries, and it treats barriers only as synchronization points.  Timeouts
below are hang detectors, not attempts to guess a lucky interleaving.

Production storage failures are injected only through the content-file and
repository seams the application already uses.  No application, repository,
or media code is patched to change observable behavior.
"""

from __future__ import annotations

import base64
import hashlib
import http.client
import io
import json
import math
import struct
import sys
import threading
import wave
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from avprovenance.errors import PersistenceFailure
from avprovenance.http_api import ProvenanceHTTPServer, build_http_server
from avprovenance.media import inspect_media
from avprovenance.storage.content_files import ContentFileRepository


HANG_TIMEOUT_SECONDS = 15.0


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_wav(
    *,
    frequency: float,
    duration_seconds: float = 0.6,
    sample_rate: int = 8000,
    amplitude: int = 9000,
) -> bytes:
    frame_count = int(duration_seconds * sample_rate)
    values = [
        int(amplitude * math.sin(2 * math.pi * frequency * index / sample_rate))
        for index in range(frame_count)
    ]
    payload = io.BytesIO()
    with wave.open(payload, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(struct.pack(f"<{len(values)}h", *values))
    return payload.getvalue()


def make_y4m(
    *,
    width: int = 64,
    height: int = 36,
    frame_count: int = 7,
    rate_numerator: int = 12,
    rate_denominator: int = 1,
) -> bytes:
    header = (
        "YUV4MPEG2 "
        f"W{width} H{height} "
        f"F{rate_numerator}:{rate_denominator} "
        "Ip C420jpeg\n"
    ).encode("ascii")
    chroma_width = width // 2
    chroma_height = height // 2
    parts = [header]
    for frame_index in range(frame_count):
        parts.append(b"FRAME\n")
        y_plane = bytes(
            (x * 3 + y * 5 + frame_index * 11) % 256
            for y in range(height)
            for x in range(width)
        )
        u_plane = bytes(
            (96 + x * 2 + frame_index * 3) % 256
            for _y in range(chroma_height)
            for x in range(chroma_width)
        )
        v_plane = bytes(
            (144 + y * 2 + frame_index * 5) % 256
            for y in range(chroma_height)
            for _x in range(chroma_width)
        )
        parts.extend((y_plane, u_plane, v_plane))
    return b"".join(parts)


def make_media(media_kind: str) -> bytes:
    if media_kind == "audio":
        return make_wav(frequency=440, amplitude=10000)
    if media_kind == "video":
        return make_y4m()
    raise ValueError(f"Unsupported test media kind: {media_kind}")


def media_extension(media_kind: str) -> str:
    if media_kind == "audio":
        return ".wav"
    if media_kind == "video":
        return ".y4m"
    raise ValueError(f"Unsupported test media kind: {media_kind}")


class HTTPClient:
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port

    def request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        connection = http.client.HTTPConnection(
            self.host,
            self.port,
            timeout=20,
        )
        body = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            headers["Content-Type"] = "application/json"
            headers["Content-Length"] = str(len(body))
        try:
            connection.request(method, path, body=body, headers=headers)
            response = connection.getresponse()
            raw = response.read()
            value = json.loads(raw.decode("utf-8"))
            return response.status, value
        finally:
            connection.close()

    def upload(self, filename: str, data: bytes) -> dict[str, Any]:
        status, body = self.request(
            "POST",
            "/v1/assets",
            {
                "filename": filename,
                "data_base64": base64.b64encode(data).decode("ascii"),
            },
        )
        require_equal(status, 201, "asset upload status")
        return body

    def create_rendition(self, payload: dict[str, Any]) -> dict[str, Any]:
        status, body = self.request("POST", "/v1/renditions", payload)
        require_equal(status, 201, "rendition creation status")
        return body


def upload_payload(filename: str, data: bytes) -> dict[str, Any]:
    return {
        "filename": filename,
        "data_base64": base64.b64encode(data).decode("ascii"),
    }


@contextmanager
def running_server(
    data_dir: str | Path,
    *,
    host: str = "127.0.0.1",
) -> Iterator[tuple[HTTPClient, ProvenanceHTTPServer]]:
    server = build_http_server(host, 0, Path(data_dir))
    thread = threading.Thread(
        target=server.serve_forever,
        kwargs={"poll_interval": 0.02},
        daemon=True,
    )
    thread.start()
    address_host, address_port = server.server_address[:2]
    try:
        yield HTTPClient(str(address_host), int(address_port)), server
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run_threads(
    targets: Sequence[Callable[[], Any]],
    *,
    timeout: float = HANG_TIMEOUT_SECONDS,
) -> tuple[list[Any], list[BaseException | None]]:
    """Run targets concurrently and surface their exceptions to the caller."""

    results: list[Any] = [None] * len(targets)
    errors: list[BaseException | None] = [None] * len(targets)

    def wrapper(index: int, target: Callable[[], Any]) -> None:
        try:
            results[index] = target()
        except BaseException as exc:  # pragma: no cover - reported by assertions
            errors[index] = exc

    threads = [
        threading.Thread(target=wrapper, args=(index, target), daemon=True)
        for index, target in enumerate(targets)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout)
    alive = [thread for thread in threads if thread.is_alive()]
    if alive:
        raise AssertionError(
            f"{len(alive)} request thread(s) did not reach their synchronization "
            f"points within {timeout} seconds"
        )
    return results, errors


def content_path_for_digest(data_dir: str | Path, digest: str) -> Path:
    key = ContentFileRepository.key_for_digest(digest)
    return Path(data_dir) / "content" / key


def content_facts_for_digest(
    data_dir: str | Path,
    digest: str,
) -> dict[str, Any]:
    """Read the actual content-addressed bytes, not only recorded metadata."""

    path = content_path_for_digest(data_dir, digest)
    facts: dict[str, Any] = {
        "digest": digest,
        "content_path": str(path),
        "path_exists": path.is_file(),
        "actual_sha256": None,
        "actual_size_bytes": None,
        "descriptor": None,
        "descriptor_error": None,
        "read_error": None,
    }
    if not path.is_file():
        return facts
    try:
        data = path.read_bytes()
    except OSError as exc:
        facts["read_error"] = f"{type(exc).__name__}: {exc}"
        return facts
    facts["actual_sha256"] = sha256_bytes(data)
    facts["actual_size_bytes"] = len(data)
    try:
        facts["descriptor"] = inspect_media(data).to_dict()
    except Exception as exc:  # noqa: BLE001 - evidence, not control flow
        facts["descriptor_error"] = f"{type(exc).__name__}: {exc}"
    return facts


def first_error_code(body: dict[str, Any]) -> str | None:
    error = body.get("error")
    if isinstance(error, dict):
        code = error.get("code")
        return str(code) if code is not None else None
    return None


class _InterleavingRecorder:
    """Record synchronization facts so a failure report proves the ordering."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.records: list[dict[str, Any]] = []

    def record(self, event: str, **details: Any) -> None:
        with self._lock:
            self.records.append({"event": event, **details})


class IngestCommitFailureScheduler(_InterleavingRecorder):
    """Place both same-byte uploads before writing, then fail one commit.

    Both requests observe the blob as absent before either request writes.
    Both then write the same content-addressed target.  The winner commits its
    asset before the loser is allowed to fail, so a cleanup routine that
    deletes the shared blob would delete bytes backing a committed asset.
    """

    def __init__(
        self,
        *,
        winner_filename: str,
        loser_filename: str,
        timeout: float = HANG_TIMEOUT_SECONDS,
    ) -> None:
        super().__init__()
        self.winner_filename = winner_filename
        self.loser_filename = loser_filename
        self.timeout = timeout
        self.exists_barrier = threading.Barrier(2, timeout=timeout)
        self.write_barrier = threading.Barrier(2, timeout=timeout)
        self.winner_committed = threading.Event()
        self._application: Any = None
        self._originals: dict[str, Any] = {}
        self._installed = False

    def install(self, application: Any) -> None:
        self._application = application
        content_files = application.content_files
        repository = application.repository
        self._originals = {
            "exists": content_files.exists,
            "write": content_files.write,
            "remove": content_files.remove,
            "create_asset": repository.create_asset,
        }

        original_exists = self._originals["exists"]
        original_write = self._originals["write"]
        original_remove = self._originals["remove"]
        original_create_asset = self._originals["create_asset"]

        def exists(key: str) -> bool:
            result = original_exists(key)
            self.record("exists_observed", key=key, result=result)
            self.exists_barrier.wait()
            return result

        def write(data: bytes, digest: str) -> str:
            self.record("write_entered", digest=digest)
            result = original_write(data, digest)
            self.record("write_returned", digest=digest, key=result)
            self.write_barrier.wait()
            return result

        def remove(key: str) -> None:
            self.record(
                "remove_called",
                key=key,
                present_before_remove=original_exists(key),
                thread=threading.current_thread().name,
            )
            original_remove(key)

        def create_asset(asset: Any) -> Any:
            if asset.filename == self.loser_filename:
                self.record("loser_waiting_for_winner", filename=asset.filename)
                self.winner_committed.wait(self.timeout)
                self.record("loser_failing_commit", filename=asset.filename)
                raise PersistenceFailure("simulated asset commit failure")
            result = original_create_asset(asset)
            self.record("winner_committed", filename=asset.filename)
            self.winner_committed.set()
            return result

        content_files.exists = exists
        content_files.write = write
        content_files.remove = remove
        repository.create_asset = create_asset
        self._installed = True

    def restore(self) -> None:
        if not self._installed:
            return
        content_files = self._application.content_files
        repository = self._application.repository
        content_files.exists = self._originals["exists"]
        content_files.write = self._originals["write"]
        content_files.remove = self._originals["remove"]
        repository.create_asset = self._originals["create_asset"]
        self._installed = False


class RenditionOutputFailureScheduler(_InterleavingRecorder):
    """Place two same-output renditions before writing, then fail one commit.

    Both requests observe the output digest as absent, both write the same
    content-addressed target, and the successful rendition commits before the
    losing rendition is allowed to fail.  Cleanup by the loser therefore
    targets bytes that already back a completed rendition.
    """

    def __init__(
        self,
        *,
        winner_output_name: str,
        loser_output_name: str,
        timeout: float = HANG_TIMEOUT_SECONDS,
    ) -> None:
        super().__init__()
        self.winner_output_name = winner_output_name
        self.loser_output_name = loser_output_name
        self.timeout = timeout
        self.exists_barrier = threading.Barrier(2, timeout=timeout)
        self.write_barrier = threading.Barrier(2, timeout=timeout)
        self.winner_completed = threading.Event()
        self._application: Any = None
        self._originals: dict[str, Any] = {}
        self._installed = False

    def install(self, application: Any) -> None:
        self._application = application
        content_files = application.content_files
        repository = application.repository
        self._originals = {
            "exists": content_files.exists,
            "write": content_files.write,
            "remove": content_files.remove,
            "complete_rendition": repository.complete_rendition,
        }

        original_exists = self._originals["exists"]
        original_write = self._originals["write"]
        original_remove = self._originals["remove"]
        original_complete = self._originals["complete_rendition"]

        def exists(key: str) -> bool:
            result = original_exists(key)
            self.record("exists_observed", key=key, result=result)
            self.exists_barrier.wait()
            return result

        def write(data: bytes, digest: str) -> str:
            self.record("write_entered", digest=digest)
            result = original_write(data, digest)
            self.record("write_returned", digest=digest, key=result)
            self.write_barrier.wait()
            return result

        def remove(key: str) -> None:
            self.record(
                "remove_called",
                key=key,
                present_before_remove=original_exists(key),
                thread=threading.current_thread().name,
            )
            original_remove(key)

        def complete_rendition(
            rendition_id: str,
            output_asset: Any,
            edge: Any,
        ) -> Any:
            if output_asset.filename == self.loser_output_name:
                self.record("loser_waiting_for_winner", output_name=output_asset.filename)
                self.winner_completed.wait(self.timeout)
                self.record("loser_failing_commit", output_name=output_asset.filename)
                raise PersistenceFailure("simulated rendition commit failure")
            result = original_complete(rendition_id, output_asset, edge)
            self.record(
                "winner_completed",
                rendition_id=rendition_id,
                output_name=output_asset.filename,
            )
            self.winner_completed.set()
            return result

        content_files.exists = exists
        content_files.write = write
        content_files.remove = remove
        repository.complete_rendition = complete_rendition
        self._installed = True

    def restore(self) -> None:
        if not self._installed:
            return
        content_files = self._application.content_files
        repository = self._application.repository
        content_files.exists = self._originals["exists"]
        content_files.write = self._originals["write"]
        content_files.remove = self._originals["remove"]
        repository.complete_rendition = self._originals["complete_rendition"]
        self._installed = False


class Collector:
    """Collect all contract violations instead of stopping at the first one."""

    def __init__(self) -> None:
        self.violations: list[dict[str, Any]] = []
        self.context: dict[str, Any] = {}

    def add_context(self, **values: Any) -> None:
        self.context.update(values)

    def expect(
        self,
        condition: bool,
        code: str,
        message: str,
        **observed: Any,
    ) -> bool:
        if not condition:
            self.violations.append(
                {
                    "code": code,
                    "message": message,
                    "observed": observed,
                }
            )
        return condition

    def report(self, *, media_kind: str) -> str:
        return json.dumps(
            {
                "media_kind": media_kind,
                "interleaving_records": self.context.get(
                    "interleaving_records", []
                ),
                "violations": self.violations,
            },
            indent=2,
            sort_keys=True,
            default=str,
        )


def require_interleaving(
    collector: Collector,
    scheduler: Any,
    *,
    label: str,
) -> None:
    """Verify the barrier ordering actually happened before trusting a failure."""

    records = list(getattr(scheduler, "records", []))
    collector.add_context(interleaving_records=records)

    exists_records = [r for r in records if r["event"] == "exists_observed"]
    write_entered = [r for r in records if r["event"] == "write_entered"]
    write_records = [r for r in records if r["event"] == "write_returned"]
    winner_committed = [
        r for r in records if r["event"] in {"winner_committed", "winner_completed"}
    ]
    loser_failed = [
        r
        for r in records
        if r["event"]
        in {"loser_waiting_for_winner", "loser_failing_commit"}
    ]
    collector.expect(
        len(exists_records) >= 2
        and all(record["result"] is False for record in exists_records),
        f"{label}.both_saw_absent_blob",
        f"{label}: both requests must observe the shared blob as absent before "
        "either writes it",
        exists_records=exists_records,
    )
    collector.expect(
        len(write_records) == 2,
        f"{label}.two_writes",
        f"{label}: both requests must write their shared output once",
        write_records=write_records,
    )
    collector.expect(
        len({record["digest"] for record in write_records}) == 1,
        f"{label}.same_target",
        f"{label}: both requests must write the same content-addressed target",
        write_records=write_records,
    )
    collector.expect(
        exists_records
        and write_entered
        and records.index(exists_records[-1]) < records.index(write_entered[0]),
        f"{label}.absence_before_write",
        f"{label}: both absence checks must complete before either write starts",
        records=records,
    )
    collector.expect(
        winner_committed
        and loser_failed
        and records.index(winner_committed[0]) < records.index(loser_failed[-1]),
        f"{label}.commit_order",
        f"{label}: the winner must commit before the loser fails its save",
        records=records,
    )
    remove_records = [r for r in records if r["event"] == "remove_called"]
    collector.expect(
        len(remove_records) == 1
        and remove_records[0]["present_before_remove"] is True
        and loser_failed
        and records.index(loser_failed[0]) < records.index(remove_records[0]),
        f"{label}.failed_cleanup_removed_shared_blob",
        f"{label}: the failed request must remove the shared blob that was "
        "still present after the winner committed",
        remove_records=remove_records,
        records=records,
    )
