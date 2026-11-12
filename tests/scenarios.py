"""Deterministic shared-media / save-failure scenarios.

Each scenario returns an evidence report instead of raising on contract
violations, so a caller can show every violated promise in one repeatable
run.  Harness problems (a hung barrier, a request thread that never finishes)
still raise, because they mean the scenario itself did not execute as
intended.

The scenarios arrange the interleaving with barriers rather than sleeps:

1. Both requests observe the content-addressed digest as absent.
2. Both write the same digest, and each waits until the other has written.
3. The winning request commits its record.
4. The losing request is released only after that commit, then its save fails.

Without the production rollback under test this would leave the shared bytes
in place.  With the current rollback, the loser deletes bytes that the winner
already references.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from avprovenance.errors import PersistenceFailure
from tests.support import (
    Collector,
    IngestCommitFailureScheduler,
    RenditionOutputFailureScheduler,
    content_facts_for_digest,
    first_error_code,
    make_media,
    media_extension,
    require_interleaving,
    run_threads,
    running_server,
    sha256_bytes,
    upload_payload,
)


MEDIA_KINDS = ("audio", "video")


def output_name(media_kind: str, stem: str) -> str:
    return f"{stem}{media_extension(media_kind)}"


def child_plan(media_kind: str) -> list[dict[str, Any]]:
    if media_kind == "audio":
        return [{"type": "audio_gain", "db": -3}]
    return [{"type": "video_luma_offset", "amount": 3}]


def _report(
    collector: Collector,
    *,
    scenario: str,
    media_kind: str,
) -> dict[str, Any]:
    return {
        "scenario": scenario,
        "media_kind": media_kind,
        "contract": "held" if not collector.violations else "violated",
        "violation_count": len(collector.violations),
        "violations": collector.violations,
        "interleaving_records": collector.context.get(
            "interleaving_records", []
        ),
    }


def _http(
    collector: Collector,
    label: str,
    client: Any,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    *,
    expected_status: int,
) -> dict[str, Any] | None:
    try:
        status, body = client.request(method, path, payload)
    except Exception as exc:  # noqa: BLE001 - evidence, not control flow
        collector.expect(
            False,
            f"{label}.exception",
            f"{label}: request raised instead of returning an HTTP response",
            exception=f"{type(exc).__name__}: {exc}",
        )
        return None
    collector.expect(
        status == expected_status,
        f"{label}.status",
        f"{label}: unexpected HTTP status",
        status=status,
        expected_status=expected_status,
        body=body,
    )
    return body


def _content_matches_record(
    collector: Collector,
    *,
    label: str,
    data_dir: Path,
    recorded_asset: dict[str, Any] | None,
    actual_facts: dict[str, Any],
) -> None:
    """Compare bytes actually on disk with the durable asset record."""

    if recorded_asset is None:
        collector.expect(
            False,
            f"{label}.record_missing",
            f"{label}: no durable asset record was available to compare",
        )
        return
    digest = recorded_asset["sha256"]
    collector.expect(
        actual_facts["path_exists"],
        f"{label}.content_missing",
        f"{label}: content-addressed media for digest {digest} is missing",
        content_path=actual_facts["content_path"],
    )
    collector.expect(
        actual_facts["actual_sha256"] == digest,
        f"{label}.actual_digest",
        f"{label}: disk bytes do not hash to the recorded asset digest",
        recorded_digest=digest,
        actual_digest=actual_facts["actual_sha256"],
    )
    collector.expect(
        actual_facts["actual_size_bytes"] == recorded_asset["size_bytes"],
        f"{label}.actual_size",
        f"{label}: on-disk size differs from the asset record",
        recorded_size=recorded_asset["size_bytes"],
        actual_size=actual_facts["actual_size_bytes"],
    )
    collector.expect(
        actual_facts["descriptor"] == recorded_asset["descriptor"],
        f"{label}.actual_descriptor",
        f"{label}: the media descriptor derived from disk bytes differs",
        recorded_descriptor=recorded_asset["descriptor"],
        actual_descriptor=actual_facts["descriptor"],
        descriptor_error=actual_facts["descriptor_error"],
        read_error=actual_facts["read_error"],
    )


def _state_rendition_by_output_name(
    state: dict[str, Any],
    name: str,
) -> dict[str, Any] | None:
    for raw in state.get("renditions", {}).values():
        if isinstance(raw, dict) and raw.get("output_name") == name:
            return raw
    return None


def _edges_for(state: dict[str, Any], rendition_id: str) -> list[dict[str, Any]]:
    return [
        raw
        for raw in state.get("provenance_edges", [])
        if isinstance(raw, dict) and raw.get("rendition_id") == rendition_id
    ]


def run_shared_blob_ingest_scenario(media_kind: str) -> dict[str, Any]:
    """Two same-byte uploads; the loser's asset commit fails after the winner.

    Afterwards the scenario requires, for both the original data directory and
    a freshly opened application:

    * the winner's durable asset record, descriptor, and actual bytes agree;
    * the winner's WAV/Y4M can still be used to derive a new asset;
    * that new asset passes a reproduction check against bytes on disk.
    """

    collector = Collector()
    media = make_media(media_kind)
    digest = sha256_bytes(media)
    winner_name = output_name(media_kind, "winner")
    loser_name = output_name(media_kind, "loser")
    scheduler = IngestCommitFailureScheduler(
        winner_filename=winner_name,
        loser_filename=loser_name,
    )

    with tempfile.TemporaryDirectory(
        prefix="avprovenance-ingest-race-"
    ) as temporary:
        data_dir = Path(temporary)
        with running_server(data_dir) as (client, server):
            scheduler.install(server.application)
            try:
                responses, errors = run_threads(
                    [
                        lambda: client.request(
                            "POST",
                            "/v1/assets",
                            upload_payload(winner_name, media),
                        ),
                        lambda: client.request(
                            "POST",
                            "/v1/assets",
                            upload_payload(loser_name, media),
                        ),
                    ]
                )
            finally:
                scheduler.restore()
            state = server.application.state_file.read()
            facts_while_running = content_facts_for_digest(data_dir, digest)
            require_interleaving(collector, scheduler, label="ingest")

        winner = responses[0]
        loser = responses[1]
        for index, error in enumerate(errors):
            collector.expect(
                error is None,
                f"request_{index}.exception",
                f"request {index} raised instead of returning an HTTP result",
                exception=f"{type(error).__name__}: {error}" if error else None,
            )
        collector.expect(
            winner is not None and winner[0] == 201,
            "winner.status",
            "the successful ingest did not return 201",
            status=winner[0] if winner else None,
            body=winner[1] if winner else None,
        )
        collector.expect(
            loser is not None and loser[0] == 500,
            "loser.status",
            "the failed save did not return a stable 500 response",
            status=loser[0] if loser else None,
            body=loser[1] if loser else None,
        )
        collector.expect(
            loser is not None
            and first_error_code(loser[1]) == "PERSISTENCE_FAILURE",
            "loser.error_code",
            "the failed save did not report PERSISTENCE_FAILURE",
            body=loser[1] if loser else None,
        )

        winner_body = winner[1] if winner else {}
        winner_id = winner_body.get("id")
        durable_winner = state["assets"].get(winner_id or "")
        collector.expect(
            durable_winner is not None,
            "winner.durable_record",
            "the successful ingest left no durable asset record",
            state_assets=sorted(state["assets"]),
        )
        collector.expect(
            len(state["assets"]) == 1,
            "loser.durable_record",
            "the failed ingest left an asset record",
            state_assets=sorted(state["assets"]),
        )
        _content_matches_record(
            collector,
            label="winner.immediately_after_failure",
            data_dir=data_dir,
            recorded_asset=durable_winner,
            actual_facts=facts_while_running,
        )

        # Reopen the data directory in a completely fresh application.
        with running_server(data_dir) as (client2, _server2):
            fetched = _http(
                collector,
                "reopen.asset_read",
                client2,
                "GET",
                f"/v1/assets/{winner_id}",
                expected_status=200,
            )
            collector.expect(
                (fetched or {}).get("sha256") == digest,
                "reopen.asset_digest",
                "the surviving asset digest changed after reopening",
                fetched_digest=(fetched or {}).get("sha256"),
                expected_digest=digest,
            )

            created = _http(
                collector,
                "reopen.child_create",
                client2,
                "POST",
                "/v1/renditions",
                {
                    "primary_asset_id": winner_id,
                    "output_name": output_name(media_kind, "reopened-child"),
                    "operations": child_plan(media_kind),
                },
                expected_status=201,
            )
            collector.expect(
                (created or {}).get("status") == "completed"
                and (created or {}).get("output_asset") is not None,
                "reopen.child_completed",
                "the surviving media could not be derived after reopening",
                rendition=created,
            )
            child_id = (created or {}).get("id")
            if child_id:
                check = _http(
                    collector,
                    "reopen.child_integrity",
                    client2,
                    "POST",
                    f"/v1/renditions/{child_id}/integrity-checks",
                    {},
                    expected_status=201,
                )
                collector.expect(
                    (check or {}).get("status") == "verified",
                    "reopen.child_verified",
                    "a derivative of the surviving media did not verify",
                    check=check,
                )
                if (check or {}).get("status") == "verified":
                    child_digest = check["reproduced_digest"]
                    child_facts = content_facts_for_digest(data_dir, child_digest)
                    collector.expect(
                        child_facts["path_exists"]
                        and child_facts["actual_sha256"] == child_digest,
                        "reopen.child_bytes",
                        "the reopened derivative's bytes are not on disk",
                        facts=child_facts,
                    )

    return _report(collector, scenario="shared-blob-ingest", media_kind=media_kind)


def run_shared_blob_rendition_scenario(media_kind: str) -> dict[str, Any]:
    """Two renditions with identical output bytes; the loser's commit fails.

    Afterwards the scenario requires, for both the original data directory and
    a freshly opened application:

    * the completed rendition's status, output asset, and single edge agree;
    * the failed rendition is terminal, outputless, edge-free, and empty in
      lineage;
    * the completed output's actual bytes still hash and inspect correctly;
    * a fresh integrity check verifies against those bytes;
    * the output can still be used as a parent for another rendition whose
      own reproduction check passes.
    """

    collector = Collector()
    source_bytes = make_media(media_kind)
    winner_output_name = output_name(media_kind, "winner-output")
    loser_output_name = output_name(media_kind, "loser-output")
    plan = child_plan(media_kind)
    scheduler = RenditionOutputFailureScheduler(
        winner_output_name=winner_output_name,
        loser_output_name=loser_output_name,
    )

    with tempfile.TemporaryDirectory(
        prefix="avprovenance-rendition-race-"
    ) as temporary:
        data_dir = Path(temporary)
        with running_server(data_dir) as (client, server):
            source = client.upload(
                output_name(media_kind, "source"),
                source_bytes,
            )
            scheduler.install(server.application)
            try:
                responses, errors = run_threads(
                    [
                        lambda: client.request(
                            "POST",
                            "/v1/renditions",
                            {
                                "primary_asset_id": source["id"],
                                "output_name": winner_output_name,
                                "operations": plan,
                            },
                        ),
                        lambda: client.request(
                            "POST",
                            "/v1/renditions",
                            {
                                "primary_asset_id": source["id"],
                                "output_name": loser_output_name,
                                "operations": plan,
                            },
                        ),
                    ]
                )
            finally:
                scheduler.restore()
            state = server.application.state_file.read()
            require_interleaving(collector, scheduler, label="rendition")

        winner = responses[0]
        loser = responses[1]
        for index, error in enumerate(errors):
            collector.expect(
                error is None,
                f"request_{index}.exception",
                f"request {index} raised instead of returning an HTTP result",
                exception=f"{type(error).__name__}: {error}" if error else None,
            )
        collector.expect(
            winner is not None
            and winner[0] == 201
            and winner[1].get("status") == "completed",
            "winner.status",
            "the successful rendition did not complete",
            status=winner[0] if winner else None,
            body=winner[1] if winner else None,
        )
        collector.expect(
            loser is not None and loser[0] == 500,
            "loser.status",
            "the failed rendition save did not return a stable 500 response",
            status=loser[0] if loser else None,
            body=loser[1] if loser else None,
        )
        collector.expect(
            loser is not None
            and first_error_code(loser[1]) == "PERSISTENCE_FAILURE",
            "loser.error_code",
            "the failed rendition save did not report PERSISTENCE_FAILURE",
            body=loser[1] if loser else None,
        )

        winner_rendition = _state_rendition_by_output_name(
            state, winner_output_name
        )
        loser_rendition = _state_rendition_by_output_name(
            state, loser_output_name
        )
        collector.expect(
            winner_rendition is not None
            and winner_rendition.get("status") == "completed",
            "winner.durable_status",
            "the successful rendition is not durably completed",
            rendition=winner_rendition,
        )
        collector.expect(
            loser_rendition is not None
            and loser_rendition.get("status") == "failed",
            "loser.durable_status",
            "the failed rendition is not durably failed",
            rendition=loser_rendition,
        )
        collector.expect(
            loser_rendition is not None
            and loser_rendition.get("output_asset_id") is None,
            "loser.no_output",
            "the failed rendition exposes an output asset",
            rendition=loser_rendition,
        )
        if loser_rendition is not None:
            collector.expect(
                _edges_for(state, loser_rendition["id"]) == [],
                "loser.no_edge",
                "the failed rendition left a provenance edge",
                edges=_edges_for(state, loser_rendition["id"]),
            )

        if winner_rendition is None:
            return _report(
                collector,
                scenario="shared-blob-rendition",
                media_kind=media_kind,
            )
        winner_rendition_id = winner_rendition["id"]
        winner_edges = _edges_for(state, winner_rendition_id)
        collector.expect(
            len(winner_edges) == 1,
            "winner.edge_count",
            "the completed rendition does not have exactly one edge",
            edges=winner_edges,
        )
        if len(winner_edges) != 1:
            return _report(
                collector,
                scenario="shared-blob-rendition",
                media_kind=media_kind,
            )
        output_asset_id = winner_rendition.get("output_asset_id")
        winner_asset = state["assets"].get(output_asset_id or "")
        collector.expect(
            winner_asset is not None,
            "winner.output_record",
            "the completed rendition has no durable output asset",
            output_asset_id=output_asset_id,
        )
        collector.expect(
            winner_edges[0].get("output_asset_id") == output_asset_id,
            "winner.edge_output",
            "the provenance edge and rendition disagree about the output",
            edge=winner_edges[0],
            rendition=winner_rendition,
        )
        collector.expect(
            winner_edges[0].get("plan_digest")
            == winner_rendition.get("plan_digest"),
            "winner.edge_plan",
            "the provenance edge and rendition disagree about the plan",
            edge=winner_edges[0],
            rendition=winner_rendition,
        )
        collector.expect(
            winner_edges[0].get("parent_asset_ids") == [source["id"]],
            "winner.edge_parents",
            "the provenance edge does not name the source parent",
            edge=winner_edges[0],
        )
        if winner_asset is None:
            return _report(
                collector,
                scenario="shared-blob-rendition",
                media_kind=media_kind,
            )
        output_digest = winner_asset["sha256"]
        _content_matches_record(
            collector,
            label="winner.output_immediately_after_failure",
            data_dir=data_dir,
            recorded_asset=winner_asset,
            actual_facts=content_facts_for_digest(data_dir, output_digest),
        )

        with running_server(data_dir) as (client2, _server2):
            winner_view = _http(
                collector,
                "reopen.winner_view",
                client2,
                "GET",
                f"/v1/renditions/{winner_rendition_id}",
                expected_status=200,
            )
            collector.expect(
                (winner_view or {}).get("status") == "completed"
                and ((winner_view or {}).get("output_asset") or {}).get("id")
                == output_asset_id,
                "reopen.winner_state",
                "the reopened rendition state does not match the output",
                rendition=winner_view,
            )

            if loser_rendition is not None:
                loser_view = _http(
                    collector,
                    "reopen.loser_view",
                    client2,
                    "GET",
                    f"/v1/renditions/{loser_rendition['id']}",
                    expected_status=200,
                )
                collector.expect(
                    (loser_view or {}).get("status") == "failed"
                    and (loser_view or {}).get("output_asset") is None,
                    "reopen.loser_state",
                    "the reopened failed rendition is not failed and outputless",
                    rendition=loser_view,
                )
                loser_lineage = _http(
                    collector,
                    "reopen.loser_lineage",
                    client2,
                    "GET",
                    f"/v1/renditions/{loser_rendition['id']}/lineage",
                    expected_status=200,
                )
                collector.expect(
                    (loser_lineage or {}).get("status") == "failed"
                    and (loser_lineage or {}).get("parents") == []
                    and (loser_lineage or {}).get("edge") is None,
                    "reopen.loser_lineage_state",
                    "the failed rendition exposed lineage after reopening",
                    lineage=loser_lineage,
                )

            lineage = _http(
                collector,
                "reopen.lineage",
                client2,
                "GET",
                f"/v1/renditions/{winner_rendition_id}/lineage",
                expected_status=200,
            )
            collector.expect(
                (lineage or {}).get("status") == "completed"
                and [
                    parent.get("id")
                    for parent in (lineage or {}).get("parents", [])
                ]
                == [source["id"]]
                and ((lineage or {}).get("output") or {}).get("sha256")
                == output_digest
                and ((lineage or {}).get("edge") or {}).get("plan_digest")
                == winner_rendition.get("plan_digest"),
                "reopen.lineage_state",
                "lineage after reopening does not describe the output",
                lineage=lineage,
            )

            check = _http(
                collector,
                "reopen.winner_integrity",
                client2,
                "POST",
                f"/v1/renditions/{winner_rendition_id}/integrity-checks",
                {},
                expected_status=201,
            )
            collector.expect(
                (check or {}).get("status") == "verified",
                "reopen.winner_verified",
                "the completed output did not verify after reopening",
                check=check,
            )
            if (check or {}).get("status") == "verified":
                collector.expect(
                    check.get("expected_digest") == output_digest
                    and check.get("reproduced_digest") == output_digest
                    and check.get("expected_descriptor")
                    == winner_asset["descriptor"]
                    and check.get("reproduced_descriptor")
                    == winner_asset["descriptor"],
                    "reopen.winner_check_matches_record",
                    "the verified check disagrees with the durable output record",
                    check=check,
                    recorded_digest=output_digest,
                    recorded_descriptor=winner_asset["descriptor"],
                )

            # The surviving output must also work as a parent for a later
            # derivation; verified equality alone would not prove that.
            grandchild = _http(
                collector,
                "reopen.grandchild_create",
                client2,
                "POST",
                "/v1/renditions",
                {
                    "primary_asset_id": output_asset_id,
                    "output_name": output_name(media_kind, "grandchild"),
                    "operations": plan,
                },
                expected_status=201,
            )
            collector.expect(
                (grandchild or {}).get("status") == "completed"
                and (grandchild or {}).get("output_asset") is not None,
                "reopen.grandchild_completed",
                "the completed output could not be used as a later parent",
                rendition=grandchild,
            )
            grandchild_id = (grandchild or {}).get("id")
            if grandchild_id:
                grandchild_check = _http(
                    collector,
                    "reopen.grandchild_integrity",
                    client2,
                    "POST",
                    f"/v1/renditions/{grandchild_id}/integrity-checks",
                    {},
                    expected_status=201,
                )
                collector.expect(
                    (grandchild_check or {}).get("status") == "verified",
                    "reopen.grandchild_verified",
                    "the later derivative did not verify",
                    check=grandchild_check,
                )
                if (grandchild_check or {}).get("status") == "verified":
                    grandchild_digest = grandchild_check["reproduced_digest"]
                    grandchild_facts = content_facts_for_digest(
                        data_dir, grandchild_digest
                    )
                    collector.expect(
                        grandchild_facts["path_exists"]
                        and grandchild_facts["actual_sha256"]
                        == grandchild_digest,
                        "reopen.grandchild_bytes",
                        "the later derivative's bytes are not on disk",
                        facts=grandchild_facts,
                    )

    return _report(
        collector,
        scenario="shared-blob-rendition",
        media_kind=media_kind,
    )


def run_repeat_ingest_scenario(media_kind: str) -> dict[str, Any]:
    """Positive control: two successful uploads of identical bytes."""

    collector = Collector()
    media = make_media(media_kind)
    digest = sha256_bytes(media)
    with tempfile.TemporaryDirectory(prefix="avprovenance-repeat-") as temporary:
        data_dir = Path(temporary)
        with running_server(data_dir) as (client, _server):
            first = client.upload(output_name(media_kind, "first"), media)
            second = client.upload(output_name(media_kind, "second"), media)
        collector.expect(
            first["id"] != second["id"],
            "repeat.distinct_ids",
            "reusing bytes must create a distinct asset identifier",
            first=first["id"],
            second=second["id"],
        )
        collector.expect(
            first["sha256"] == digest and second["sha256"] == digest,
            "repeat.shared_digest",
            "identical bytes must share one digest",
            digests=[first["sha256"], second["sha256"]],
            expected=digest,
        )
        facts = content_facts_for_digest(data_dir, digest)
        collector.expect(
            facts["path_exists"] and facts["actual_sha256"] == digest,
            "repeat.shared_bytes",
            "successful repeats did not leave the shared bytes on disk",
            facts=facts,
        )

        with running_server(data_dir) as (client2, _server2):
            third = client2.upload(output_name(media_kind, "third"), media)
            collector.expect(
                third["sha256"] == digest,
                "repeat.reopen_digest",
                "a fresh upload after reopening changed the digest",
                digest=third["sha256"],
            )
            created = _http(
                collector,
                "repeat.child_create",
                client2,
                "POST",
                "/v1/renditions",
                {
                    "primary_asset_id": first["id"],
                    "output_name": output_name(media_kind, "child"),
                    "operations": child_plan(media_kind),
                },
                expected_status=201,
            )
            collector.expect(
                (created or {}).get("status") == "completed",
                "repeat.child_completed",
                "the repeated media could not be derived",
                rendition=created,
            )

    return _report(
        collector,
        scenario="repeat-identical-ingest",
        media_kind=media_kind,
    )


def run_isolated_save_failure_scenario() -> dict[str, Any]:
    """Positive control: a failed save with no sharing leaves no trace."""

    collector = Collector()
    media = make_media("audio")
    with tempfile.TemporaryDirectory(prefix="avprovenance-isolated-") as temporary:
        data_dir = Path(temporary)
        with running_server(data_dir) as (client, server):
            original_create_asset = server.application.repository.create_asset

            def fail_create_asset(asset: Any) -> Any:
                del asset
                raise PersistenceFailure(
                    "simulated isolated asset commit failure"
                )

            server.application.repository.create_asset = fail_create_asset
            try:
                status, body = client.request(
                    "POST",
                    "/v1/assets",
                    upload_payload("isolated-failure.wav", media),
                )
            finally:
                server.application.repository.create_asset = original_create_asset
            state = server.application.state_file.read()
            facts = content_facts_for_digest(data_dir, sha256_bytes(media))

        collector.expect(
            status == 500 and first_error_code(body) == "PERSISTENCE_FAILURE",
            "isolated.status",
            "an isolated failed save did not report PERSISTENCE_FAILURE",
            status=status,
            body=body,
        )
        collector.expect(
            state["assets"] == {},
            "isolated.no_record",
            "an isolated failed save left an asset record",
            assets=state["assets"],
        )
        collector.expect(
            not facts["path_exists"],
            "isolated.orphan_cleanup",
            "an isolated failed save left an unreferenced blob",
            facts=facts,
        )

    return _report(
        collector,
        scenario="isolated-save-failure",
        media_kind="audio",
    )
