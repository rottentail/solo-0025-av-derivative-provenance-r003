"""unittest wrappers around the deterministic shared-media scenarios.

Failure scenarios intentionally fail on the current production implementation.
Their assertion text contains the full evidence report: violation codes plus
the interleaving records that prove the two requests really shared bytes and
that the loser failed after the winner committed.  They must not be skipped,
weakened, or rewritten to pass without fixing production behavior.
"""

from __future__ import annotations

import json
import unittest

from tests.scenarios import (
    MEDIA_KINDS,
    run_isolated_save_failure_scenario,
    run_repeat_ingest_scenario,
    run_shared_blob_ingest_scenario,
    run_shared_blob_rendition_scenario,
)


def _assert_contract(report: dict) -> None:
    if report["contract"] != "held":
        raise AssertionError(json.dumps(report, indent=2, sort_keys=True))


class SharedBlobFailureContractTests(unittest.TestCase):
    def test_repeat_identical_ingest_keeps_shared_bytes(self) -> None:
        """Two successful requests sharing bytes must leave one usable blob."""

        for media_kind in MEDIA_KINDS:
            with self.subTest(media_kind=media_kind):
                _assert_contract(run_repeat_ingest_scenario(media_kind))

    def test_failed_ingest_keeps_successful_requests_blob(self) -> None:
        """Two requests share bytes; the losing save fails after the winner."""

        for media_kind in MEDIA_KINDS:
            with self.subTest(media_kind=media_kind):
                _assert_contract(run_shared_blob_ingest_scenario(media_kind))

    def test_failed_rendition_keeps_completed_renditions_output(self) -> None:
        """Two renditions share output bytes; the loser fails after completion."""

        for media_kind in MEDIA_KINDS:
            with self.subTest(media_kind=media_kind):
                _assert_contract(run_shared_blob_rendition_scenario(media_kind))

    def test_isolated_failed_saves_do_not_leave_visible_records(self) -> None:
        """Without any sharing, failed saves still expose no record or blob."""

        _assert_contract(run_isolated_save_failure_scenario())


if __name__ == "__main__":
    unittest.main()
