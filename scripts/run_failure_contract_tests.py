#!/usr/bin/env python3
"""Run the repeatable shared-media/save-failure contract suite.

The runner never hides a failing scenario.  Each failing scenario is repeated
with the same deterministic barriers and the same injected storage failure.
Its report contains:

* the violation codes observed;
* the interleaving records proving that both requests reached the shared
  content-addressed boundary, that each observed the blob as absent, and that
  the loser failed its save only after the winner committed.

Exit codes:

* 0: positive controls passed, and every shared-blob scenario failed with the
  same repeatable evidence set on every run;
* 1: a positive control failed, a shared-blob scenario unexpectedly passed,
  or repeated runs produced different evidence.  Any of these means the
  contract suite no longer describes the implementation honestly.

Typical use:

    python3 scripts/run_failure_contract_tests.py
    python3 scripts/run_failure_contract_tests.py --repeats 3
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_MODULE = "tests.test_shared_blob_failures"
TEST_CLASS = f"{TEST_MODULE}.SharedBlobFailureContractTests"

POSITIVE_CONTROLS = (
    (
        "repeat-identical-ingest",
        "test_repeat_identical_ingest_keeps_shared_bytes",
        "Repeated identical ingest keeps shared bytes usable.",
    ),
    (
        "isolated-save-failure",
        "test_isolated_failed_saves_do_not_leave_visible_records",
        "An isolated failed save leaves no visible record or orphan blob.",
    ),
)
SHARED_BLOB_SCENARIOS = (
    (
        "shared-blob-ingest",
        "test_failed_ingest_keeps_successful_requests_blob",
        "A failing ingest must not delete a successful request's WAV/Y4M bytes.",
    ),
    (
        "shared-blob-rendition",
        "test_failed_rendition_keeps_completed_renditions_output",
        "A failing rendition must not delete a completed rendition's output.",
    ),
)


def _run(test_name: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "unittest", f"{TEST_CLASS}.{test_name}", "-v"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )


def _json_objects(output: str) -> list[dict[str, object]]:
    objects: list[dict[str, object]] = []
    position = 0
    while True:
        start = output.find("{", position)
        if start < 0:
            return objects
        depth = 0
        in_string = False
        escaped = False
        end = -1
        for offset in range(start, len(output)):
            character = output[offset]
            if in_string:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == '"':
                    in_string = False
                continue
            if character == '"':
                in_string = True
            elif character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                if depth == 0:
                    end = offset + 1
                    break
        if end < 0:
            return objects
        try:
            value = json.loads(output[start:end])
        except json.JSONDecodeError:
            value = None
        if isinstance(value, dict) and "scenario" in value and "contract" in value:
            objects.append(value)
        position = end


def _evidence_codes(reports: list[dict[str, object]]) -> list[list[str]]:
    return [
        sorted(str(violation.get("code")) for violation in report.get("violations", []))
        for report in reports
    ]


def _run_once(test_name: str) -> tuple[bool, list[dict[str, object]]]:
    outcome = _run(test_name)
    combined = outcome.stdout + outcome.stderr
    return outcome.returncode == 0, _json_objects(combined)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of identical runs per shared-blob scenario (default: 3).",
    )
    args = parser.parse_args()

    results: list[dict[str, object]] = []
    ok = True

    for scenario_id, test_name, description in POSITIVE_CONTROLS:
        passed, reports = _run_once(test_name)
        ok = ok and passed and all(
            report.get("contract") == "held" for report in reports
        )
        results.append(
            {
                "scenario": scenario_id,
                "description": description,
                "expectation": "must-pass",
                "status": "pass" if passed else "fail",
                "reports": reports,
            }
        )

    for scenario_id, test_name, description in SHARED_BLOB_SCENARIOS:
        baseline: list[list[str]] | None = None
        runs: list[dict[str, object]] = []
        for repeat in range(1, max(1, args.repeats) + 1):
            passed, reports = _run_once(test_name)
            codes = _evidence_codes(reports)
            runs.append(
                {
                    "repeat": repeat,
                    "failed_as_expected": not passed,
                    "evidence_codes": codes,
                    "violations": [
                        violation
                        for report in reports
                        for violation in report.get("violations", [])
                    ],
                    "interleaving_records": [
                        record
                        for report in reports
                        for record in report.get("interleaving_records", [])
                    ],
                    "contract_statuses": [
                        report.get("contract") for report in reports
                    ],
                }
            )
            if not passed and baseline is None:
                baseline = codes
        reproducible = (
            baseline is not None
            and all(
                run["failed_as_expected"]
                and run["evidence_codes"] == baseline
                and run["contract_statuses"]
                and all(status == "violated" for status in run["contract_statuses"])
                for run in runs
            )
        )
        ok = ok and reproducible
        results.append(
            {
                "scenario": scenario_id,
                "description": description,
                "expectation": "must-fail-until-production-is-fixed",
                "status": "evidence-reproduced" if reproducible else "unexpected",
                "runs": runs,
            }
        )

    print(json.dumps({"results": results}, indent=2, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
