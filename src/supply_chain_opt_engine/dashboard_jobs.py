"""Persistent single-host scenario jobs with deduplication and a serialized worker."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

from supply_chain_opt_engine.m5 import checksum
from supply_chain_opt_engine.optimization import OptimizationConfig, run_optimization

ACTIVE = {"pending", "running"}
MODEL_FILES = ["optimization.py", "inventory_data.py", "inventory_model.py", "simulation.py"]


def now():
    return datetime.now(UTC).isoformat()


def atomic_json(path, document):
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(document, indent=2) + "\n")
    temp.replace(path)


@contextmanager
def file_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def model_hashes():
    directory = Path(__file__).parent
    return {name: checksum(directory / name) for name in MODEL_FILES}


def selected_config(defaults, quantile, capacity_percent, delay):
    if (
        isinstance(capacity_percent, bool)
        or not isinstance(capacity_percent, (int, float))
        or not 10 <= capacity_percent <= 150
    ):
        raise ValueError("Choose a capacity between 10% and 150% of baseline.")
    if isinstance(delay, bool) or not isinstance(delay, (int, float)) or delay != int(delay):
        raise ValueError("Choose a whole-day lead-time delay from 0 to 4.")
    return replace(
        defaults,
        service_quantile=quantile,
        capacity_multiplier=capacity_percent / 100.0,
        lead_time_shock_days=int(delay),
    )


def process_alive(pid, job_id):
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        # A reused PID or zombie is not an active scenario worker.
        return job_id.encode() in Path(f"/proc/{pid}/cmdline").read_bytes()
    except (OSError, ValueError):
        return False


class JobManager:
    def __init__(self, root, recpath, basepath, defaults=None, launcher=subprocess.Popen):
        self.root = Path(root).resolve()
        self.recpath = Path(recpath).resolve()
        self.basepath = Path(basepath).resolve()
        self.defaults = defaults or OptimizationConfig()
        self.directory = self.root / "artifacts/dashboard/jobs"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.launcher = launcher

    def path(self, job_id):
        if not isinstance(job_id, str) or not re.fullmatch("[a-f0-9]{64}", job_id):
            raise ValueError("Unknown scenario job.")
        return self.directory / f"{job_id}.json"

    def identity(self, phase, config):
        if phase not in {"holdout", "latest"}:
            raise ValueError("Choose the holdout or latest evaluation window.")
        request = {
            "phase": phase,
            "config": asdict(config),
            "reconciliation_manifest": str(self.recpath.relative_to(self.root)),
            "reconciliation_sha256": checksum(self.recpath),
            "baseline_manifest": str(self.basepath.relative_to(self.root)),
            "baseline_sha256": checksum(self.basepath),
            "model_sha256": model_hashes(),
        }
        key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        return key, request

    def read(self, job_id):
        path = self.path(job_id)
        with file_lock(self.directory / ".submit.lock"):
            if not path.exists():
                raise ValueError("This scenario job is no longer available.")
            state = json.loads(path.read_text())
            age = (datetime.now(UTC) - datetime.fromisoformat(state["updated_at"])).total_seconds()
            if (
                state["status"] in ACTIVE
                and age > 5
                and not process_alive(state.get("worker_pid"), job_id)
            ):
                state.update(
                    status="failed",
                    updated_at=now(),
                    error="The scenario worker stopped. Run optimizer to retry.",
                )
                atomic_json(path, state)
        return state

    def submit(self, phase, config, reference=None):
        key, request = self.identity(phase, config)
        path = self.path(key)
        with file_lock(self.directory / ".submit.lock"):
            if path.exists():
                state = json.loads(path.read_text())
                if state["status"] in ACTIVE or state["status"] == "completed":
                    # Stale detection is performed by read before resubmission.
                    return state
            state = {
                "job_id": key,
                "request": request,
                "status": "pending",
                "created_at": now(),
                "updated_at": now(),
                "worker_pid": None,
                "result_manifest": None,
                "error": None,
            }
            if reference is not None:
                reference = Path(reference).resolve()
                if not reference.is_relative_to(self.root / "artifacts/orders"):
                    raise ValueError("Reference result must be a local inventory bundle.")
                state.update(
                    status="completed",
                    result_manifest=str(reference.relative_to(self.root)),
                    reused_reference=True,
                )
                atomic_json(path, state)
                return state
            atomic_json(path, state)
            command = [
                sys.executable,
                "-m",
                "supply_chain_opt_engine.dashboard_jobs",
                "--root",
                str(self.root),
                "--job",
                key,
            ]
            env = {**os.environ, "MPLCONFIGDIR": str(self.root / "artifacts/dashboard/mpl")}
            try:
                with (self.directory / f"{key}.log").open("ab") as log:
                    child = self.launcher(
                        command,
                        cwd=self.root,
                        stdout=log,
                        stderr=log,
                        env=env,
                        start_new_session=True,
                    )
                state["worker_pid"] = child.pid
            except OSError:
                state.update(status="failed", error="The scenario worker could not start.")
            state["updated_at"] = now()
            atomic_json(path, state)
        return state


def execute_job(root, job_id, runner=run_optimization):
    root = Path(root).resolve()
    directory = root / "artifacts/dashboard/jobs"
    if not re.fullmatch("[a-f0-9]{64}", job_id):
        raise ValueError("Invalid scenario ID.")
    path = directory / f"{job_id}.json"
    # Workers can wait here without occupying the Dash request process.
    with file_lock(directory / ".worker.lock"):
        with file_lock(directory / ".submit.lock"):
            state = json.loads(path.read_text())
            if state["status"] != "pending":
                return state
            state.update(status="running", worker_pid=os.getpid(), updated_at=now())
            atomic_json(path, state)
        try:
            request = state["request"]
            for name, expected in [
                ("reconciliation_manifest", "reconciliation_sha256"),
                ("baseline_manifest", "baseline_sha256"),
            ]:
                if checksum(root / request[name]) != request[expected]:
                    raise ValueError("Saved source data changed. Restart the dashboard.")
            if request["model_sha256"] != model_hashes():
                raise ValueError("Optimizer code changed. Restart the dashboard.")
            config = OptimizationConfig(**request["config"])
            bundle = runner(
                root,
                config,
                reconciliation_path=root / request["reconciliation_manifest"],
                baseline_path=root / request["baseline_manifest"],
                phases=(request["phase"],),
            )
            bundle = Path(bundle).resolve()
            manifest = json.loads((bundle / "manifest.json").read_text())
            if (
                not bundle.is_relative_to(root / "artifacts/orders")
                or manifest.get("status") != "optimized"
            ):
                raise ValueError("No validated inventory result was produced.")
            state.update(
                status="completed",
                result_manifest=str((bundle / "manifest.json").relative_to(root)),
                updated_at=now(),
            )
        except Exception as error:
            logging.exception("Scenario %s failed", job_id)
            if isinstance(error, ValueError) and "No complete feasible plan" in str(error):
                message = (
                    "No feasible plan for this inventory, pipeline and capacity. "
                    "Increase capacity or review the supply scenario."
                )
            elif isinstance(error, ValueError) and "capacity" in str(error).lower():
                message = (
                    "This capacity cannot accommodate the common inventory and receipts. "
                    "Increase capacity and retry."
                )
            else:
                message = (
                    "The optimizer could not complete this scenario. "
                    "Check the local job log and retry."
                )
            state.update(status="failed", updated_at=now(), error=message)
        with file_lock(directory / ".submit.lock"):
            atomic_json(path, state)
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = execute_job(args.root, args.job)
    raise SystemExit(0 if result["status"] == "completed" else 1)
