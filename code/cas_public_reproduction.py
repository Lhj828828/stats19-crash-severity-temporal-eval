"""Run the public CAS replication in an isolated workspace.

Only the fixed CAS input snapshot, tracked code/configuration, official
metadata snapshots and compact result references are copied. Author models,
record-level predictions, bootstrap arrays and SHAP arrays are never copied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import venv
from dataclasses import dataclass
from pathlib import Path
from typing import Any


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(errors="replace")


VERSION = "CAS_PUBLIC_REPRODUCTION_V1"
SOURCE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_WORKSPACE = SOURCE_DIR.parent / f"{SOURCE_DIR.name}_cas_public_reproduction"
DEFAULT_SNAPSHOT = (
    SOURCE_DIR / "data" / "raw" / "cas" / "cas_injury_2022_2025_snapshot.jsonl.gz"
)
EXPECTED_SNAPSHOT_SHA256 = (
    "7db99dd4ba92716d751dabbc08b03f72373c635025b7d5335cf0b6705a7bd7f3"
)
STAGE_TIMEOUT_SECONDS = 8 * 60 * 60
TEST_TIMEOUT_SECONDS = 30 * 60
THREAD_CAP = 8

THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": str(THREAD_CAP),
    "OPENBLAS_NUM_THREADS": str(THREAD_CAP),
    "MKL_NUM_THREADS": str(THREAD_CAP),
    "NUMEXPR_NUM_THREADS": str(THREAD_CAP),
    "VECLIB_MAXIMUM_THREADS": str(THREAD_CAP),
    "BLIS_NUM_THREADS": str(THREAD_CAP),
    "LOKY_MAX_CPU_COUNT": str(THREAD_CAP),
    "PYTHONHASHSEED": "0",
    "PYTHONIOENCODING": "utf-8",
    "MPLBACKEND": "Agg",
}


@dataclass(frozen=True)
class Stage:
    name: str
    arguments: tuple[str, ...]
    timeout_seconds: int = STAGE_TIMEOUT_SECONDS


PIPELINE_STAGES = (
    Stage(
        "CAS_D1_OFFLINE_FEASIBILITY_AUDIT",
        ("code/cas_offline_feasibility_audit.py",),
    ),
    Stage(
        "CAS_D1_NORMALIZE_SOURCE_MANIFEST",
        ("code/cas_normalize_public_manifest.py",),
    ),
    Stage(
        "CAS_D1_WRITE_FEASIBILITY_REPORT",
        ("code/cas_write_public_feasibility_report.py",),
    ),
    Stage("CAS_D2_MODELING_TABLE", ("code/cas_prepare_modeling_data.py",)),
    Stage("CAS_D3_FREEZE_SPLITS", ("code/cas_freeze_splits.py",)),
    Stage(
        "CAS_D4_FREEZE_BASELINE_PROTOCOL",
        ("code/cas_train_baselines.py", "--freeze"),
    ),
    Stage(
        "CAS_D4_FIT_BASELINES",
        ("code/cas_train_baselines.py", "--fit-validation"),
    ),
    Stage(
        "CAS_D5_FREEZE_LIGHTGBM_PROTOCOL",
        ("code/cas_train_lightgbm.py", "--freeze"),
    ),
    Stage("CAS_D5_TUNE_LIGHTGBM", ("code/cas_train_lightgbm.py", "--tune")),
    Stage(
        "CAS_D5_FIT_LIGHTGBM_ALL_SPLITS",
        ("code/cas_train_lightgbm.py", "--fit-all"),
    ),
    Stage(
        "CAS_D6_FREEZE_EVALUATION_PROTOCOL",
        ("code/cas_evaluate_frozen_models.py", "--freeze"),
    ),
    Stage(
        "CAS_D6_ONE_TIME_2025_EVALUATION",
        ("code/cas_evaluate_frozen_models.py", "--evaluate-once"),
    ),
    Stage(
        "CAS_POSTHOC_BIND_FEATURE_ABLATION_PROTOCOL",
        (
            "code/cas_public_reproduction.py",
            "--internal",
            "bind-feature-ablation-protocol",
        ),
    ),
    Stage(
        "CAS_POSTHOC_FEATURE_ABLATION_SMOKE",
        ("code/cas_feature_ablation_sensitivity.py", "--smoke"),
    ),
    Stage(
        "CAS_POSTHOC_FEATURE_ABLATION_SENSITIVITY",
        ("code/cas_feature_ablation_sensitivity.py", "--run"),
    ),
    Stage(
        "CAS_D7_FREEZE_POST_ANALYSIS_PROTOCOL",
        ("code/cas_freeze_post_analysis.py",),
    ),
    Stage(
        "CAS_D7_BOOTSTRAP_UNCERTAINTY",
        ("code/cas_bootstrap_uncertainty.py",),
    ),
    Stage("CAS_D8_SHAP_STABILITY", ("code/cas_shap_stability.py",)),
    Stage(
        "CAS_PUBLIC_BIND_REPORT_PROTOCOL",
        ("code/cas_public_reproduction.py", "--internal", "bind-report"),
    ),
    Stage("CAS_PUBLIC_CLOSEOUT", ("code/cas_finalize_replication.py",)),
    Stage(
        "CAS_PUBLIC_WRITE_WORKSPACE_MANIFEST",
        (
            "code/cas_public_reproduction.py",
            "--internal",
            "write-workspace-manifest",
        ),
    ),
)

STANDALONE_TESTS = (
    "tests/test_cas_feasibility_audit.py",
    "tests/test_cas_modeling_inputs.py",
    "tests/test_cas_baselines.py",
    "tests/test_cas_lightgbm.py",
    "tests/test_cas_evaluation.py",
    "tests/test_cas_bootstrap.py",
    "tests/test_cas_shap.py",
    "tests/test_cas_closeout.py",
    "tests/test_cas_feature_ablation.py",
)

COMPACT_REFERENCE_FILES = (
    "results/cas_baseline/validation_metrics.csv",
    "results/cas_lightgbm/tuning_results.csv",
    "results/cas_lightgbm/validation_metrics.csv",
    "results/cas_evaluation/test_metrics.csv",
    "results/cas_post_analysis/h1_across_seed_summary.csv",
    "results/cas_post_analysis/h2_bootstrap.csv",
    "results/cas_post_analysis/h3_shap_stability.csv",
    "results/cas_post_analysis/cross_dataset_directional_comparison.csv",
)


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def ensure_external_workspace(workspace: Path, source: Path = SOURCE_DIR) -> Path:
    resolved = workspace.expanduser().resolve()
    source = source.resolve()
    if resolved == source or source in resolved.parents:
        raise ValueError("The CAS public workspace must be outside the source clone")
    return resolved


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return payload


def input_map(source: Path = SOURCE_DIR) -> dict[str, Path]:
    mapping: dict[str, Path] = {}

    def add(destination: str, origin: Path) -> None:
        if not origin.is_file():
            raise FileNotFoundError(f"Missing public CAS input: {origin}")
        if destination in mapping:
            raise RuntimeError(f"Duplicate public input destination: {destination}")
        mapping[destination] = origin

    for path in sorted((source / "code").glob("cas_*.py")):
        add(f"code/{path.name}", path)
    add("code/public_result_verifier.py", source / "code" / "public_result_verifier.py")
    for path in sorted((source / "tests").glob("test_cas_*.py")):
        add(f"tests/{path.name}", path)

    for name in (
        "README.md",
        "DATA_SOURCES.md",
        "REPRODUCING.md",
        "CAS_PUBLIC_REPRODUCTION.md",
        "LICENSE",
        "CITATION.cff",
        "requirements.txt",
        "requirements-lock.txt",
        "run_cas_public_reproduction.py",
        "run_cas_public_reproduction.ps1",
        "verify_cas_public_results.py",
        "RELEASE_NOTES_v1.2.0.md",
    ):
        add(name, source / name)

    add(
        "config/cas_public_result_contract.json",
        source / "config" / "cas_public_result_contract.json",
    )
    reference_root = source / "config" / "cas_public_result_reference"
    for path in sorted(reference_root.rglob("*")):
        if path.is_file():
            add(relative(path, source), path)
    protocol_root = source / "config" / "cas_public_reference_protocols"
    for path in sorted(protocol_root.rglob("*")):
        if path.is_file():
            add(relative(path, source), path)
    add(
        "config/cas_public_reference_protocols/cas_feature_ablation_protocol.json",
        source / "config" / "cas" / "cas_feature_ablation_protocol.json",
    )

    upstream_root = source / "references" / "stats19_upstream"
    for path in sorted(upstream_root.rglob("*")):
        if path.is_file():
            add(relative(path, source), path)
    metadata_root = source / "data" / "external" / "cas"
    for path in sorted(metadata_root.glob("*.json")):
        add(relative(path, source), path)
    return mapping


def verify_snapshot(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            "Fixed CAS JSONL snapshot is missing: "
            f"{path}. Obtain the exact snapshot documented in DATA_SOURCES.md."
        )
    actual = hash_file(path)
    if actual != EXPECTED_SNAPSHOT_SHA256:
        raise RuntimeError(
            "CAS snapshot SHA-256 does not match the frozen input contract: "
            f"expected {EXPECTED_SNAPSHOT_SHA256}, observed {actual}"
        )
    return path


def copy_file(origin: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(origin, destination)


def prepare_workspace(
    workspace: Path,
    *,
    snapshot: Path,
    source: Path = SOURCE_DIR,
) -> None:
    workspace = ensure_external_workspace(workspace, source)
    if workspace.exists():
        raise FileExistsError(
            f"Workspace already exists: {workspace}. Use --resume to continue it."
        )
    snapshot = verify_snapshot(snapshot)
    files = input_map(source)
    workspace.mkdir(parents=True)
    copied: dict[str, dict[str, Any]] = {}
    try:
        for destination_name, origin in files.items():
            destination = workspace / destination_name
            copy_file(origin, destination)
            copied[destination_name] = {
                "bytes": origin.stat().st_size,
                "sha256": hash_file(origin),
                "method": "copy",
            }

        snapshot_destination = (
            workspace
            / "data"
            / "raw"
            / "cas"
            / "cas_injury_2022_2025_snapshot.jsonl.gz"
        )
        copy_file(snapshot, snapshot_destination)
        copied[relative(snapshot_destination, workspace)] = {
            "bytes": snapshot.stat().st_size,
            "sha256": hash_file(snapshot),
            "method": "copy",
        }

        for directory in (
            "data/raw/cas",
            "data/interim",
            "data/processed",
            "logs/cas",
            "logs/cas_public_stage_logs",
            "models",
            "results",
        ):
            (workspace / directory).mkdir(parents=True, exist_ok=True)

        marker = {
            "version": VERSION,
            "status": "CAS_PUBLIC_WORKSPACE_PREPARED",
            "source_project": str(source.resolve()),
            "workspace": str(workspace),
            "prepared_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "fixed_snapshot": {
                "relative_path": relative(snapshot_destination, workspace),
                "sha256": hash_file(snapshot_destination),
                "expected_sha256": EXPECTED_SNAPSHOT_SHA256,
                "source_path_at_prepare": str(snapshot),
            },
            "copied_inputs": copied,
            "boundary": {
                "stats19_records_copied": False,
                "cas_author_models_copied": False,
                "cas_record_predictions_copied": False,
                "cas_shap_arrays_copied": False,
                "cas_bootstrap_arrays_copied": False,
                "live_api_queried": False,
            },
        }
        write_json_atomic(workspace / "logs" / "cas_public_workspace.json", marker)
        verify_workspace_inputs(workspace)
    except Exception:
        marker_path = workspace / "logs" / "cas_public_workspace.json"
        if marker_path.exists():
            marker_path.unlink()
        raise
    print(f"CAS_PUBLIC_WORKSPACE_PREPARED={workspace}")


def load_workspace_marker(workspace: Path) -> dict[str, Any]:
    path = workspace / "logs" / "cas_public_workspace.json"
    if not path.is_file():
        raise RuntimeError(f"CAS public workspace marker is missing: {path}")
    marker = load_json(path)
    if marker.get("version") != VERSION or marker.get("status") != (
        "CAS_PUBLIC_WORKSPACE_PREPARED"
    ):
        raise RuntimeError("Unexpected CAS public workspace marker")
    if Path(str(marker.get("workspace"))).resolve() != workspace.resolve():
        raise RuntimeError("CAS public workspace marker path mismatch")
    return marker


def verify_workspace_inputs(workspace: Path) -> None:
    marker = load_workspace_marker(workspace)
    for raw_path, expected in marker["copied_inputs"].items():
        path = workspace / raw_path
        if not path.is_file():
            raise FileNotFoundError(f"Prepared CAS input missing: {path}")
        if path.stat().st_size != int(expected["bytes"]):
            raise RuntimeError(f"Prepared CAS input size changed: {raw_path}")
        if hash_file(path) != expected["sha256"]:
            raise RuntimeError(f"Prepared CAS input hash changed: {raw_path}")


def require_runtime_workspace(project: Path) -> dict[str, Any]:
    marker = load_workspace_marker(project)
    if Path(str(marker["source_project"])).resolve() == project.resolve():
        raise RuntimeError("Runtime binding is forbidden in the source clone")
    return marker


def venv_python(workspace: Path) -> Path:
    if os.name == "nt":
        return workspace / ".venv" / "Scripts" / "python.exe"
    return workspace / ".venv" / "bin" / "python"


def run_process(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    log_file: Path,
    timeout_seconds: int,
    label: str,
) -> dict[str, Any]:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert process.stdout is not None
    output_queue: queue.Queue[str | None] = queue.Queue()

    def read_output() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            output_queue.put(line)
        output_queue.put(None)

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    tail: list[str] = []
    stream_closed = False
    with log_file.open("w", encoding="utf-8", newline="") as log:
        log.write("COMMAND: " + subprocess.list2cmdline(command) + "\n")
        while not (stream_closed and process.poll() is not None):
            if time.perf_counter() - started > timeout_seconds:
                process.kill()
                process.wait()
                raise TimeoutError(f"{label} exceeded {timeout_seconds} seconds")
            try:
                line = output_queue.get(timeout=0.25)
            except queue.Empty:
                continue
            if line is None:
                stream_closed = True
                continue
            log.write(line)
            log.flush()
            tail.append(line)
            tail = tail[-20:]
            if line.strip():
                print(f"[{label}] {line.strip()}", flush=True)
    returncode = process.wait()
    elapsed = time.perf_counter() - started
    if returncode != 0:
        raise subprocess.CalledProcessError(
            returncode, command, output="".join(tail)
        )
    return {
        "status": "PASS",
        "command": command,
        "seconds": round(elapsed, 3),
        "log": relative(log_file, cwd),
    }


def prepare_python(
    workspace: Path,
    base_python: Path | None,
    *,
    use_current_environment: bool,
) -> Path:
    environment = os.environ.copy()
    environment.update(THREAD_ENVIRONMENT)
    if use_current_environment:
        python = (base_python or Path(sys.executable)).resolve()
        if not python.is_file():
            raise FileNotFoundError(python)
        mode = "existing_environment"
    else:
        python = venv_python(workspace)
        ready = workspace / "logs" / "cas_public_environment_ready.json"
        if python.is_file() and ready.is_file():
            return python
        candidate = (base_python or Path(sys.executable)).resolve()
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        if not python.is_file():
            print(f"[environment] creating clean virtual environment with {candidate}")
            venv.EnvBuilder(with_pip=True, symlinks=False).create(workspace / ".venv")
        run_process(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--requirement",
                str(workspace / "requirements-lock.txt"),
            ],
            cwd=workspace,
            environment=environment,
            log_file=workspace / "logs" / "cas_public_stage_logs" / "00_environment.log",
            timeout_seconds=STAGE_TIMEOUT_SECONDS,
            label="environment",
        )
        mode = "clean_virtual_environment"

    version = subprocess.check_output(
        [str(python), "-c", "import platform; print(platform.python_version())"],
        cwd=workspace,
        env=environment,
        text=True,
        encoding="utf-8",
    ).strip()
    if tuple(int(part) for part in version.split(".")[:2]) != (3, 13):
        raise RuntimeError(f"Python 3.13 is required; observed {version}")
    write_json_atomic(
        workspace / "logs" / "cas_public_environment_ready.json",
        {
            "version": VERSION,
            "status": "ENVIRONMENT_READY",
            "mode": mode,
            "python": str(python),
            "python_version": version,
            "requirements_lock_sha256": hash_file(
                workspace / "requirements-lock.txt"
            ),
            "completed_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
    )
    return python


def state_file(workspace: Path) -> Path:
    return workspace / "logs" / "cas_public_run_state.json"


def load_state(workspace: Path) -> dict[str, Any]:
    path = state_file(workspace)
    if not path.is_file():
        return {
            "version": VERSION,
            "status": "RUNNING",
            "workspace": str(workspace),
            "stages": [],
        }
    state = load_json(path)
    if state.get("version") != VERSION:
        raise RuntimeError("Unsupported CAS public reproduction state version")
    return state


def completed_stage_names(state: dict[str, Any]) -> set[str]:
    return {
        str(item["name"])
        for item in state.get("stages", [])
        if item.get("status") == "PASS"
    }


def run_recorded_stage(
    *,
    workspace: Path,
    python: Path,
    command_args: tuple[str, ...],
    label: str,
    index: int,
    timeout_seconds: int,
    environment: dict[str, str],
) -> dict[str, Any]:
    command = [str(python), str(workspace / command_args[0]), *command_args[1:]]
    return run_process(
        command,
        cwd=workspace,
        environment=environment,
        log_file=(
            workspace
            / "logs"
            / "cas_public_stage_logs"
            / f"{index:02d}_{label}.log"
        ),
        timeout_seconds=timeout_seconds,
        label=label,
    )


def execute_pipeline(
    workspace: Path,
    python: Path,
    *,
    resume: bool,
    stop_after: str | None,
) -> dict[str, Any]:
    verify_workspace_inputs(workspace)
    state = load_state(workspace)
    if state.get("stages") and not resume:
        raise RuntimeError("The workspace has completed stages; use --resume")
    if resume:
        state["status"] = "RUNNING"
        state.pop("failure", None)
    completed = completed_stage_names(state)
    environment = os.environ.copy()
    environment.update(THREAD_ENVIRONMENT)
    environment["PYTHONPATH"] = str(workspace / "code")

    stage_names = [stage.name for stage in PIPELINE_STAGES]
    if stop_after is not None and stop_after not in stage_names:
        raise ValueError(f"Unknown --stop-after stage: {stop_after}")

    for index, stage in enumerate(PIPELINE_STAGES, start=1):
        if stage.name in completed:
            print(f"[resume] skipping {stage.name}")
        else:
            try:
                result = run_recorded_stage(
                    workspace=workspace,
                    python=python,
                    command_args=stage.arguments,
                    label=stage.name,
                    index=index,
                    timeout_seconds=stage.timeout_seconds,
                    environment=environment,
                )
            except Exception as exc:
                state["status"] = "FAILED"
                state["failure"] = {
                    "stage": stage.name,
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
                write_json_atomic(state_file(workspace), state)
                raise
            state["stages"].append({"name": stage.name, **result})
            write_json_atomic(state_file(workspace), state)
        if stage.name == stop_after:
            state["status"] = "PAUSED_AFTER_REQUESTED_STAGE"
            state["paused_after"] = stage.name
            write_json_atomic(state_file(workspace), state)
            print(f"CAS_PUBLIC_REPRODUCTION_PAUSED_AFTER={stage.name}")
            return state
        completed.add(stage.name)

    completed = completed_stage_names(state)
    for offset, raw_path in enumerate(STANDALONE_TESTS, start=1):
        name = f"TEST_{Path(raw_path).stem}"
        if name in completed:
            print(f"[resume] skipping {name}")
            continue
        try:
            result = run_process(
                [str(python), str(workspace / raw_path)],
                cwd=workspace,
                environment=environment,
                log_file=(
                    workspace
                    / "logs"
                    / "cas_public_stage_logs"
                    / f"{len(PIPELINE_STAGES) + offset:02d}_{name}.log"
                ),
                timeout_seconds=TEST_TIMEOUT_SECONDS,
                label=name,
            )
        except Exception as exc:
            state["status"] = "FAILED"
            state["failure"] = {
                "stage": name,
                "type": type(exc).__name__,
                "message": str(exc),
            }
            write_json_atomic(state_file(workspace), state)
            raise
        state["stages"].append({"name": name, **result})
        write_json_atomic(state_file(workspace), state)

    verifier_name = "CAS_PUBLIC_COMPACT_RESULT_VERIFICATION"
    completed = completed_stage_names(state)
    if verifier_name in completed:
        print(f"[resume] skipping {verifier_name}")
    else:
        verifier_args = (
            "code/public_result_verifier.py",
            "--contract",
            "config/cas_public_result_contract.json",
            "--candidate-root",
            str(workspace),
            "--report",
            str(workspace / "logs" / "cas_public_result_verification.json"),
        )
        try:
            result = run_recorded_stage(
                workspace=workspace,
                python=python,
                command_args=verifier_args,
                label=verifier_name,
                index=len(PIPELINE_STAGES) + len(STANDALONE_TESTS) + 1,
                timeout_seconds=TEST_TIMEOUT_SECONDS,
                environment=environment,
            )
        except Exception as exc:
            state["status"] = "FAILED"
            state["failure"] = {
                "stage": verifier_name,
                "type": type(exc).__name__,
                "message": str(exc),
            }
            write_json_atomic(state_file(workspace), state)
            raise
        state["stages"].append({"name": verifier_name, **result})
        write_json_atomic(state_file(workspace), state)

    state["status"] = "PIPELINE_COMPLETE"
    state["completed_local"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    write_json_atomic(state_file(workspace), state)
    print("CAS_PUBLIC_REPRODUCTION_PIPELINE=COMPLETE")
    return state


def bind_report_protocol(project: Path) -> None:
    require_runtime_workspace(project)
    template_path = (
        project
        / "config"
        / "cas_public_reference_protocols"
        / "cas_cross_dataset_reporting_protocol.json"
    )
    active_path = (
        project
        / "config"
        / "cas"
        / "cas_cross_dataset_reporting_protocol.json"
    )
    template = load_json(template_path)
    if template.get("version") != "CAS_CROSS_DATASET_REPORTING_V1":
        raise RuntimeError("Unexpected CAS cross-dataset reporting template")
    if template.get("status") != "POST_RESULT_DETERMINISTIC_REPORTING_CONTRACT":
        raise RuntimeError("CAS cross-dataset reporting template is not frozen")

    cas_paths = {
        "h1_summary": "results/cas_post_analysis/h1_across_seed_summary.csv",
        "h2_bootstrap": "results/cas_post_analysis/h2_bootstrap.csv",
        "h3_complete": "config/cas/cas_h3_complete.json",
        "evaluation_complete": "config/cas/cas_evaluation_complete.json",
        "bootstrap_complete": "config/cas/cas_bootstrap_complete.json",
    }
    for name, raw_path in cas_paths.items():
        path = project / raw_path
        if not path.is_file():
            raise FileNotFoundError(f"CAS closeout input is missing: {path}")
        template["cas_upstream"][name]["path"] = raw_path
        template["cas_upstream"][name]["sha256"] = hash_file(path)

    stats19_paths = {
        "final_summary": "references/stats19_upstream/d14_final_summary.json",
        "final_comparison_protocol": (
            "references/stats19_upstream/d16_final_comparison_protocol.json"
        ),
    }
    for name, raw_path in stats19_paths.items():
        path = project / raw_path
        if not path.is_file():
            raise FileNotFoundError(f"STATS19 compact upstream is missing: {path}")
        template["stats19_upstream"][name]["path"] = raw_path
        template["stats19_upstream"][name]["sha256"] = hash_file(path)

    template["public_reproduction_instance"] = {
        "version": VERSION,
        "action": "Rebound only to freshly generated CAS outputs and local compact STATS19 references.",
        "scientific_parameters_changed": False,
        "live_api_queried": False,
    }
    write_json_atomic(active_path, template)
    print(f"CAS_PUBLIC_REPORT_PROTOCOL_BOUND={hash_file(active_path)}")


def bind_feature_ablation_protocol(project: Path) -> None:
    """Bind post-hoc sensitivity checks to regenerated public upstream files.

    The scientific settings remain frozen in the tracked template. Only hashes
    of files regenerated in this isolated workspace are rebound, because
    timestamps and runtime metadata can differ from the author's execution.
    """

    require_runtime_workspace(project)
    template_path = (
        project
        / "config"
        / "cas_public_reference_protocols"
        / "cas_feature_ablation_protocol.json"
    )
    active_path = (
        project / "config" / "cas" / "cas_feature_ablation_protocol.json"
    )
    template = load_json(template_path)
    if template.get("version") != "CAS_FEATURE_ABLATION_POSTHOC_V1":
        raise RuntimeError("Unexpected CAS feature-ablation template")
    if template.get("status") != (
        "FROZEN_AFTER_PRIMARY_RESULTS_BEFORE_POSTHOC_ABLATION_FITS"
    ):
        raise RuntimeError("CAS feature-ablation template is not frozen")
    disclosure = template.get("post_hoc_disclosure", {})
    if disclosure.get("not_preregistered") is not True:
        raise RuntimeError(
            "CAS feature-ablation template must disclose post-hoc status"
        )
    if disclosure.get("eligible_for_primary_model_selection") is not False:
        raise RuntimeError(
            "CAS feature-ablation cannot select the primary model"
        )

    for item in template["upstream_files"]:
        raw_path = str(item["file"])
        path = project / raw_path
        if not path.is_file():
            raise FileNotFoundError(
                f"CAS feature-ablation upstream is missing: {path}"
            )
        item["sha256"] = hash_file(path)

    template["public_reproduction_instance"] = {
        "version": VERSION,
        "action": "Bind only regenerated upstream hashes before post-hoc fitting.",
        "scientific_parameters_changed": False,
        "primary_models_or_results_modified": False,
        "live_api_queried": False,
    }
    write_json_atomic(active_path, template)
    print(
        "CAS_PUBLIC_FEATURE_ABLATION_PROTOCOL_BOUND="
        f"{hash_file(active_path)}"
    )


def write_workspace_manifest(project: Path) -> None:
    marker = require_runtime_workspace(project)
    config_dir = project / "config" / "cas"
    schema = load_json(config_dir / "cas_modeling_schema.json")
    analysis = load_json(config_dir / "cas_analysis_protocol.json")
    evaluation = load_json(config_dir / "cas_evaluation_complete.json")
    bootstrap = load_json(config_dir / "cas_bootstrap_complete.json")
    h3 = load_json(config_dir / "cas_h3_complete.json")
    closeout = load_json(config_dir / "cas_replication_complete.json")
    feature_ablation = load_json(
        config_dir / "cas_feature_ablation_complete.json"
    )
    raw_csv = project / "data" / "raw" / "cas" / "cas_injury_2022_2025_snapshot.csv.gz"
    processed = project / "data" / "processed" / "cas_modeling_dataset.csv.gz"
    assignments = project / "data" / "processed" / "cas_split_assignments.csv.gz"

    if closeout.get("status") != "CAS_REPLICATION_ANALYSIS_COMPLETE":
        raise RuntimeError("CAS closeout is incomplete")
    payload = {
        "version": "CAS_WORKSPACE_V2",
        "status": "CAS_REPLICATION_ANALYSIS_COMPLETE",
        "created_local": marker["prepared_local"],
        "updated_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "workspace": str(project),
        "source_project": marker["source_project"],
        "separation_policy": {
            "source_action": "public_isolated_copy_only",
            "stats19_frozen_project_modified": False,
            "records_pooled_across_datasets": False,
            "live_api_queried": False,
        },
        "cas_snapshot": {
            "years": [2022, 2023, 2024, 2025],
            "row_count": 43_121,
            "jsonl_sha256": marker["fixed_snapshot"]["sha256"],
            "csv_sha256": hash_file(raw_csv),
            "license": "CC BY 4.0 International",
        },
        "temporal_protocol": {
            "train_years": analysis["temporal_protocol"]["train_years"],
            "validation_years": analysis["temporal_protocol"]["validation_years"],
            "test_years": analysis["temporal_protocol"]["test_years"],
            "test_status": "FROZEN_MODELS_EVALUATED_ONCE_AFTER_MODEL_FREEZE",
        },
        "modeling_status": {
            "model_ready_table": True,
            "split_assignments": True,
            "models_trained": True,
            "models_frozen_before_2025_evaluation": True,
            "test_model_evaluated": True,
            "one_time_2025_evaluation": True,
            "test_feasibility_audited": True,
            "preprocessing_refit_on_test": False,
            "post_test_model_selection": False,
            "h1_h2_bootstrap_complete": bootstrap.get("status") == "H1_H2_BOOTSTRAP_COMPLETE",
            "h3_shap_complete": h3.get("status") == "H3_SHAP_COMPLETE",
            "posthoc_feature_ablation_complete": feature_ablation.get(
                "status"
            )
            == "POST_HOC_FEATURE_ABLATION_COMPLETE",
            "cross_dataset_directional_closeout": True,
        },
        "frozen_modeling_table": {
            "file": "data/processed/cas_modeling_dataset.csv.gz",
            "row_count": int(schema["row_count"]),
            "feature_count": int(schema["feature_count"]),
            "sha256": hash_file(processed),
        },
        "frozen_splits": {
            "file": "data/processed/cas_split_assignments.csv.gz",
            "sha256": hash_file(assignments),
            "temporal_rows": analysis["temporal_protocol"]["row_counts"],
            "random_rows_per_seed": analysis["random_reference_protocol"][
                "row_counts_per_seed"
            ],
        },
        "verification": {
            "public_runner": VERSION,
            "closeout_status": closeout["status"],
            "evaluation_status": evaluation["status"],
            "bootstrap_status": bootstrap["status"],
            "h3_status": h3["status"],
        },
        "claim_boundary": {
            "records_pooled_across_datasets": False,
            "absolute_metrics_compared_as_exchangeable": False,
            "cross_national_generalizability_claim_allowed": False,
            "allowed_cross_dataset_claim": (
                "workflow executability and separately reported directional agreement or disagreement"
            ),
        },
        "public_input_boundary": {
            "author_models_copied": False,
            "record_predictions_copied": False,
            "shap_arrays_copied": False,
            "bootstrap_arrays_copied": False,
        },
        "posthoc_feature_ablation": {
            "status": feature_ablation.get("status"),
            "scenarios": feature_ablation.get("scenarios", []),
            "primary_models_or_results_modified": feature_ablation.get(
                "primary_models_or_results_modified"
            ),
            "eligible_for_model_selection": feature_ablation.get(
                "eligible_for_model_selection"
            ),
        },
    }
    write_json_atomic(config_dir / "cas_workspace_manifest.json", payload)
    print("CAS_PUBLIC_WORKSPACE_MANIFEST=WRITTEN")


def self_test(source: Path = SOURCE_DIR) -> None:
    names = [stage.name for stage in PIPELINE_STAGES]
    if len(names) != len(set(names)):
        raise RuntimeError("Duplicate CAS public stage names")
    if PIPELINE_STAGES[0].arguments[0] != "code/cas_offline_feasibility_audit.py":
        raise RuntimeError("CAS public pipeline does not start with offline audit")
    if any("d16" in argument.lower() for stage in PIPELINE_STAGES for argument in stage.arguments):
        raise RuntimeError("CAS public pipeline must not invoke D16 forensic code")
    required = {stage.arguments[0] for stage in PIPELINE_STAGES}
    required.update(STANDALONE_TESTS)
    required.update({"code/public_result_verifier.py"})
    files = input_map(source)
    missing = sorted(path for path in required if not (source / path).is_file())
    if missing:
        raise FileNotFoundError(f"Missing CAS public pipeline script(s): {missing}")
    if (
        "config/cas_public_reference_protocols/cas_feature_ablation_protocol.json"
        not in files
    ):
        raise FileNotFoundError(
            "Missing public CAS feature-ablation protocol template"
        )
    forbidden = [
        path
        for path in files
        if path.startswith(("data/interim/", "data/processed/", "models/", "results/", "logs/"))
    ]
    if forbidden:
        raise RuntimeError(f"Author artifacts entered CAS public inputs: {forbidden}")
    contract = load_json(source / "config" / "cas_public_result_contract.json")
    if contract.get("schema_version") != "PUBLIC_RESULT_CONTRACT_V1":
        raise RuntimeError("CAS compact-result contract has an unexpected version")
    snapshot_status = "READY" if DEFAULT_SNAPSHOT.is_file() else "SUPPLY_FIXED_SNAPSHOT"
    print("CAS_PUBLIC_REPRODUCTION_SELF_TEST=PASS")
    print(f"CAS_PUBLIC_PIPELINE_STAGES={len(PIPELINE_STAGES)}")
    print(f"CAS_PUBLIC_STANDALONE_TESTS={len(STANDALONE_TESTS)}")
    print(f"CAS_PUBLIC_COPIED_INPUTS={len(files)}")
    print(f"CAS_PUBLIC_SNAPSHOT={snapshot_status}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--run", action="store_true")
    action.add_argument("--all", action="store_true")
    action.add_argument("--self-test", action="store_true")
    action.add_argument("--list-stages", action="store_true")
    action.add_argument(
        "--internal",
        choices=(
            "bind-report",
            "bind-feature-ablation-protocol",
            "write-workspace-manifest",
        ),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--base-python", type=Path)
    parser.add_argument(
        "--use-current-environment",
        action="store_true",
        help="Use an existing Python 3.13 environment instead of creating .venv.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--stop-after",
        choices=[stage.name for stage in PIPELINE_STAGES],
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.internal is not None:
        project = SOURCE_DIR
        if args.internal == "bind-report":
            bind_report_protocol(project)
        elif args.internal == "bind-feature-ablation-protocol":
            bind_feature_ablation_protocol(project)
        else:
            write_workspace_manifest(project)
        return 0
    if args.list_stages:
        for index, stage in enumerate(PIPELINE_STAGES, start=1):
            print(f"{index:02d} {stage.name}")
        print(f"{len(PIPELINE_STAGES) + 1:02d} TESTS_and_COMPACT_RESULT_VERIFICATION")
        return 0
    if args.self_test:
        self_test()
        return 0

    workspace = ensure_external_workspace(args.workspace)
    if args.prepare:
        snapshot = verify_snapshot(args.snapshot or DEFAULT_SNAPSHOT)
        prepare_workspace(workspace, snapshot=snapshot)
        return 0
    if args.all:
        if not workspace.exists():
            snapshot = verify_snapshot(args.snapshot or DEFAULT_SNAPSHOT)
            prepare_workspace(workspace, snapshot=snapshot)
        elif not args.resume:
            raise FileExistsError(
                f"Workspace already exists: {workspace}; use --resume or another path"
            )
    else:
        load_workspace_marker(workspace)

    python = prepare_python(
        workspace,
        args.base_python,
        use_current_environment=args.use_current_environment,
    )
    execute_pipeline(
        workspace,
        python,
        resume=args.resume,
        stop_after=args.stop_after,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
