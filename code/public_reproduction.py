"""Run the public STATS19 reconstruction in an isolated workspace.

This workflow starts from the seven fixed annual collision files. It does not
use the author-side D16 comparator, the 1.53 GB all-years source, previously
fitted models, or previously saved record-level predictions.
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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from public_data import load_manifest, select_specs, verify_files


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(errors="replace")


VERSION = "PUBLIC_REPRODUCTION_V1"
SOURCE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_WORKSPACE = SOURCE_DIR.parent / f"{SOURCE_DIR.name}_public_reproduction"
STAGE_TIMEOUT_SECONDS = 4 * 60 * 60
THREAD_CAP = 4

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

ROOT_INPUTS = (
    "requirements.txt",
    "requirements-lock.txt",
)
CONFIG_INPUTS = (
    "config/public_data_manifest.json",
    "config/d10_execution_amendment.json",
    "config/d10b_tree_sensitivity_protocol.json",
    "config/d14_exclude2020_planning_evidence.json",
    "config/d14_exclude2020_protocol.json",
    "config/d14_closeout_protocol.json",
)
MUTABLE_RUNTIME_TEMPLATES = {
    "config/d10_execution_amendment.json",
    "config/d10b_tree_sensitivity_protocol.json",
}
AUXILIARY_INPUTS = ("logs/d10b_execution_incident.json",)


@dataclass(frozen=True)
class Stage:
    name: str
    arguments: tuple[str, ...]
    timeout_seconds: int = STAGE_TIMEOUT_SECONDS


PIPELINE_STAGES = (
    Stage("PUBLIC_D1_input_audit", ("code/public_input_audit.py",)),
    Stage("D2_schema_and_codes", ("code/d2_audit_schema_and_codes.py",)),
    Stage("D3_leakage_audit", ("code/d3_build_leakage_audit.py",)),
    Stage("D4_feature_table", ("code/d4_build_features.py",)),
    Stage("D5_quality_control", ("code/d5_quality_control.py",)),
    Stage("D6_protocol_and_splits", ("code/d6_freeze_protocol_and_plot.py",)),
    Stage("D7_training_inputs", ("code/d7_finalize_training_inputs.py",)),
    Stage("D8_full_baselines", ("code/d8_train_baselines.py", "--full")),
    Stage("D9_freeze_protocol", ("code/d9_train_ordered_logit.py", "--freeze")),
    Stage("D9_runtime_benchmark", ("code/d9_train_ordered_logit.py", "--benchmark")),
    Stage(
        "D9_ordered_logit_subset",
        ("code/d9_train_ordered_logit.py", "--run", "--mode", "subset"),
    ),
    Stage(
        "D9S1_freeze_protocol",
        ("code/d9s1_matched_subset_logistic.py", "--freeze"),
    ),
    Stage("D9S1_matched_subset", ("code/d9s1_matched_subset_logistic.py", "--run")),
    Stage("D10_freeze_protocol", ("code/d10_tune_lightgbm.py", "--freeze")),
    Stage(
        "D10_bind_runtime_amendment",
        ("code/public_reproduction.py", "--internal", "bind-d10"),
    ),
    Stage("D10_tune", ("code/d10_tune_lightgbm.py", "--tune")),
    Stage("D10_fit_all", ("code/d10_tune_lightgbm.py", "--fit-all")),
    Stage(
        "D10B_bind_protocol",
        ("code/public_reproduction.py", "--internal", "bind-d10b"),
    ),
    Stage("D10B_tree_sensitivity", ("code/d10b_tree_sensitivity.py",)),
    Stage(
        "D11_freeze_protocol",
        ("code/d11_evaluate_frozen_models.py", "--freeze"),
    ),
    Stage("D11_preflight", ("code/d11_evaluate_frozen_models.py", "--preflight")),
    Stage("D11_one_time_evaluation", ("code/d11_evaluate_frozen_models.py", "--run")),
    Stage(
        "D12_freeze_protocol",
        ("code/d12_bootstrap_uncertainty.py", "--freeze-protocol"),
    ),
    Stage("D12_bootstrap", ("code/d12_bootstrap_uncertainty.py", "--run")),
    Stage(
        "D13_freeze_protocol",
        ("code/d13_freeze_results.py", "--freeze-protocol"),
    ),
    Stage("D13_freeze_results", ("code/d13_freeze_results.py", "--run")),
    Stage("D14_SHAP_freeze_protocol", ("code/d14_shap_stability.py", "--freeze")),
    Stage("D14_SHAP", ("code/d14_shap_stability.py", "--run")),
    Stage(
        "D14_exclude2020_freeze_protocol",
        ("code/d14_exclude2020_sensitivity.py", "--freeze"),
    ),
    Stage("D14_exclude2020", ("code/d14_exclude2020_sensitivity.py", "--run")),
    Stage("D14_closeout_freeze_protocol", ("code/d14_closeout.py", "--freeze")),
    Stage("D14_closeout", ("code/d14_closeout.py", "--run")),
)

STANDALONE_TESTS = (
    "code/test_d6_protocol.py",
    "code/test_d7_training_inputs.py",
    "code/test_d8_baselines.py",
    "code/test_d9_ordered_logit.py",
    "code/test_d9s1_matched_subset.py",
    "code/test_d10_lightgbm.py",
    "code/test_d10b_tree_sensitivity.py",
    "code/test_d11_evaluation.py",
    "code/test_d12_bootstrap.py",
    "code/test_d13_checkpoint.py",
    "code/test_d14_shap.py",
    "code/test_d14_exclude2020.py",
    "code/test_d14_closeout.py",
    "code/test_environment.py",
)


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
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
        raise ValueError("The public reproduction workspace must be outside the source clone")
    return resolved


def verify_source_stats19(source: Path = SOURCE_DIR) -> list[str]:
    manifest = source / "config" / "public_data_manifest.json"
    _, all_specs = load_manifest(manifest)
    specs = select_specs(all_specs, ["stats19"])
    results = verify_files(source, specs)
    failures = [result for result in results if result.status != "PASS"]
    if failures:
        detail = ", ".join(
            f"{result.spec.relative_path}={result.status}" for result in failures
        )
        raise RuntimeError(f"Fixed STATS19 inputs are not ready: {detail}")
    return [spec.relative_path for spec in specs]


def source_input_paths(source: Path = SOURCE_DIR) -> list[Path]:
    paths = sorted((source / "code").glob("*.py"))
    paths.extend(source / item for item in ROOT_INPUTS)
    paths.extend(source / item for item in CONFIG_INPUTS)
    paths.extend(source / item for item in AUXILIARY_INPUTS)
    paths.extend(sorted((source / "data" / "external" / "documentation").glob("*")))
    paths.extend(source / item for item in verify_source_stats19(source))
    files = sorted({path.resolve() for path in paths if path.is_file()})
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing public reproduction input(s): " + "; ".join(missing))
    return files


def copy_or_link(source: Path, destination: Path, *, link: bool) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if link:
        try:
            os.link(source, destination)
            return "hardlink"
        except OSError:
            pass
    shutil.copy2(source, destination)
    return "copy"


def prepare_workspace(workspace: Path, source: Path = SOURCE_DIR) -> None:
    workspace = ensure_external_workspace(workspace, source)
    if workspace.exists():
        raise FileExistsError(
            f"Workspace already exists: {workspace}. Use --resume to continue it."
        )
    inputs = source_input_paths(source)
    workspace.mkdir(parents=True)
    copied: dict[str, dict[str, Any]] = {}
    try:
        for path in inputs:
            raw_path = relative(path, source)
            destination = workspace / raw_path
            method = copy_or_link(path, destination, link=False)
            copied[raw_path] = {
                "bytes": path.stat().st_size,
                "sha256": hash_file(path),
                "method": method,
            }
        for directory in (
            "data/interim",
            "data/processed",
            "figures",
            "logs/public_stage_logs",
            "models",
            "results",
        ):
            (workspace / directory).mkdir(parents=True, exist_ok=True)
        marker = {
            "version": VERSION,
            "status": "PUBLIC_WORKSPACE_PREPARED",
            "source_project": str(source.resolve()),
            "workspace": str(workspace),
            "prepared_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "copied_inputs": copied,
            "boundary": {
                "all_years_source_copied": False,
                "existing_models_copied": False,
                "existing_predictions_copied": False,
                "author_D16_artifacts_used": False,
            },
        }
        write_json_atomic(workspace / "logs" / "public_workspace.json", marker)
        verify_workspace_inputs(workspace)
    except Exception:
        marker = workspace / "logs" / "public_workspace.json"
        if marker.exists():
            marker.unlink()
        raise
    print(f"PUBLIC_WORKSPACE_PREPARED={workspace}")


def load_workspace_marker(workspace: Path) -> dict[str, Any]:
    marker_path = workspace / "logs" / "public_workspace.json"
    if not marker_path.is_file():
        raise RuntimeError("Public workspace marker is missing")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("version") != VERSION or marker.get("status") != (
        "PUBLIC_WORKSPACE_PREPARED"
    ):
        raise RuntimeError("Unexpected public workspace marker")
    if Path(str(marker.get("workspace"))).resolve() != workspace.resolve():
        raise RuntimeError("Public workspace marker path mismatch")
    return marker


def verify_workspace_inputs(workspace: Path) -> None:
    marker = load_workspace_marker(workspace)
    reference_dir = workspace / "config" / "public_reference_protocols"
    for raw_path, expected in marker["copied_inputs"].items():
        path = workspace / raw_path
        if raw_path in MUTABLE_RUNTIME_TEMPLATES and reference_dir.exists():
            archived = reference_dir / Path(raw_path).name
            if archived.is_file():
                path = archived
        if not path.is_file():
            raise FileNotFoundError(f"Prepared input missing: {path}")
        if path.stat().st_size != int(expected["bytes"]):
            raise RuntimeError(f"Prepared input size changed: {raw_path}")
        if hash_file(path) != expected["sha256"]:
            raise RuntimeError(f"Prepared input hash changed: {raw_path}")


def require_runtime_workspace(project: Path) -> None:
    marker = load_workspace_marker(project)
    if Path(str(marker["source_project"])).resolve() == project.resolve():
        raise RuntimeError("Runtime protocol binding is forbidden in the source clone")


def archive_template(project: Path, path: Path) -> Path:
    archive = project / "config" / "public_reference_protocols"
    archive.mkdir(parents=True, exist_ok=True)
    destination = archive / path.name
    if not destination.exists():
        shutil.copy2(path, destination)
    return destination


def bind_d10_amendment(project: Path) -> None:
    require_runtime_workspace(project)
    protocol_path = project / "config" / "d10_lightgbm_protocol.json"
    amendment_path = project / "config" / "d10_execution_amendment.json"
    reference = archive_template(project, amendment_path)
    protocol_hash = hash_file(protocol_path)
    payload = json.loads(reference.read_text(encoding="utf-8"))
    if payload.get("version") != "D10_EXECUTION_AMENDMENT_V1":
        raise RuntimeError("Unexpected D10 execution-amendment template")
    if payload.get("scientific_parameters_changed") is not False:
        raise RuntimeError("D10 runtime amendment changes scientific parameters")
    if int(payload.get("amended_n_jobs", -1)) != 4:
        raise RuntimeError("D10 public reconstruction requires the frozen four-thread cap")
    payload["original_protocol_sha256"] = protocol_hash
    payload["public_reproduction_instance"] = {
        "version": VERSION,
        "action": "Rebound only to the freshly generated D10 upstream hashes.",
        "reference_template_sha256": hash_file(reference),
        "scientific_parameters_changed": False,
    }
    if amendment_path.is_file():
        current = json.loads(amendment_path.read_text(encoding="utf-8"))
        reference_payload = json.loads(reference.read_text(encoding="utf-8"))
        if current not in (reference_payload, payload):
            raise RuntimeError("D10 execution amendment was modified unexpectedly")
    write_json_atomic(amendment_path, payload)
    print(f"PUBLIC_D10_AMENDMENT_BOUND={hash_file(amendment_path)}")


def bind_d10b_protocol(project: Path) -> None:
    require_runtime_workspace(project)
    protocol_path = project / "config" / "d10b_tree_sensitivity_protocol.json"
    reference = archive_template(project, protocol_path)
    payload = json.loads(reference.read_text(encoding="utf-8"))
    if payload.get("version") != "D10B_V1":
        raise RuntimeError("Unexpected D10b protocol template")
    if payload["model_lock"]["candidate_id"] != "C03":
        raise RuntimeError("D10b candidate lock changed")
    if payload["diagnostic"]["checkpoints"] != [1200, 1500, 2000]:
        raise RuntimeError("D10b checkpoint lock changed")
    mappings = {
        "D5_schema_sha256": "config/d5_dataset_schema.json",
        "D6_protocol_sha256": "config/d6_analysis_protocol.json",
        "D6_assignments_sha256": "data/processed/d6_split_assignments.csv.gz",
        "D7_training_inputs_sha256": "config/d7_training_inputs.json",
        "D10_protocol_sha256": "config/d10_lightgbm_protocol.json",
        "D10_execution_amendment_sha256": "config/d10_execution_amendment.json",
        "D10_selected_model_sha256": "config/d10_selected_lightgbm.json",
        "D10_tuning_results_sha256": "results/d10/d10_tuning_results.csv",
        "D10_source_sha256": "code/d10_tune_lightgbm.py",
        "baseline_modeling_source_sha256": "code/baseline_modeling.py",
        "modeling_data_source_sha256": "code/modeling_data.py",
    }
    for key, raw_path in mappings.items():
        if key not in payload["upstream"]:
            raise KeyError(f"D10b protocol lacks upstream key {key}")
        payload["upstream"][key] = hash_file(project / raw_path)
    payload["public_reproduction_instance"] = {
        "version": VERSION,
        "action": "Refreshed run-specific upstream hashes only.",
        "reference_template_sha256": hash_file(reference),
        "scientific_parameters_changed": False,
    }
    current = json.loads(protocol_path.read_text(encoding="utf-8"))
    reference_payload = json.loads(reference.read_text(encoding="utf-8"))
    if current not in (reference_payload, payload):
        raise RuntimeError("D10b protocol was modified unexpectedly")
    write_json_atomic(protocol_path, payload)
    print(f"PUBLIC_D10B_PROTOCOL_BOUND={hash_file(protocol_path)}")


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
        ready = workspace / "logs" / "public_environment_ready.json"
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
            log_file=workspace / "logs" / "public_stage_logs" / "00_environment.log",
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
        workspace / "logs" / "public_environment_ready.json",
        {
            "version": VERSION,
            "status": "ENVIRONMENT_READY",
            "mode": mode,
            "python": str(python),
            "python_version": version,
            "requirements_lock_sha256": hash_file(workspace / "requirements-lock.txt"),
            "completed_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
    )
    return python


def state_file(workspace: Path) -> Path:
    return workspace / "logs" / "public_run_state.json"


def load_state(workspace: Path) -> dict[str, Any]:
    path = state_file(workspace)
    if not path.is_file():
        return {
            "version": VERSION,
            "status": "RUNNING",
            "workspace": str(workspace),
            "stages": [],
        }
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("version") != VERSION:
        raise RuntimeError("Unsupported public reproduction state version")
    return state


def completed_stage_names(state: dict[str, Any]) -> set[str]:
    return {
        str(item["name"])
        for item in state.get("stages", [])
        if item.get("status") == "PASS"
    }


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
        raise RuntimeError("The workspace already has completed stages; use --resume")
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
            command = [str(python), str(workspace / stage.arguments[0]), *stage.arguments[1:]]
            try:
                result = run_process(
                    command,
                    cwd=workspace,
                    environment=environment,
                    log_file=(
                        workspace
                        / "logs"
                        / "public_stage_logs"
                        / f"{index:02d}_{stage.name}.log"
                    ),
                    timeout_seconds=stage.timeout_seconds,
                    label=stage.name,
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
            print(f"PUBLIC_REPRODUCTION_PAUSED_AFTER={stage.name}")
            return state

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
                    / "public_stage_logs"
                    / f"{len(PIPELINE_STAGES) + offset:02d}_{name}.log"
                ),
                timeout_seconds=900,
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

    state["status"] = "PIPELINE_COMPLETE"
    state["completed_local"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    write_json_atomic(state_file(workspace), state)
    print("PUBLIC_REPRODUCTION_PIPELINE=COMPLETE")
    return state


def self_test(source: Path = SOURCE_DIR) -> None:
    stage_names = [stage.name for stage in PIPELINE_STAGES]
    if len(stage_names) != len(set(stage_names)):
        raise RuntimeError("Duplicate public stage names")
    if PIPELINE_STAGES[0].arguments[0] != "code/public_input_audit.py":
        raise RuntimeError("Public pipeline does not start from the annual-input audit")
    if any("d16" in argument.lower() for stage in PIPELINE_STAGES for argument in stage.arguments):
        raise RuntimeError("Public pipeline must not invoke D16")
    required_scripts = {stage.arguments[0] for stage in PIPELINE_STAGES} | set(
        STANDALONE_TESTS
    )
    missing = sorted(path for path in required_scripts if not (source / path).is_file())
    if missing:
        raise FileNotFoundError(f"Missing public pipeline script(s): {missing}")
    files = source_input_paths(source)
    raw_paths = [relative(path, source) for path in files]
    if any("latest-published-year.csv" in Path(path).name for path in raw_paths):
        raise RuntimeError("The complete mutable all-years file entered public inputs")
    forbidden_prefixes = (
        "data/interim/",
        "data/processed/",
        "data/raw/cas/",
        "models/",
        "results/",
        "logs/d16",
        "config/d16",
    )
    forbidden = [
        path for path in raw_paths if path.lower().startswith(forbidden_prefixes)
    ]
    if forbidden:
        raise RuntimeError(f"Excluded author artifact entered public inputs: {forbidden}")
    print("PUBLIC_REPRODUCTION_SELF_TEST=PASS")
    print(f"PUBLIC_PIPELINE_STAGES={len(PIPELINE_STAGES)}")
    print(f"PUBLIC_STANDALONE_TESTS={len(STANDALONE_TESTS)}")
    print(f"PUBLIC_COPIED_INPUTS={len(files)}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--run", action="store_true")
    action.add_argument("--all", action="store_true")
    action.add_argument("--self-test", action="store_true")
    action.add_argument("--list-stages", action="store_true")
    action.add_argument(
        "--internal", choices=("bind-d10", "bind-d10b"), help=argparse.SUPPRESS
    )
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--base-python", type=Path)
    parser.add_argument(
        "--use-current-environment",
        action="store_true",
        help="Use an existing Python 3.13 environment instead of creating .venv.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", choices=[stage.name for stage in PIPELINE_STAGES])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.internal is not None:
        project = SOURCE_DIR
        if args.internal == "bind-d10":
            bind_d10_amendment(project)
        else:
            bind_d10b_protocol(project)
        return 0
    if args.list_stages:
        for index, stage in enumerate(PIPELINE_STAGES, start=1):
            print(f"{index:02d} {stage.name}")
        return 0
    if args.self_test:
        self_test()
        return 0

    workspace = ensure_external_workspace(args.workspace)
    if args.prepare:
        prepare_workspace(workspace)
        return 0
    if args.all:
        if not workspace.exists():
            prepare_workspace(workspace)
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
