"""Independent assertions for the D12 bootstrap outputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from baseline_modeling import classification_metrics
from d12_bootstrap_uncertainty import (
    BOOTSTRAP_ITERATIONS,
    BOOTSTRAP_SEED,
    CANDIDATE_MODEL,
    CI_ALPHA,
    DRAW_FILE,
    DRAW_METADATA_FILE,
    GAIN_SUMMARY_FILE,
    HIGHER_IS_BETTER,
    LOWER_IS_BETTER,
    MANIFEST_FILE,
    METRICS,
    MODEL_IDS,
    MODEL_INTERVAL_FILE,
    OPTIMISM_FILE,
    OPTIMISM_SUMMARY_FILE,
    PAIRWISE_FILE,
    PROJECT_DIR,
    PROTOCOL_FILE,
    RANDOM_SEEDS,
    REFERENCE_MODEL,
    SUMMARY_FILE,
    confusion_from_vectors,
    metrics_from_confusions,
)


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def q(values: np.ndarray) -> tuple[float, float]:
    result = np.quantile(values, [CI_ALPHA / 2, 1 - CI_ALPHA / 2], method="linear")
    return float(result[0]), float(result[1])


def assert_metric_kernel() -> None:
    rng = np.random.default_rng(65537)
    target = np.repeat(np.arange(3), [2_000, 400, 80]).astype(np.int8)
    for _ in range(10):
        predicted = rng.integers(0, 3, size=len(target), dtype=np.int8)
        matrix = confusion_from_vectors(target, predicted[:, None])
        expected = classification_metrics(target, predicted)
        actual = metrics_from_confusions(matrix)
        for metric in METRICS:
            assert np.isclose(expected[metric], actual[metric][0], rtol=0, atol=1e-14), metric


def assert_d11_unchanged(protocol: dict) -> None:
    for name, expected_hash in protocol["parent_artifacts"].items():
        assert hash_file(PROJECT_DIR / name) == expected_hash, name
    manifest = pd.read_csv(PROJECT_DIR / "logs" / "d11_artifact_manifest.csv")
    assert len(manifest) == 64
    assert manifest["relative_path"].str.startswith("results/d11/predictions/").sum() == 55
    for row in manifest.itertuples(index=False):
        path = PROJECT_DIR / row.relative_path
        assert path.stat().st_size == int(row.bytes)
        assert hash_file(path) == row.sha256


def main() -> None:
    protocol = json.loads(PROTOCOL_FILE.read_text(encoding="utf-8"))
    assert protocol["bootstrap"]["iterations"] == BOOTSTRAP_ITERATIONS == 2_000
    assert protocol["bootstrap"]["master_seed"] == BOOTSTRAP_SEED == 20_260_828
    assert protocol["scientific_lock"]["model_fits"] == 0
    assert protocol["scientific_lock"]["D11_reruns"] == 0
    assert len(protocol["evaluation_groups"]) == 11
    assert set(protocol["metrics"]["higher_is_better"]) == HIGHER_IS_BETTER
    assert set(protocol["metrics"]["lower_is_better"]) == LOWER_IS_BETTER
    assert_d11_unchanged(protocol)
    assert_metric_kernel()

    intervals = pd.read_csv(MODEL_INTERVAL_FILE)
    pairwise = pd.read_csv(PAIRWISE_FILE)
    optimism = pd.read_csv(OPTIMISM_FILE)
    optimism_summary = pd.read_csv(OPTIMISM_SUMMARY_FILE)
    gain_summary = pd.read_csv(GAIN_SUMMARY_FILE)
    metadata = json.loads(DRAW_METADATA_FILE.read_text(encoding="utf-8"))
    summary = json.loads(SUMMARY_FILE.read_text(encoding="utf-8"))

    assert len(intervals) == 11 * len(MODEL_IDS) * len(METRICS) == 385
    assert len(pairwise) == 11 * len(METRICS) == 77
    assert len(optimism) == len(RANDOM_SEEDS) * len(MODEL_IDS) * len(METRICS) == 175
    assert len(optimism_summary) == len(MODEL_IDS) * len(METRICS) == 35
    assert len(gain_summary) == len(METRICS) == 7
    assert intervals.duplicated(["group_key", "model", "metric"]).sum() == 0
    assert pairwise.duplicated(["group_key", "metric"]).sum() == 0
    assert optimism.duplicated(["seed", "model", "metric"]).sum() == 0

    arrays = np.load(DRAW_FILE, allow_pickle=False)
    assert arrays["model_ids"].tolist() == list(MODEL_IDS)
    assert arrays["metric_ids"].tolist() == list(METRICS)
    reverse_map = {group_key: array_key for array_key, group_key in metadata["draw_key_map"].items()}
    assert len(reverse_map) == 11
    for group_key, array_key in reverse_map.items():
        draws = arrays[array_key]
        assert draws.shape == (BOOTSTRAP_ITERATIONS, len(MODEL_IDS), len(METRICS))
        assert np.isfinite(draws).all()
        subset = intervals[intervals["group_key"].eq(group_key)]
        for row in subset.itertuples(index=False):
            model_index = MODEL_IDS.index(row.model)
            metric_index = METRICS.index(row.metric)
            lower, upper = q(draws[:, model_index, metric_index])
            assert np.isclose(lower, row.ci_lower, rtol=0, atol=1e-14)
            assert np.isclose(upper, row.ci_upper, rtol=0, atol=1e-14)

    temporal_key = reverse_map["temporal_test_2024"]
    temporal_draws = arrays[temporal_key]
    candidate_index = MODEL_IDS.index(CANDIDATE_MODEL)
    reference_index = MODEL_IDS.index(REFERENCE_MODEL)
    temporal_rows = pairwise[pairwise["group_key"].eq("temporal_test_2024")]
    for row in temporal_rows.itertuples(index=False):
        metric_index = METRICS.index(row.metric)
        raw = temporal_draws[:, candidate_index, metric_index] - temporal_draws[:, reference_index, metric_index]
        lower, upper = q(raw)
        assert np.isclose(lower, row.raw_delta_ci_lower, rtol=0, atol=1e-14)
        assert np.isclose(upper, row.raw_delta_ci_upper, rtol=0, atol=1e-14)

    for row in optimism.itertuples(index=False):
        internal_key = reverse_map[f"random_seed_{row.seed}__internal_test"]
        future_key = reverse_map[f"random_seed_{row.seed}__2024_diagnostic"]
        model_index = MODEL_IDS.index(row.model)
        metric_index = METRICS.index(row.metric)
        sign = 1.0 if row.metric in HIGHER_IS_BETTER else -1.0
        gap = sign * (
            arrays[internal_key][:, model_index, metric_index]
            - arrays[future_key][:, model_index, metric_index]
        )
        lower, upper = q(gap)
        assert np.isclose(lower, row.optimism_gap_ci_lower, rtol=0, atol=1e-14)
        assert np.isclose(upper, row.optimism_gap_ci_upper, rtol=0, atol=1e-14)

    stream_entropies = [tuple(item["entropy"]) for item in metadata["streams"]]
    assert len(stream_entropies) == len(set(stream_entropies)) == 11
    assert summary["fit_operations"] == 0
    assert summary["threshold_changes"] == 0
    assert summary["model_selection_operations"] == 0

    manifest = pd.read_csv(MANIFEST_FILE)
    assert len(manifest) == 15
    for row in manifest.itertuples(index=False):
        path = PROJECT_DIR / row.relative_path
        assert path.stat().st_size == int(row.bytes)
        assert hash_file(path) == row.sha256

    print("INDEPENDENT_D12_ASSERTIONS=PASS")
    print(f"BOOTSTRAP_DRAWS={11 * BOOTSTRAP_ITERATIONS}")
    print(f"MODEL_METRIC_VALUES={11 * BOOTSTRAP_ITERATIONS * len(MODEL_IDS) * len(METRICS)}")


if __name__ == "__main__":
    main()
