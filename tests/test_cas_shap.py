"""Independent checks for frozen CAS H3 SHAP artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "code"))

from cas_modeling_common import (  # noqa: E402
    RANDOM_SEEDS,
    TARGET_CODES,
    hash_file,
    load_aligned_data,
    load_contracts,
)
from cas_shap_stability import (  # noqa: E402
    CLASS_DIRECTION_FILE,
    CLASS_LABELS,
    COMPLETE_FILE,
    IMPORTANCE_FILE,
    LIGHTGBM_FROZEN_FILE,
    MANIFEST_FILE,
    RAW_SHAP_FILE,
    SAMPLE_AUDIT_FILE,
    SAMPLE_FILE,
    STABILITY_FILE,
    TEST_FILE,
    hash_positions,
    matched_positions,
)


PROTOCOL_FILE = (
    PROJECT_DIR / "config" / "cas" / "cas_post_analysis_protocol.json"
)
BOOTSTRAP_DRAWS_FILE = (
    PROJECT_DIR / "results" / "cas_post_analysis" / "h3_shap_bootstrap_draws.csv.gz"
)


def main() -> None:
    protocol = json.loads(PROTOCOL_FILE.read_text(encoding="utf-8"))
    complete = json.loads(COMPLETE_FILE.read_text(encoding="utf-8"))
    assert protocol["version"] == "CAS_POST_ANALYSIS_V1"
    assert protocol["common"]["iterations"] == 2_000
    assert protocol["common"]["base_seed"] == 20_260_901
    assert complete["status"] == "H3_SHAP_COMPLETE"
    assert complete["protocol_sha256"] == hash_file(PROTOCOL_FILE)
    assert complete["model_fits"] == 0
    assert complete["post_test_model_selection"] is False
    assert complete["causal_claim_allowed"] is False
    assert complete["raw_shap_shape"] == [5, 2, 4_887, 3, 15]

    stability = pd.read_csv(STABILITY_FILE)
    importance = pd.read_csv(IMPORTANCE_FILE)
    direction = pd.read_csv(CLASS_DIRECTION_FILE)
    samples = pd.read_csv(SAMPLE_FILE, dtype={"meta_crash_id": "string"})
    audit = pd.read_csv(SAMPLE_AUDIT_FILE)
    draws = pd.read_csv(BOOTSTRAP_DRAWS_FILE)
    raw = np.load(RAW_SHAP_FILE, allow_pickle=False, mmap_mode="r")

    assert len(stability) == 5
    assert len(importance) == 5 * 2 * 15
    assert len(direction) == 5 * 2 * 3 * 15
    assert len(samples) == 5 * 2 * 4_887
    assert len(audit) == 5 * 2 * 3
    assert len(draws) == 5 * 2_000
    assert raw.shape == (5, 2, 4_887, 3, 15)
    assert np.isfinite(raw).all()
    assert np.isfinite(
        stability[["spearman_rho", "ci_lower", "ci_upper"]].to_numpy(float)
    ).all()
    assert (stability["ci_lower"] <= stability["ci_upper"]).all()
    assert draws.groupby("seed").size().eq(2_000).all()
    assert draws["iteration"].between(1, 2_000).all()
    schema, analysis, _training = load_contracts()
    _features, target, metadata, assignments = load_aligned_data(
        schema, analysis
    )
    feature_names = list(schema["feature_columns"])
    assert len(feature_names) == 15
    for seed_index, seed in enumerate(RANDOM_SEEDS):
        internal, future, counts = matched_positions(
            assignments=assignments,
            target=target,
            seed=seed,
            base_seed=int(protocol["common"]["base_seed"]),
        )
        for cohort_index, (cohort, positions) in enumerate(
            (
                ("random_internal_test", internal),
                ("matched_2025", future),
            )
        ):
            subset = samples.loc[
                samples["seed"].eq(seed)
                & samples["explanation_cohort"].eq(cohort)
            ].sort_values("sample_order")
            assert len(subset) == 4_887
            assert subset["row_position"].to_numpy(int).tolist() == positions.tolist()
            assert hash_positions(positions) == str(
                audit.loc[
                    audit["seed"].eq(seed)
                    & audit["explanation_cohort"].eq(cohort),
                    "row_positions_sha256",
                ].iloc[0]
            )
            assert subset["meta_crash_id"].is_unique
            assert subset["target_severity"].value_counts().to_dict() == counts
            if cohort == "random_internal_test":
                assert set(subset["meta_crash_year"]) <= {2022, 2023, 2024}
            else:
                assert set(subset["meta_crash_year"]) == {2025}
                assert subset["selection_seed"].eq(
                    str(int(protocol["common"]["base_seed"]) + seed)
                ).all()
            raw_subset = np.asarray(raw[seed_index, cohort_index])
            abs_mean = np.abs(raw_subset).mean(axis=(0, 1))
            expected_ranks = rankdata(-abs_mean, method="average")
            importance_subset = importance.loc[
                importance["seed"].eq(seed)
                & importance["explanation_cohort"].eq(cohort)
            ].set_index("feature").loc[feature_names]
            assert np.allclose(
                importance_subset["mean_abs_shap"].to_numpy(float),
                abs_mean,
                rtol=0,
                atol=1e-12,
            )
            assert np.allclose(
                importance_subset["importance_rank"].to_numpy(float),
                expected_ranks,
                rtol=0,
                atol=1e-12,
            )
            assert subset["target_severity"].to_numpy(int).tolist() == target.iloc[
                positions
            ].to_numpy(int).tolist()

        internal_importance = np.abs(
            np.asarray(raw[seed_index, 0])
        ).mean(axis=(0, 1))
        future_importance = np.abs(
            np.asarray(raw[seed_index, 1])
        ).mean(axis=(0, 1))
        expected_rho = float(
            spearmanr(internal_importance, future_importance).statistic
        )
        row = stability.loc[stability["seed"].eq(seed)].iloc[0]
        assert np.isclose(float(row["spearman_rho"]), expected_rho, atol=1e-12)
        seed_draws = draws.loc[draws["seed"].eq(seed)].sort_values("iteration")
        lower, upper = np.quantile(
            seed_draws["spearman_rho"].to_numpy(float), [0.025, 0.975]
        )
        assert np.isclose(float(row["ci_lower"]), lower, atol=1e-12)
        assert np.isclose(float(row["ci_upper"]), upper, atol=1e-12)

    assert set(direction["output_class_code"]) == set(TARGET_CODES)
    assert set(direction["interpretation"]) == {
        "raw_class_score_attribution_not_causal"
    }
    assert set(audit["target_label"]) == set(CLASS_LABELS.values())
    assert hash_file(LIGHTGBM_FROZEN_FILE) == protocol["upstream"][
        "lightgbm_models_frozen_sha256"
    ]
    manifest = pd.read_csv(MANIFEST_FILE)
    assert manifest["relative_path"].is_unique
    for _, row in manifest.iterrows():
        path = PROJECT_DIR / str(row["relative_path"])
        assert path.exists()
        assert path.stat().st_size == int(row["bytes"])
        assert hash_file(path) == row["sha256"]
    assert str(TEST_FILE.relative_to(PROJECT_DIR).as_posix()) in set(
        manifest["relative_path"]
    )
    print("CAS H3 stability rows:", len(stability))
    print("CAS H3 raw SHAP shape:", tuple(raw.shape))
    print("CAS_H3_INDEPENDENT_CHECKS=PASS")


if __name__ == "__main__":
    main()
