"""独立核验阶段 4 的数据、选择规则、冻结顺序、保存预测及汇总指标。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss
from threadpoolctl import threadpool_limits

from data_utils import DATA_DIR, array_digest, load_npz
from run_experiments import (OUTPUT_DIR, C_VALUES, DIMENSIONS, SEEDS, WIDTHS,
                             predict_logits, read_json, recover_raw, sha256,
                             validate_freeze, write_json)


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def check_metrics(y: np.ndarray, probabilities: np.ndarray, expected: dict, prefix: str = "") -> None:
    if not np.isfinite(probabilities).all() or probabilities.min() < 0:
        raise AssertionError("概率存在非有限值或负数")
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=1e-13)
    predictions = probabilities.argmax(axis=1)
    independent = {
        "accuracy": accuracy_score(y, predictions),
        "macro_f1": f1_score(y, predictions, average="macro", labels=range(10), zero_division=0),
        "loss": log_loss(y, probabilities, labels=range(10)),
        "n_correct": int(np.sum(predictions == y)), "n_samples": len(y),
    }
    for name, value in independent.items():
        np.testing.assert_allclose(float(expected[prefix + name]), value, rtol=0, atol=1e-12)


def verify(output: Path) -> dict:
    frozen = validate_freeze(output)
    freeze_digest = sha256(output / "frozen_protocol.json")
    # 这是冻结后结果核验，允许读取原始全体快照，以独立验证像素还原和标签对应。
    raw = load_npz(DATA_DIR / "digits_raw.npz")
    splits = load_npz(DATA_DIR / "split_indices.npz")
    assert array_digest(raw["X"], raw["y"]) == frozen["dataset_content_sha256"]
    assert array_digest(*[splits[name] for name in ("train", "validation", "test")]) == frozen["split_content_sha256"]
    train_cache = load_npz(DATA_DIR / "prepared_train_validation.npz")
    source_state = load_npz(DATA_DIR / "preprocessor.npz")
    for subset in ("train", "validation"):
        restored = recover_raw(train_cache[f"X_{subset}_centered64"], source_state["mean"])
        np.testing.assert_array_equal(restored, raw["X"][splits[subset]])
        np.testing.assert_array_equal(train_cache[f"y_{subset}"], raw["y"][splits[subset]])

    records = [read_json(output / "runs" / run_id / "record.json") for run_id in
               [f"E1_h{h}_seed42" for h in WIDTHS] + [f"E2_C{C:g}" for C in C_VALUES] +
               [f"E3_{'raw64' if d == 64 else 'pca' + str(d)}_seed{s}" for d in DIMENSIONS for s in SEEDS]]
    assert len(records) == 18
    assert len(read_rows(output / "development_results.csv")) == 18
    chosen_bp = max((row for row in records if row["experiment"] == "E1"),
                    key=lambda row: (row["validation_accuracy"], row["validation_macro_f1"], -row["hidden_dim"]))
    chosen_lr = max((row for row in records if row["experiment"] == "E2"),
                    key=lambda row: (row["validation_accuracy"], row["validation_macro_f1"], -row["regularization_C"]))
    assert chosen_bp["hidden_dim"] == frozen["selected_hidden_dim"]
    assert chosen_lr["regularization_C"] == frozen["selected_C"]
    ranked_pca = []
    for dimension in (8, 16, 32):
        rows = [row for row in records if row["experiment"] == "E3" and row["input_dim"] == dimension]
        assert sorted(row["initialization_seed"] for row in rows) == list(SEEDS)
        ranked_pca.append((np.mean([row["validation_accuracy"] for row in rows]),
                           np.mean([row["validation_macro_f1"] for row in rows]), -dimension))
    assert -max(ranked_pca)[2] == frozen["recommended_pca_dimension_from_validation"]

    X_train_scaled = raw["X"][splits["train"]] / 16.0
    mean = X_train_scaled.mean(axis=0)
    centered = X_train_scaled - mean
    covariance = centered.T @ centered / (len(centered) - 1)
    max_probability_error = 0.0
    for row in records:
        directory = output / "runs" / row["run_id"]
        state = load_npz(directory / "preprocessor.npz")
        saved = load_npz(directory / "predictions_train_validation.npz")
        np.testing.assert_allclose(state["mean"], mean, rtol=0, atol=1e-15)
        assert int(state["fit_n_samples"]) == 1257
        dimension = row["input_dim"]
        if dimension != 64:
            vectors = state["components"]
            assert vectors.shape == (dimension, 64)
            np.testing.assert_allclose(vectors @ vectors.T, np.eye(dimension), rtol=0, atol=1e-12)
            np.testing.assert_allclose(covariance @ vectors.T, vectors.T * state["explained_variance"], rtol=0, atol=1e-12)
        assert row["pipeline_parameter_count"] == row["n_parameters"] + state["mean"].size + state["components"].size
        assert row["preprocessing_seconds"] > 0 and row["model_training_seconds"] > 0
        np.testing.assert_allclose(row["pipeline_training_seconds"], row["preprocessing_seconds"] + row["model_training_seconds"], rtol=0, atol=1e-14)
        if row["experiment"] == "E3":
            assert row["hidden_dim"] == frozen["selected_hidden_dim"]
            assert row["shuffle_seed"] == row["initialization_seed"] + 1000
        for subset in ("train", "validation"):
            indices = splits[subset]
            np.testing.assert_array_equal(saved[f"indices_{subset}"], indices)
            np.testing.assert_array_equal(saved[f"y_{subset}"], raw["y"][indices])
            X = raw["X"][indices] / 16.0 - mean
            if dimension != 64:
                X = X @ state["components"].T
            logits = predict_logits(row["model_kind"], directory / "model.npz", X)
            shifted = logits - logits.max(axis=1, keepdims=True)
            probabilities = np.exp(shifted)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            max_probability_error = max(max_probability_error, float(np.abs(probabilities - saved[f"probabilities_{subset}"]).max()))
            np.testing.assert_allclose(probabilities, saved[f"probabilities_{subset}"], rtol=0, atol=1e-13)
            np.testing.assert_array_equal(saved[f"predictions_{subset}"], probabilities.argmax(axis=1))
            check_metrics(raw["y"][indices], saved[f"probabilities_{subset}"], row, prefix=subset + "_")
        if row["model_kind"] == "bp":
            history = read_rows(directory / "history.csv")
            checkpoint = history[row["best_epoch"]]
            assert int(checkpoint["epoch"]) == row["best_epoch"] and checkpoint["checkpoint_saved"] == "1"
            assert len(history) == row["stopped_epoch"] + 1
            np.testing.assert_allclose(float(checkpoint["validation_loss"]), row["validation_loss"], rtol=0, atol=1e-13)

    test_rows = read_rows(output / "test_results.csv")
    assert len(test_rows) == 13
    assert {row["run_id"] for row in test_rows} == set(frozen["evaluation_run_ids"])
    record_map = {row["run_id"]: row for row in records}
    for row in test_rows:
        directory = output / "runs" / row["run_id"]
        prediction = load_npz(directory / "predictions_test.npz")
        np.testing.assert_array_equal(prediction["indices"], splits["test"])
        np.testing.assert_array_equal(prediction["y"], raw["y"][splits["test"]])
        np.testing.assert_array_equal(prediction["predictions"], prediction["probabilities"].argmax(axis=1))
        check_metrics(prediction["y"], prediction["probabilities"], row, prefix="test_")
        saved_metrics = read_json(directory / "test_metrics.json")
        assert saved_metrics["frozen_protocol_sha256"] == freeze_digest
        check_metrics(prediction["y"], prediction["probabilities"], saved_metrics)
        # 用保存模型和原始测试像素重新推理，核对预测文件确实来自冻结模型。
        record = record_map[row["run_id"]]
        state = load_npz(directory / "preprocessor.npz")
        X = raw["X"][splits["test"]] / 16.0 - state["mean"]
        if record["input_dim"] != 64:
            X = X @ state["components"].T
        logits = predict_logits(record["model_kind"], directory / "model.npz", X)
        shifted = logits - logits.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(probabilities, prediction["probabilities"], rtol=0, atol=1e-13)

    grouped = read_rows(output / "group_summary.csv")
    assert len(grouped) == 4
    for group in grouped:
        rows = [row for row in test_rows if row["experiment"] == "E3" and row["input_dim"] == group["input_dim"]]
        assert len(rows) == 3
        for key in ("test_accuracy", "test_macro_f1", "test_loss", "validation_accuracy", "validation_macro_f1", "pipeline_training_seconds"):
            values = np.asarray([float(row[key]) for row in rows])
            np.testing.assert_allclose(float(group[key + "_mean"]), values.mean(), rtol=0, atol=1e-14)
            np.testing.assert_allclose(float(group[key + "_std"]), values.std(ddof=1), rtol=0, atol=1e-14)
    summary = read_json(output / "experiment_summary.json")
    assert summary["frozen_protocol_sha256"] == freeze_digest
    assert summary["recommended_pca_dimension_from_validation"] == frozen["recommended_pca_dimension_from_validation"]
    for group in summary["bp_groups"]:
        csv_group = next(row for row in grouped if int(row["input_dim"]) == group["input_dim"])
        for name, value in group.items():
            if isinstance(value, (int, float)):
                np.testing.assert_allclose(float(csv_group[name]), value, rtol=0, atol=1e-14)

    import json
    events = [json.loads(line) for line in (output / "execution_log.jsonl").read_text(encoding="utf-8").splitlines()]
    fit_positions = [i for i, row in enumerate(events) if row["action"] == "fit_completed"]
    frozen_position = next(i for i, row in enumerate(events) if row["action"] == "protocol_frozen")
    test_positions = [i for i, row in enumerate(events) if row["action"] == "test_loaded"]
    assert len(fit_positions) == 18 and max(fit_positions) < frozen_position < min(test_positions)
    assert events[frozen_position]["frozen_protocol_sha256"] == freeze_digest
    assert all(events[i]["frozen_protocol_sha256"] == freeze_digest for i in test_positions)
    validate_freeze(output)
    result = {
        "status": "passed", "development_fits_verified": 18, "test_models_verified": 13,
        "fixed_raw_data_and_split_fingerprints_match": True,
        "raw_pixel_recovery_exactly_matches_original_snapshot": True,
        "training_only_preprocessing_mean_and_covariance_verified": True,
        "selection_rules_recomputed_from_validation_only": True,
        "all_fits_precede_freeze_and_test_load": True,
        "frozen_code_models_and_development_records_unchanged": True,
        "predictions_match_saved_models_and_original_labels": True,
        "independent_accuracy_macro_f1_cross_entropy_verified": True,
        "bp_summary_uses_all_three_seeds_and_sample_std": True,
        "max_development_probability_recomputation_error": max_probability_error,
        "frozen_protocol_sha256": freeze_digest,
    }
    write_json(output / "verification_report.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        result = verify(args.output_dir.resolve())
    import json
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
