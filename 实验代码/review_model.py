"""阶段6复查：核心验证、多种子参数敏感性及PCA8训练预算检查。

只读取训练/验证数据及阶段4开发记录，不读取测试数据和测试评价文件。
历史测试结果不替换；新候选仅属于同一验证集上的探索性补充。
"""
from __future__ import annotations
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss
from threadpoolctl import threadpool_limits

from bp_network import BPNetwork, TrainingConfig
from data_utils import PROJECT_DIR, DATA_DIR, load_prepared_data, load_npz
from run_experiments import validate_freeze, read_json, write_json, write_rows, sha256
from validate_bp import gradient_check, numerical_and_training_checks

OUT = PROJECT_DIR / "结果" / "模型复查"
ORIGINAL = PROJECT_DIR / "结果" / "阶段4"
WIDTHS = (16, 32, 64)
RATES = (0.05, 0.1)
SEEDS = (0, 1, 2)


def check_prediction(model, X, y, expected):
    p = model.predict_proba(X)
    np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-14)
    for key, value in {"accuracy": accuracy_score(y, p.argmax(axis=1)), "macro_f1": f1_score(y, p.argmax(axis=1), average="macro"), "loss": log_loss(y, p, labels=np.arange(10))}.items():
        np.testing.assert_allclose(value, expected[key], rtol=0, atol=1e-12)
    return p


def run_one(data, dimension, width, rate, epochs, seed):
    mode = "raw64" if dimension == 64 else "pca8"
    suffix = "centered64" if dimension == 64 else "pca8"
    run_id = f"{mode}_h{width}_lr{rate:g}_max{epochs}_seed{seed}"
    folder = OUT / "runs" / run_id
    folder.mkdir(parents=True, exist_ok=False)
    X, y = data[f"X_train_{suffix}"], data["y_train"]
    Xv, yv = data[f"X_validation_{suffix}"], data["y_validation"]
    reused = width == 16 and rate == .05 and epochs == 300
    start = perf_counter()
    if reused:
        original = ORIGINAL / "runs" / f"E3_{mode}_seed{seed}"
        model = BPNetwork.load(original / "model.npz")
        record = read_json(original / "record.json")
        with (original / "history.csv").open(encoding="utf-8-sig", newline="") as stream:
            history = list(csv.DictReader(stream))
        best, stopped = record["best_epoch"], record["stopped_epoch"]
    else:
        model = BPNetwork(dimension, width, 10, seed)
        result = model.fit(X, y, Xv, yv, TrainingConfig(learning_rate=rate, max_epochs=epochs))
        history, best, stopped = result.history, result.best_epoch, result.stopped_epoch
    elapsed = perf_counter() - start
    metrics = {"train": model.evaluate(X, y), "validation": model.evaluate(Xv, yv)}
    probs = {subset: check_prediction(model, inputs, target, metrics[subset]) for subset, inputs, target in [("train", X, y), ("validation", Xv, yv)]}
    active = np.mean(model.forward(X)["H"] > 0, axis=0)
    row = {"run_id": run_id, "input_dim": dimension, "hidden_dim": width, "learning_rate": rate, "max_epochs": epochs,
           "seed": seed, "source": "reused_original_development" if reused else "new_validation_only_fit",
           "best_epoch": best, "stopped_epoch": stopped, "hit_max_epochs": stopped == epochs,
           "n_parameters": model.n_parameters, "dead_hidden_units_train": int(np.sum(active == 0)),
           "validation_accuracy": metrics["validation"]["accuracy"], "validation_macro_f1": metrics["validation"]["macro_f1"],
           "validation_loss": metrics["validation"]["loss"], "train_accuracy": metrics["train"]["accuracy"], "train_loss": metrics["train"]["loss"],
           "measured_seconds_new_fit_only": None if reused else elapsed}
    model.save(folder / "model.npz")
    restored = BPNetwork.load(folder / "model.npz")
    np.testing.assert_array_equal(restored.predict_proba(Xv), probs["validation"])
    np.savez_compressed(folder / "predictions_train_validation.npz", probabilities_train=probs["train"], probabilities_validation=probs["validation"],
                        y_train=y, y_validation=yv, indices_train=data["indices_train"], indices_validation=data["indices_validation"])
    write_rows(folder / "history.csv", history)
    write_json(folder / "record.json", row)
    print(f"{run_id}: validation={row['validation_accuracy']:.6f}, loss={row['validation_loss']:.6f}, stopped={stopped}, reused={reused}", flush=True)
    return row


def summarize(rows):
    grouped = []
    configs = sorted({(r["input_dim"], r["hidden_dim"], r["learning_rate"], r["max_epochs"]) for r in rows})
    for dimension, width, rate, epochs in configs:
        runs = [r for r in rows if (r["input_dim"], r["hidden_dim"], r["learning_rate"], r["max_epochs"]) == (dimension, width, rate, epochs)]
        assert sorted(r["seed"] for r in runs) == list(SEEDS)
        g = {"input_dim": dimension, "hidden_dim": width, "learning_rate": rate, "max_epochs": epochs, "n_runs": 3,
             "n_parameters": runs[0]["n_parameters"], "hit_max_count": sum(r["hit_max_epochs"] for r in runs)}
        for metric in ("validation_accuracy", "validation_macro_f1", "validation_loss", "train_accuracy", "train_loss", "stopped_epoch"):
            values = np.asarray([r[metric] for r in runs])
            g[metric + "_mean"], g[metric + "_std"] = float(values.mean()), float(values.std(ddof=1))
        grouped.append(g)
    return grouped


def main():
    if OUT.exists():
        raise FileExistsError("模型复查结果已存在；保留预先声明的协议和原始记录，不自动覆盖")
    validate_freeze(ORIGINAL)
    OUT.mkdir(parents=True)
    protocol = {"purpose": "post_report_validation_only_model_review", "widths": list(WIDTHS), "learning_rates": list(RATES),
                "raw64_max_epochs": 300, "pca8_width": 16, "pca8_learning_rate": .05, "pca8_max_epochs": [300, 600],
                "seeds": list(SEEDS), "unchanged_training_options": asdict(TrainingConfig()), "total_configurations": 8,
                "total_runs": 24, "reused_runs": 6, "new_fits": 18,
                "selection": "mean validation accuracy, mean macro F1, lower mean cross entropy, fewer parameters, lower learning rate",
                "material_validation_gain_threshold": 1 / 270,
                "test_data_allowed": False, "historical_test_results_replaceable": False,
                "inference_limit": "same validation set reused, exploratory; no fresh generalization claim",
                "frozen_protocol_sha256": sha256(ORIGINAL / "frozen_protocol.json"),
                "code_sha256": {name: sha256(Path(__file__).parent / name) for name in ("review_model.py", "bp_network.py", "data_utils.py")},
                "training_validation_cache_sha256": sha256(DATA_DIR / "prepared_train_validation.npz")}
    write_json(OUT / "protocol_before_fits.json", protocol)
    data = load_prepared_data()
    assert not any("test" in k for k in data)
    with threadpool_limits(limits=1):
        gradients = gradient_check()
        checks = numerical_and_training_checks(OUT)
        write_rows(OUT / "gradient_check.csv", gradients)
        write_json(OUT / "core_checks.json", checks)
        rows = [run_one(data, 64, width, rate, 300, seed) for width in WIDTHS for rate in RATES for seed in SEEDS]
        rows += [run_one(data, 8, 16, .05, epochs, seed) for epochs in (300, 600) for seed in SEEDS]
    groups = summarize(rows)
    write_rows(OUT / "all_runs.csv", rows)
    write_rows(OUT / "validation_summary.csv", groups)
    raw = [g for g in groups if g["input_dim"] == 64]
    baseline = next(g for g in raw if g["hidden_dim"] == 16 and g["learning_rate"] == .05)
    candidate = max(raw, key=lambda g: (g["validation_accuracy_mean"], g["validation_macro_f1_mean"], -g["validation_loss_mean"], -g["n_parameters"], -g["learning_rate"]))
    gain = candidate["validation_accuracy_mean"] - baseline["validation_accuracy_mean"]
    budget = sorted([g for g in groups if g["input_dim"] == 8], key=lambda g: g["max_epochs"])
    findings = {"status": "passed", "implementation_bug_found_in_exercised_paths": False,
                "checked_gradient_parameters": sum(g["checked_entries"] for g in gradients),
                "max_gradient_absolute_error": max(g["max_absolute_error"] for g in gradients),
                "raw_baseline": baseline, "raw_validation_candidate": candidate, "raw_validation_gain_percentage_points": gain * 100,
                "material_validation_gain": gain + 1e-14 >= protocol["material_validation_gain_threshold"],
                "pca8_budget_comparison": budget,
                "pca8_budget_accuracy_gain_percentage_points": (budget[1]["validation_accuracy_mean"] - budget[0]["validation_accuracy_mean"]) * 100,
                "new_fits": sum(r["source"] == "new_validation_only_fit" for r in rows), "reused_runs": sum(r["source"] != "new_validation_only_fit" for r in rows),
                "n_test_evaluations": 0, "frozen_protocol_sha256": protocol["frozen_protocol_sha256"],
                "historical_main_results_preserved": True}
    write_json(OUT / "review_summary.json", findings)
    validate_freeze(ORIGINAL)
    immutable = {str(p.relative_to(OUT).as_posix()): sha256(p) for p in sorted(OUT.rglob("*")) if p.is_file()}
    write_json(OUT / "artifact_manifest.json", immutable)
    print(json.dumps(findings, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
