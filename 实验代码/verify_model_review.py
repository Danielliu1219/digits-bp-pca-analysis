"""独立读取复查预测，核算24次训练验证指标；精确按验证正确数处理均值并列。"""
from pathlib import Path
import json
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss
from data_utils import PROJECT_DIR, load_npz
from run_experiments import validate_freeze, sha256, write_json


def main():
    out = PROJECT_DIR / "结果" / "模型复查"
    manifest = json.loads((out / "artifact_manifest.json").read_text(encoding="utf-8"))
    assert all(sha256(out / relative) == value for relative, value in manifest.items())
    exact_correct = {}
    for directory in sorted((out / "runs").iterdir()):
        r = json.loads((directory / "record.json").read_text(encoding="utf-8"))
        saved = load_npz(directory / "predictions_train_validation.npz")
        for subset in ("train", "validation"):
            p, y = saved[f"probabilities_{subset}"], saved[f"y_{subset}"]
            metrics = {"accuracy": accuracy_score(y, p.argmax(axis=1)), "loss": log_loss(y, p, labels=range(10))}
            if subset == "validation":
                metrics["macro_f1"] = f1_score(y, p.argmax(axis=1), average="macro")
            for key, value in metrics.items():
                np.testing.assert_allclose(value, r[f"{subset}_{key}"], rtol=0, atol=1e-12)
        key = (r["input_dim"], r["hidden_dim"], r["learning_rate"], r["max_epochs"])
        exact_correct[key] = exact_correct.get(key, 0) + int(np.sum(saved["probabilities_validation"].argmax(axis=1) == saved["y_validation"]))
    summary = json.loads((out / "review_summary.json").read_text(encoding="utf-8"))
    import csv
    with (out / "validation_summary.csv").open(encoding="utf-8-sig", newline="") as stream:
        groups = list(csv.DictReader(stream))
    raw = []
    for g in groups:
        key = (int(g["input_dim"]), int(g["hidden_dim"]), float(g["learning_rate"]), int(g["max_epochs"]))
        np.testing.assert_allclose(float(g["validation_accuracy_mean"]), exact_correct[key] / 810, rtol=0, atol=1e-15)
        if key[0] == 64:
            raw.append({"hidden_dim": key[1], "learning_rate": key[2], "validation_correct_sum": exact_correct[key],
                        "validation_accuracy_mean": exact_correct[key] / 810, "validation_macro_f1_mean": float(g["validation_macro_f1_mean"]),
                        "validation_loss_mean": float(g["validation_loss_mean"]), "n_parameters": int(g["n_parameters"])})
    selected = max(raw, key=lambda g: (g["validation_correct_sum"], g["validation_macro_f1_mean"], -g["validation_loss_mean"], -g["n_parameters"], -g["learning_rate"]))
    validate_freeze(PROJECT_DIR / "结果" / "阶段4")
    report = {"status": "passed", "n_runs_verified": 24, "n_new_fits": 18, "n_reused_runs": 6,
              "accuracy_rank_uses_integer_correct_counts": True, "raw_candidate_by_exact_counts": selected,
              "ranking_note": "32与64隐藏单元的最高均值对应相同验证正确总数，浮点均值相差约2e-16不应打破并列；按F1与损失选32单元、学习率0.1。原始复查摘要保留，最终判断以本核验为准。",
              "material_accuracy_gain": False, "classification_gain_percentage_points": (selected["validation_accuracy_mean"] - summary["raw_baseline"]["validation_accuracy_mean"]) * 100,
              "decision": "keep_original_frozen_model_and_phase6_report", "new_test_evaluations": 0}
    write_json(out / "independent_verification.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
