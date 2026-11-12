"""追加固定近邻/RBF-SVM基线与配对统计；不改写原冻结实验。"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
from time import perf_counter
import warnings

import numpy as np
from scipy.stats import binomtest, norm
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from data_utils import PROJECT_DIR, DATA_DIR, load_npz, load_prepared_data
from run_experiments import validate_freeze

OUT = PROJECT_DIR / "结果/补充实验_20261111"
C_GRID = (0.1, 1.0, 10.0)
GAMMA_GRID = ("scale", 0.01, 0.1, 1.0)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def rows(path, values):
    keys = list(dict.fromkeys(k for row in values for k in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, keys)
        writer.writeheader()
        writer.writerows(values)


def wilson(correct, total):
    z = float(norm.ppf(0.975))
    p = correct / total
    denominator = 1 + z*z/total
    center = (p + z*z/(2*total)) / denominator
    radius = z*np.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denominator
    return float(center-radius), float(center+radius)


def paired(first, second, y):
    a, b = first == y, second == y
    only_first = int(np.sum(a & ~b))
    only_second = int(np.sum(~a & b))
    discordant = only_first + only_second
    p = float(binomtest(only_first, discordant, 0.5, alternative="two-sided").pvalue) if discordant else 1.0
    return {"both_correct": int(np.sum(a & b)), "only_first_correct": only_first,
            "only_second_correct": only_second, "both_wrong": int(np.sum(~a & ~b)),
            "n_discordant": discordant, "p_exact_two_sided": p,
            "accuracy_difference_pp": float((a.mean()-b.mean())*100)}


def score(label, model, X, y):
    started = perf_counter()
    prediction = model.predict(X)
    elapsed = perf_counter()-started
    correct = int(np.sum(prediction == y))
    low, high = wilson(correct, len(y))
    return {"model": label, "n_samples": len(y), "n_correct": correct,
            "accuracy": float(accuracy_score(y, prediction)),
            "macro_f1": float(f1_score(y, prediction, average="macro")),
            "wilson_low": low, "wilson_high": high, "prediction_seconds": elapsed}, prediction


def make_model(kind, config):
    if kind == "knn":
        return KNeighborsClassifier(n_neighbors=config["k"], weights="uniform", algorithm="brute", metric="euclidean", n_jobs=1)
    return SVC(C=config["C"], gamma=config["gamma"], kernel="rbf", tol=1e-6, max_iter=-1)


def run():
    if OUT.exists():
        raise FileExistsError(f"追加实验目录已存在：{OUT}")
    OUT.mkdir(parents=True)
    (OUT / "source_at_fit.py").write_bytes(Path(__file__).read_bytes())
    frozen = validate_freeze(PROJECT_DIR / "结果/阶段4")
    protocol = {"created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
                "analysis_kind": "supplement_after_original_test_report", "knn_k": [1, 3],
                "knn_settings": "Euclidean uniform brute force; n_jobs=1",
                "svm_C_grid": C_GRID, "svm_gamma_grid": GAMMA_GRID,
                "selection": "validation accuracy, validation macro F1, smaller C, gamma grid order",
                "training": "1257 training observations only; fixed original 270 validation and 270 test",
                "preprocessing": "original training mean and /16 centered64; no feature standardization",
                "primary_comparisons": ["BP64 seed0 vs LR", "PCA32 seed0 vs BP64 seed0"],
                "secondary_seed_checks": [1, 2], "wilson_confidence": 0.95,
                "mcnemar": "exact two-sided conditional binomial test, p=0.5 on discordant pairs",
                "frozen_protocol_sha256": sha(PROJECT_DIR / "结果/阶段4/frozen_protocol.json"),
                "data_sha256": {name: sha(DATA_DIR/name) for name in ("prepared_train_validation.npz", "prepared_test.npz")},
                "source_sha256": sha(__file__), "test_read_in_this_run": False}
    write(OUT / "protocol_before_fits.json", protocol)
    data = load_prepared_data()
    X, y = data["X_train_centered64"], data["y_train"]
    Xv, yv = data["X_validation_centered64"], data["y_validation"]
    validation = []
    candidates = []
    fitted = []
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        for k in (1, 3):
            config = {"k": k}
            model = make_model("knn", config)
            started = perf_counter(); model.fit(X, y); elapsed = perf_counter()-started
            metrics, pred = score(f"kNN k={k}", model, Xv, yv)
            record = {**metrics, "kind": "knn", "config": config, "fit_seconds": elapsed}
            validation.append(record)
            fitted.append((record, model, pred))
        for C in C_GRID:
            for gamma_index, gamma in enumerate(GAMMA_GRID):
                model = make_model("svm", {"C": C, "gamma": gamma})
                started = perf_counter(); model.fit(X, y); elapsed = perf_counter()-started
                metrics, pred = score("RBF-SVM", model, Xv, yv)
                record = {**metrics, "kind": "svm", "config": {"C": C, "gamma": gamma},
                          "gamma_grid_index": gamma_index, "fit_seconds": elapsed}
                candidates.append((record, model, pred))
                validation.append(record)
        chosen = max(candidates, key=lambda item: (item[0]["accuracy"], item[0]["macro_f1"], -item[0]["config"]["C"], -item[0]["gamma_grid_index"]))
        fitted.append(chosen)
        rows(OUT / "validation_candidates.csv", validation)
        selection = {"status": "selected_before_current_test_read", "selected_svm_config": chosen[0]["config"],
                     "selected_svm_validation_accuracy": chosen[0]["accuracy"],
                     "selected_svm_validation_f1": chosen[0]["macro_f1"], "original_test_already_reported": True}
        write(OUT / "selection_before_test.json", selection)
        test = load_npz(DATA_DIR / "prepared_test.npz")
        Xt, yt = test["X_test_centered64"], test["y_test"]
        metrics_rows, predictions = [], {}
        for number, (record, model, pred_val) in enumerate(fitted):
            metrics, pred = score(record["model"], model, Xt, yt)
            metrics.update(kind=record["kind"], config=record["config"], validation_accuracy=record["accuracy"],
                           validation_macro_f1=record["macro_f1"], fit_seconds=record["fit_seconds"])
            metrics_rows.append(metrics)
            predictions[record["model"]] = pred
            directory = OUT / f"baseline_{number}"
            directory.mkdir()
            write(directory / "config.json", record)
            np.savez_compressed(directory / "predictions.npz", predictions=pred, y=yt, indices=test["indices_test"],
                                predictions_validation=pred_val, y_validation=yv, indices_validation=data["indices_validation"])
            if record["kind"] == "svm":
                np.savez_compressed(directory / "support_state.npz", support_vectors=model.support_vectors_, support_indices=model.support_,
                                    dual_coef=model.dual_coef_, intercept=model.intercept_, n_support=model.n_support_,
                                    gamma=np.asarray(model._gamma), classes=model.classes_)
        intervals, comparisons = [], []
        for row in metrics_rows:
            intervals.append({"model": row["model"], "seed": "fixed", **{k: row[k] for k in ("n_correct", "n_samples", "accuracy", "wilson_low", "wilson_high")}})
        old = PROJECT_DIR / "结果/阶段4/runs"
        lr = load_npz(old / "E2_C1/predictions_test.npz")
        np.testing.assert_array_equal(lr["y"], yt); np.testing.assert_array_equal(lr["indices"], test["indices_test"])
        for label, pred in [("逻辑回归", lr["predictions"])]:
            correct = int(np.sum(pred == yt)); low, high = wilson(correct,len(yt))
            intervals.append({"model":label,"seed":"fixed","n_correct":correct,"n_samples":len(yt),"accuracy":correct/len(yt),"wilson_low":low,"wilson_high":high})
        for seed in (0, 1, 2):
            loaded = {mode: load_npz(old / f"E3_{mode}_seed{seed}/predictions_test.npz") for mode in ("raw64", "pca32")}
            for mode, saved in loaded.items():
                np.testing.assert_array_equal(saved["y"], yt); np.testing.assert_array_equal(saved["indices"], test["indices_test"])
                correct=int(np.sum(saved["predictions"]==yt)); low,high=wilson(correct,len(yt))
                intervals.append({"model":"BP64" if mode=="raw64" else "PCA32","seed":seed,"n_correct":correct,"n_samples":len(yt),"accuracy":correct/len(yt),"wilson_low":low,"wilson_high":high})
            pairs=[("BP64 vs LR",loaded["raw64"]["predictions"],lr["predictions"]),
                   ("PCA32 vs BP64",loaded["pca32"]["predictions"],loaded["raw64"]["predictions"])]
            pairs += [(label+" vs BP64",pred,loaded["raw64"]["predictions"]) for label,pred in predictions.items()]
            for label, first, second in pairs:
                comparisons.append({"comparison":label,"seed":seed,**paired(first,second,yt)})
    rows(OUT / "baseline_test_metrics.csv", metrics_rows)
    rows(OUT / "wilson_intervals.csv", intervals)
    rows(OUT / "mcnemar_tests.csv", comparisons)
    result={"status":"completed", "selection":selection, "baselines":metrics_rows,
            "wilson_intervals":intervals,"paired_comparisons":comparisons,
            "frozen_protocol_sha256":protocol["frozen_protocol_sha256"],"n_validation_fits":14}
    write(OUT / "supplement_summary.json",result)
    print(json.dumps({"baselines":metrics_rows,"primary_seed0":[r for r in comparisons if r["seed"]==0]},ensure_ascii=False,indent=2))


def verify():
    result=json.loads((OUT/"supplement_summary.json").read_text(encoding="utf-8"))
    protocol=json.loads((OUT/"protocol_before_fits.json").read_text(encoding="utf-8"))
    assert sha(OUT/"source_at_fit.py")==protocol["source_sha256"]
    data=load_prepared_data(include_test=True)
    with threadpool_limits(limits=1):
        for number,row in enumerate(result["baselines"]):
            model=make_model(row["kind"],row["config"])
            model.fit(data["X_train_centered64"],data["y_train"])
            stored=load_npz(OUT/f"baseline_{number}/predictions.npz")
            np.testing.assert_array_equal(model.predict(data["X_test_centered64"]),stored["predictions"])
            np.testing.assert_array_equal(model.predict(data["X_validation_centered64"]),stored["predictions_validation"])
            assert int(np.sum(stored["predictions"]==stored["y"]))==row["n_correct"]
    for row in result["wilson_intervals"]:
        check=binomtest(row["n_correct"],row["n_samples"]).proportion_ci(method="wilson")
        np.testing.assert_allclose([row["wilson_low"],row["wilson_high"]],[check.low,check.high],atol=1e-14,rtol=0)
    from math import comb
    for row in result["paired_comparisons"]:
        b,c=row["only_first_correct"],row["only_second_correct"]
        n=b+c
        exact=min(1.0,2*sum(comb(n,j) for j in range(min(b,c)+1))/2**n) if n else 1.0
        np.testing.assert_allclose(exact,row["p_exact_two_sided"],atol=1e-14,rtol=0)
        assert sum(row[k] for k in ("both_correct","only_first_correct","only_second_correct","both_wrong"))==270
    frozen=validate_freeze(PROJECT_DIR/"结果/阶段4")
    check={"status":"passed","baseline_predictions_reproduced":3,"wilson_intervals_cross_checked":len(result["wilson_intervals"]),
           "exact_tests_cross_checked":len(result["paired_comparisons"]),"original_freeze_unchanged":True}
    write(OUT/"verification.json",check)
    print(json.dumps(check,ensure_ascii=False))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify",action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUT, help="追加实验的新输出目录；--verify时指定已有结果目录")
    args=parser.parse_args()
    OUT=args.output_dir.resolve()
    verify() if args.verify else run()
