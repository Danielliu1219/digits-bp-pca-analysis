"""阶段 4：先开发并冻结配置，再使用冻结模型评价测试集。

默认依次执行 develop/evaluate；冻结文件已存在时只复用并核验模型。
本程序不使用 pickle，逻辑回归保存系数、截距及类别等纯数值状态。
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
from time import perf_counter
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_info, threadpool_limits

from bp_network import BPNetwork, TrainingConfig, classification_metrics, softmax_cross_entropy
from data_utils import DATA_DIR, PROJECT_DIR, array_digest, fit_preprocessor, load_npz, load_prepared_data, transform_inputs


OUTPUT_DIR = PROJECT_DIR / "结果" / "阶段4"
WIDTHS = (16, 32, 64)
C_VALUES = (0.1, 1.0, 10.0)
DIMENSIONS = (64, 8, 16, 32)
SEEDS = (0, 1, 2)
SOURCE_NAMES = ("run_experiments.py", "bp_network.py", "data_utils.py")


def now() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_rows(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def event(output: Path, action: str, **details) -> None:
    with (output / "execution_log.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"at": now(), "action": action, **details}, ensure_ascii=False) + "\n")


def recover_raw(centered: np.ndarray, training_mean: np.ndarray) -> np.ndarray:
    """从缓存无损还原原始整数像素，开发阶段不必打开含测试样本的原始快照。"""
    recovered = (centered + training_mean) * 16.0
    integer_pixels = np.rint(recovered)
    np.testing.assert_allclose(recovered, integer_pixels, rtol=0, atol=1e-12)
    if not np.isfinite(integer_pixels).all() or integer_pixels.min() < 0 or integer_pixels.max() > 16:
        raise ValueError("缓存无法无损还原为 0—16 的原始像素")
    return np.ascontiguousarray(integer_pixels, dtype=np.float64)


def prepare_inputs(X_train_raw: np.ndarray, X_validation_raw: np.ndarray, dimension: int) -> tuple[dict, np.ndarray, np.ndarray, float]:
    started = perf_counter()
    if dimension == 64:
        state = {"pixel_divisor": np.asarray(16.0), "mean": (X_train_raw / 16.0).mean(axis=0),
                 "components": np.empty((0, 64)), "fit_n_samples": np.asarray(len(X_train_raw))}
    else:
        state = fit_preprocessor(X_train_raw, n_components=dimension)
    dim = None if dimension == 64 else dimension
    train = transform_inputs(X_train_raw, state, dim)
    validation = transform_inputs(X_validation_raw, state, dim)
    return state, train, validation, perf_counter() - started


def predict_logits(kind: str, model_path: Path, X: np.ndarray) -> np.ndarray:
    if kind == "bp":
        return BPNetwork.load(model_path).forward(X)["logits"]
    state = load_npz(model_path)
    np.testing.assert_array_equal(state["classes"], np.arange(10))
    return X @ state["coefficients"].T + state["intercepts"]


def score_logits(logits: np.ndarray, y: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray]:
    probabilities, loss = softmax_cross_entropy(logits, y)
    prediction = probabilities.argmax(axis=1)
    metrics = classification_metrics(y, prediction, 10)
    return {"loss": loss, "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
            "n_correct": int(np.sum(prediction == y)), "n_samples": len(y)}, probabilities, prediction


def fit_one(output: Path, run_id: str, experiment: str, kind: str, dimension: int,
            hidden_dim: int | None, seed: int | None, C: float | None,
            X_train_raw: np.ndarray, X_validation_raw: np.ndarray, data: dict) -> dict:
    directory = output / "runs" / run_id
    directory.mkdir(parents=True, exist_ok=False)
    state, X_train, X_validation, preprocessing_seconds = prepare_inputs(X_train_raw, X_validation_raw, dimension)
    # 新拟合的预处理应复现阶段 2 的输入，防止维数/中心化规则漂移。
    suffix = "centered64" if dimension == 64 else f"pca{dimension}"
    np.testing.assert_allclose(X_train, data[f"X_train_{suffix}"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(X_validation, data[f"X_validation_{suffix}"], rtol=0, atol=1e-12)
    y_train, y_validation = data["y_train"], data["y_validation"]
    started = perf_counter()
    if kind == "bp":
        model = BPNetwork(dimension, hidden_dim, 10, seed)
        trained = model.fit(X_train, y_train, X_validation, y_validation, TrainingConfig())
        model_training_seconds = perf_counter() - started
        n_parameters = model.n_parameters
        epoch_info = {"best_epoch": trained.best_epoch, "stopped_epoch": trained.stopped_epoch,
                      "stopped_early": trained.stopped_early, "shuffle_seed": trained.shuffle_seed,
                      "solver_iterations": None}
        write_rows(directory / "history.csv", trained.history)
        model.save(directory / "model.npz")
    else:
        # 当前安装版本使用 l1_ratio=0 明确指定纯 L2，避免旧 penalty API。
        model = LogisticRegression(C=C, l1_ratio=0.0, solver="lbfgs", tol=1e-6, max_iter=1000)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(X_train, y_train)
        model_training_seconds = perf_counter() - started
        n_parameters = int(model.coef_.size + model.intercept_.size)
        epoch_info = {"best_epoch": None, "stopped_epoch": None, "stopped_early": None,
                      "shuffle_seed": None, "solver_iterations": int(model.n_iter_.max())}
        np.savez_compressed(directory / "model.npz", schema_version=np.asarray(1),
                            coefficients=model.coef_, intercepts=model.intercept_, classes=model.classes_,
                            C=np.asarray(C), l1_ratio=np.asarray(0.0), max_iter=np.asarray(1000), tol=np.asarray(1e-6))
    np.savez_compressed(directory / "preprocessor.npz", **state)
    record = {
        "run_id": run_id, "experiment": experiment, "model_kind": kind,
        "input_mode": suffix, "input_dim": dimension, "hidden_dim": hidden_dim,
        "initialization_seed": seed, "regularization_C": C,
        "n_parameters": n_parameters, "preprocessing_parameter_count": int(state["mean"].size + state["components"].size),
        "pipeline_parameter_count": int(n_parameters + state["mean"].size + state["components"].size),
        "cumulative_explained_variance": 1.0 if dimension == 64 else float(state["explained_variance_ratio"].sum()),
        "preprocessing_seconds": preprocessing_seconds, "model_training_seconds": model_training_seconds,
        "pipeline_training_seconds": preprocessing_seconds + model_training_seconds,
        "n_train": len(y_train), "n_validation": len(y_validation), **epoch_info,
    }
    saved_predictions = {}
    for name, X, y in [("train", X_train, y_train), ("validation", X_validation, y_validation)]:
        logits = predict_logits(kind, directory / "model.npz", X)
        metrics, probabilities, prediction = score_logits(logits, y)
        if kind == "logistic_regression":
            np.testing.assert_allclose(probabilities, model.predict_proba(X), rtol=0, atol=2e-14)
        record.update({f"{name}_{key}": value for key, value in metrics.items()})
        saved_predictions.update({f"probabilities_{name}": probabilities, f"predictions_{name}": prediction,
                                  f"y_{name}": y, f"indices_{name}": data[f"indices_{name}"]})
    if kind == "bp":
        np.testing.assert_allclose(record["validation_loss"], trained.best_validation_loss, rtol=0, atol=1e-13)
    np.savez_compressed(directory / "predictions_train_validation.npz", **saved_predictions)
    write_json(directory / "record.json", record)
    event(output, "fit_completed", run_id=run_id, validation_accuracy=record["validation_accuracy"])
    print(f"{run_id}: val_acc={record['validation_accuracy']:.6f}, val_f1={record['validation_macro_f1']:.6f}, "
          f"epochs={record['stopped_epoch']}, time={record['pipeline_training_seconds']:.3f}s", flush=True)
    return record


def group_records(records: list[dict], include_test: bool) -> list[dict]:
    groups = []
    for dimension in DIMENSIONS:
        rows = [row for row in records if row["experiment"] == "E3" and row["input_dim"] == dimension]
        if len(rows) != 3 or sorted(row["initialization_seed"] for row in rows) != list(SEEDS):
            raise AssertionError("每种维数必须完整包含预设的 3 个训练种子")
        grouped = {"model": f"BP_{'raw64' if dimension == 64 else 'pca' + str(dimension)}",
                   "input_dim": dimension, "hidden_dim": rows[0]["hidden_dim"], "n_runs": len(rows),
                   "n_parameters": rows[0]["n_parameters"], "pipeline_parameter_count": rows[0]["pipeline_parameter_count"],
                   "cumulative_explained_variance": rows[0]["cumulative_explained_variance"]}
        metrics = ["validation_accuracy", "validation_macro_f1", "validation_loss", "preprocessing_seconds",
                   "model_training_seconds", "pipeline_training_seconds", "stopped_epoch"]
        if include_test:
            metrics += ["test_accuracy", "test_macro_f1", "test_loss"]
        for metric in metrics:
            values = np.asarray([row[metric] for row in rows], dtype=np.float64)
            grouped[f"{metric}_mean"] = float(values.mean())
            grouped[f"{metric}_std"] = float(values.std(ddof=1))
        groups.append(grouped)
    return groups


def develop(output: Path) -> dict:
    if (output / "frozen_protocol.json").exists() or (output / "runs").exists():
        raise FileExistsError("此目录已有实验；可直接 evaluate 复用冻结模型，独立重做请指定新的 --output-dir")
    output.mkdir(parents=True, exist_ok=True)
    data = load_prepared_data()
    if any("test" in key for key in data):
        raise AssertionError("开发阶段不可加载测试缓存")
    source_state = load_npz(DATA_DIR / "preprocessor.npz")
    X_train_raw = recover_raw(data["X_train_centered64"], source_state["mean"])
    X_validation_raw = recover_raw(data["X_validation_centered64"], source_state["mean"])
    manifest = read_json(DATA_DIR / "data_manifest.json")
    protocol = {"research_protocol_version": "1.0", "started_at": now(),
                "width_grid": list(WIDTHS), "C_grid": list(C_VALUES), "dimensions": list(DIMENSIONS), "seeds": list(SEEDS),
                "development_seed": 42, "training_config": asdict(TrainingConfig()),
                "logistic_config": {"l1_ratio": 0.0, "solver": "lbfgs", "tol": 1e-6, "max_iter": 1000},
                "selection_rule": "validation accuracy, then validation macro F1, then smaller hidden width / C / PCA dimension",
                "pca_dimension_rule": "mean validation accuracy over all 3 seeds, then mean macro F1, then smaller dimension",
                "representative_seed_for_stage5": 0,
                "input_recovery": "lossless integer-pixel inversion from train/validation cache and training mean; tolerance 1e-12",
                "preprocessing_timing": "fit training mean/PCA and transform training/validation; raw pixel recovery excluded",
                "bp_timing": "parameter initialization through fit return and checkpoint restoration, including validation",
                "dataset_content_sha256": manifest["dataset_content_sha256"],
                "split_content_sha256": manifest["split_content_sha256"],
                "source_cache_sha256": sha256(DATA_DIR / "prepared_train_validation.npz"),
                "source_preprocessor_sha256": sha256(DATA_DIR / "preprocessor.npz"),
                "code_sha256": {name: sha256(Path(__file__).parent / name) for name in SOURCE_NAMES},
                "python": platform.python_version(), "platform": platform.platform(),
                "versions": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn", "threadpoolctl")}}
    write_json(output / "development_protocol.json", protocol)
    event(output, "development_started", planned_fit_count=18, test_data_loaded=False)
    records = []
    with threadpool_limits(limits=1):
        threadpools = threadpool_info()
        for width in WIDTHS:
            records.append(fit_one(output, f"E1_h{width}_seed42", "E1", "bp", 64, width, 42, None, X_train_raw, X_validation_raw, data))
        selected_bp = max(records, key=lambda row: (row["validation_accuracy"], row["validation_macro_f1"], -row["hidden_dim"]))
        for C in C_VALUES:
            records.append(fit_one(output, f"E2_C{C:g}", "E2", "logistic_regression", 64, None, None, C, X_train_raw, X_validation_raw, data))
        selected_lr = max((row for row in records if row["experiment"] == "E2"),
                          key=lambda row: (row["validation_accuracy"], row["validation_macro_f1"], -row["regularization_C"]))
        for dimension in DIMENSIONS:
            for seed in SEEDS:
                run_id = f"E3_{'raw64' if dimension == 64 else 'pca' + str(dimension)}_seed{seed}"
                records.append(fit_one(output, run_id, "E3", "bp", dimension, selected_bp["hidden_dim"], seed, None, X_train_raw, X_validation_raw, data))
    groups = group_records(records, include_test=False)
    selected_pca = max((group for group in groups if group["input_dim"] != 64),
                       key=lambda group: (group["validation_accuracy_mean"], group["validation_macro_f1_mean"], -group["input_dim"]))
    if len(records) != 18 or sha256(DATA_DIR / "prepared_train_validation.npz") != protocol["source_cache_sha256"]:
        raise AssertionError("拟合数量不符或原始缓存已改变")
    write_rows(output / "development_results.csv", records)
    write_rows(output / "validation_group_summary.csv", groups)
    files = {}
    for run in records:
        directory = output / "runs" / run["run_id"]
        files.update({str(path.relative_to(output).as_posix()): sha256(path) for path in directory.iterdir() if path.is_file()})
    for filename in ("development_protocol.json", "development_results.csv", "validation_group_summary.csv"):
        files[filename] = sha256(output / filename)
    frozen = {
        **protocol, "status": "frozen_before_test", "frozen_at": now(),
        "selected_hidden_dim": selected_bp["hidden_dim"], "selected_width_run": selected_bp["run_id"],
        "selected_C": selected_lr["regularization_C"], "selected_logistic_run": selected_lr["run_id"],
        "recommended_pca_dimension_from_validation": selected_pca["input_dim"],
        "evaluation_run_ids": [selected_lr["run_id"], *[row["run_id"] for row in records if row["experiment"] == "E3"]],
        "development_run_count": len(records), "test_data_loaded_at_freeze": False,
        "bp_summary_std_ddof": 1, "thread_limit": 1, "threadpools_during_training": threadpools,
        "immutable_files_sha256": files,
    }
    write_json(output / "frozen_protocol.json", frozen)
    event(output, "protocol_frozen", frozen_protocol_sha256=sha256(output / "frozen_protocol.json"),
          selected_hidden_dim=frozen["selected_hidden_dim"], selected_C=frozen["selected_C"],
          recommended_pca_dim=frozen["recommended_pca_dimension_from_validation"], test_data_loaded=False)
    print(json.dumps({key: frozen[key] for key in ["status", "selected_hidden_dim", "selected_C", "recommended_pca_dimension_from_validation", "development_run_count"]}, ensure_ascii=False, indent=2))
    return frozen


def validate_freeze(output: Path) -> dict:
    frozen = read_json(output / "frozen_protocol.json")
    if frozen["status"] != "frozen_before_test" or frozen["test_data_loaded_at_freeze"]:
        raise AssertionError("冻结协议状态不符")
    for name, digest in frozen["code_sha256"].items():
        if sha256(Path(__file__).parent / name) != digest:
            raise AssertionError(f"冻结后源代码发生变化：{name}")
    for relative, digest in frozen["immutable_files_sha256"].items():
        if sha256(output / relative) != digest:
            raise AssertionError(f"冻结后实验文件发生变化：{relative}")
    for filename, field in [("prepared_train_validation.npz", "source_cache_sha256"), ("preprocessor.npz", "source_preprocessor_sha256")]:
        if sha256(DATA_DIR / filename) != frozen[field]:
            raise AssertionError(f"数据准备文件已变化：{filename}")
    return frozen


def evaluate(output: Path) -> None:
    frozen = validate_freeze(output)
    freeze_digest = sha256(output / "frozen_protocol.json")
    # 首次读取测试数组只能出现在冻结文件及全部模型指纹核验之后。
    test = load_npz(DATA_DIR / "prepared_test.npz")
    source_state = load_npz(DATA_DIR / "preprocessor.npz")
    X_test_raw = recover_raw(test["X_test_centered64"], source_state["mean"])
    event(output, "test_loaded", frozen_protocol_sha256=freeze_digest, test_n_samples=len(test["y_test"]))
    results = []
    with threadpool_limits(limits=1):
        for run_id in frozen["evaluation_run_ids"]:
            directory = output / "runs" / run_id
            record = read_json(directory / "record.json")
            state = load_npz(directory / "preprocessor.npz")
            dimension = record["input_dim"]
            X = transform_inputs(X_test_raw, state, None if dimension == 64 else dimension)
            suffix = "centered64" if dimension == 64 else f"pca{dimension}"
            np.testing.assert_allclose(X, test[f"X_test_{suffix}"], rtol=0, atol=1e-12)
            metrics, probabilities, prediction = score_logits(predict_logits(record["model_kind"], directory / "model.npz", X), test["y_test"])
            results.append({**record, **{f"test_{key}": value for key, value in metrics.items()}})
            np.savez_compressed(directory / "predictions_test.npz", probabilities=probabilities,
                                predictions=prediction, y=test["y_test"], indices=test["indices_test"])
            write_json(directory / "test_metrics.json", {"frozen_protocol_sha256": freeze_digest, **metrics})
    groups = group_records(results, include_test=True)
    write_rows(output / "test_results.csv", results)
    write_rows(output / "group_summary.csv", groups)
    summary = {"evaluated_at": now(), "frozen_protocol_sha256": freeze_digest,
               "selected_hidden_dim": frozen["selected_hidden_dim"], "selected_C": frozen["selected_C"],
               "recommended_pca_dimension_from_validation": frozen["recommended_pca_dimension_from_validation"],
               "representative_seed_for_stage5": frozen["representative_seed_for_stage5"],
               "n_test": len(test["y_test"]), "evaluated_model_count": len(results),
               "baseline": next(row for row in results if row["model_kind"] == "logistic_regression"), "bp_groups": groups}
    write_json(output / "experiment_summary.json", summary)
    if sha256(output / "frozen_protocol.json") != freeze_digest:
        raise AssertionError("评价期间冻结协议被修改")
    validate_freeze(output)
    event(output, "evaluation_completed", frozen_protocol_sha256=freeze_digest, n_models=len(results))
    write_experiment_report(output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def write_experiment_report(output: Path, summary: dict) -> None:
    frozen = read_json(output / "frozen_protocol.json")
    development = [read_json(output / "runs" / run_id / "record.json") for run_id in
                   [f"E1_h{h}_seed42" for h in WIDTHS] + [f"E2_C{C:g}" for C in C_VALUES]]
    selection_rows = "\n".join(f"| {row['run_id']} | {row['validation_accuracy'] * 100:.2f}% | {row['validation_macro_f1']:.6f} |" for row in development)
    baseline = summary["baseline"]
    test_rows = f"| 逻辑回归 | 64 | 1 | {baseline['test_accuracy'] * 100:.2f}% | {baseline['test_macro_f1']:.6f} | {baseline['pipeline_training_seconds']:.4f} | {baseline['n_parameters']} |"
    for row in summary["bp_groups"]:
        test_rows += f"\n| BP | {row['input_dim']} | 3 | {row['test_accuracy_mean'] * 100:.2f} ± {row['test_accuracy_std'] * 100:.2f}% | {row['test_macro_f1_mean']:.6f} ± {row['test_macro_f1_std']:.6f} | {row['pipeline_training_seconds_mean']:.4f} ± {row['pipeline_training_seconds_std']:.4f} | {row['n_parameters']} |"
    report = f"""# 对比实验与 PCA 实验记录

阶段 4 按研究方案版本 1.0 完成。开发阶段共拟合 18 次：3 次 BP 宽度选择、3 次逻辑回归 C 选择、12 次 BP 主实验；冻结后使用 13 个预定模型评价同一份 270 样本测试集。

## 一 固定规则与实际参数选择

训练/验证/测试划分仍为 1257/270/270。BP 复用阶段 3 的实现，学习率 0.05、批量 32、最大 300 轮、早停耐心 20、改善阈值 0.0001；主实验仅改变输入维数和预设训练种子。逻辑回归使用 L2、L-BFGS、容差 0.000001、最大迭代 1000。

参数选择以验证准确率优先，其次验证宏平均 F1，再选更小的宽度或 C；宽度实验种子为 42。PCA 推荐维数以三个种子的平均验证准确率、平均验证 F1 和较小维数依次决定。全部比较均在测试数据读取之前完成。

| 开发配置 | 验证准确率 | 验证宏平均 F1 |
|---|---|---|
{selection_rows}

选择隐藏层宽度 h*={frozen['selected_hidden_dim']}，逻辑回归 C*={frozen['selected_C']:g}。主实验统一使用该宽度，输入为原始 64 维或 PCA 8/16/32 维，初始化种子为 0/1/2，打乱种子为 1000/1001/1002。验证集推荐的 PCA 维数为 {frozen['recommended_pca_dimension_from_validation']}；此选择已写入冻结文件。

## 二 冻结后测试结果

BP 表中为三个种子的均值 ± 样本标准差（ddof=1）；准确率标准差以百分点表示。逻辑回归为一次确定性拟合。全部模型均报告，没有挑选测试效果最好的种子代替均值。

| 模型 | 输入维数 | 运行数 | 测试准确率 | 测试宏平均 F1 | 管线训练时间（秒） | 模型参数量 |
|---|---|---|---|---|---|---|
{test_rows}

这些统计基于固定数据划分，标准差仅反映 BP 初始化与打乱的影响，不表示跨划分或跨书写者的不确定性。PCA 维数比较也改变了输入表示和网络参数量；逻辑回归与 BP 的训练器及正则化不同。

## 三 预处理与计时口径

为避免开发阶段打开包含测试样本的原始快照，训练和验证原始像素由已有中心化缓存与训练均值无损还原，整数量化前后误差经过 0.000000000001 容差检查。正式计时从训练集重新拟合均值或 PCA 开始，包含训练/验证输入变换；不包含缓存读取和像素还原。

原始 64 维路径只计算训练均值，PCA 路径重新执行完整 SVD、关闭白化。每次运行均有独立保存的预处理状态。新变换结果与阶段 2 缓存逐项核对；验证与测试不参与拟合。

模型训练时间从参数/模型初始化开始，BP 包含逐轮验证及检查点恢复，逻辑回归包含求解器拟合。管线训练时间为预处理与模型训练时间之和，单线程顺序运行，不含最终测试、图表和文件写入。时间同时受实现、优化器和停止轮次影响，不能直接解释为网络结构的固有速度优势。

模型参数量与完整管线参数量分别保存在结果表中；后者还计入训练均值和 PCA 主成分矩阵。PCA 减少网络参数不等于整体存储量必然下降。

## 四 原始记录与后续使用

- `development_protocol.json`：训练前保存的网格、种子、计时、选择规则与代码指纹。
- `development_results.csv`：18 次拟合的训练与验证指标、耗时、参数量和停止轮次。
- `validation_group_summary.csv`：测试前的四组 BP 验证均值与标准差。
- `frozen_protocol.json`：冻结配置、13 个测试模型编号及所有开发文件指纹。
- `execution_log.jsonl`：拟合、冻结、读取测试和评价完成的顺序记录。
- `test_results.csv`：13 个模型的逐次测试记录；开发候选中的其余模型不参与测试。
- `group_summary.csv`、`experiment_summary.json`：BP 汇总和所选逻辑回归结果。
- `runs/`：每次模型、预处理参数、训练验证预测、BP 历史及被评价模型的测试预测。

记录目录为 `{output.relative_to(PROJECT_DIR).as_posix()}`。模型及数据状态为纯数值 npz，不依赖 pickle。阶段 5 将直接读取这些结果，制作正式图表、混淆矩阵和误分类分析；代表种子事先固定为 0。
"""
    (output / "对比实验记录.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["all", "develop", "evaluate"], default="all")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if args.phase == "develop" or (args.phase == "all" and not (output / "frozen_protocol.json").exists()):
        develop(output)
    if args.phase in ("all", "evaluate"):
        evaluate(output)


if __name__ == "__main__":
    main()
