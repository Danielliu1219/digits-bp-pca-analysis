"""运行阶段 3 的全部验证及固定 64→32→10 初始试验。"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from bp_network import BPNetwork, TrainingConfig
from data_utils import DATA_DIR, PROJECT_DIR, array_digest, load_prepared_data
from prepare_data import setup_plot_style, write_csv, write_json
from validate_bp import gradient_check, numerical_and_training_checks, small_sample_check


OUTPUT_DIR = PROJECT_DIR / "结果" / "阶段3"
FIGURE_DIR = PROJECT_DIR / "图表" / "阶段3"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_history(path: Path, history: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def plot_history(history: list[dict], checkpoint_epoch: int, filename: str, title: str) -> None:
    epochs = np.asarray([row["epoch"] for row in history])
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6), layout="constrained")
    for subset, label, color in [("train", "训练", "#315A78"), ("validation", "验证", "#B77A36")]:
        axes[0].plot(epochs, [row[f"{subset}_loss"] for row in history], label=label, color=color, linewidth=1.7)
        axes[1].plot(epochs, [row[f"{subset}_accuracy"] * 100 for row in history], label=label, color=color, linewidth=1.7)
    axes[0].set(xlabel="训练轮次", ylabel="交叉熵（对数刻度）", yscale="log", title="损失变化")
    axes[1].set(xlabel="训练轮次", ylabel="准确率（%）", ylim=(0, 102), title="准确率变化")
    for ax in axes:
        ax.axvline(checkpoint_epoch, color="#6E7E85", linestyle="--", linewidth=1, label=f"保存检查点：{checkpoint_epoch} 轮")
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#E6E8EA")
        ax.legend(frameon=False, loc="best", fontsize=9)
    fig.suptitle(title)
    fig.savefig(FIGURE_DIR / filename, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_implementation_report(summary: dict, checks: dict, gradients: list[dict]) -> None:
    initial, final = summary["initial_metrics"], summary["restored_checkpoint_metrics"]
    gradient_rows = "\n".join(
        f"| {row['case']} | {row['parameter']} | {row['checked_entries']} | {row['max_absolute_error']:.3e} | 通过 |"
        for row in gradients
    )
    text = f"""# BP 网络实现与正确性验证

阶段 3 已完成。使用 NumPy 实现单隐藏层网络及 BP，完成梯度、数值、训练流程、保存读取检查，并按研究方案版本 1.0 完成一次初始试运行。

本次为开发验证，未开展阶段 4 的基线、宽度选择和 PCA 分类对比。仅加载训练与验证缓存，不读取测试缓存或包含全体样本的原始快照。

## 一 实现与公式对应

核心文件为 `实验代码/bp_network.py`。对于 m 个样本、d 维输入和 h 个隐藏神经元：

| 参数或变量 | 形状 | 本次配置 |
|---|---|---|
| X | m×d | d=64，输入已归一化并减去训练均值 |
| W1、b1 | d×h、h | 64×32、32 |
| Z1、H | m×h | H=ReLU(XW1+b1) |
| W2、b2 | h×10、10 | 32×10、10 |
| logits、P | m×10 | logits=HW2+b2，P=Softmax(logits) |

参数共 {summary['n_parameters']} 个。隐藏层权重标准差为 √(2/d)，输出层为 √(1/h)，偏置初始为零。损失为每个批次的平均交叉熵。

梯度为 D2=(P−Y)/m，dW2=HᵀD2，db2=ΣD2，D1=(D2W2ᵀ)⊙1[Z1>0]，dW1=XᵀD1，db1=ΣD1。四组梯度均计算完成后，统一执行参数减去学习率乘梯度，避免使用部分更新后的权重。代码使用整数标签索引完成 P−Y，与独热矩阵公式等价。

Softmax 先减去每行最大 logits；交叉熵直接从 log-softmax 计算，避免先得到零概率后计算 log(0)。ReLU 在零点的导数约定为零。

## 二 独立数值梯度检查

用中心差分 [L(θ+ε)−L(θ−ε)]/(2ε)，ε=0.000001，逐项检查两个小网络的全部 {sum(row['checked_entries'] for row in gradients)} 个参数。输入远离 ReLU 零点，检查期间恢复每个被扰动参数。

| 网络 | 参数组 | 检查数量 | 最大绝对误差 | 结果 |
|---|---|---|---|---|
{gradient_rows}

通过标准为逐项满足绝对容差 0.0000001 与相对容差 0.00001 的联合比较。完整数值保存在 `结果/阶段3/gradient_check.csv`。

## 三 训练流程与数值检查

- 极端 logits 下交叉熵为 {checks['numerical_and_training']['extreme_logits_loss']:.1f}，仍为有限值；概率归一化及 logits 平移不变性通过。
- 将整个批次复制三次，平均损失与平均梯度保持一致，验证批次归一化。
- 7 个样本按批量 3 训练，覆盖 3/3/1 的末批情况。
- 人工小步更新降低损失；相同配置与种子的两次短训练逐项一致。
- 人工构造验证损失恶化的例子，在第 4 轮触发早停并恢复第 1 轮检查点，恢复参数与独立训练 1 轮的参数完全一致。
- 模型保存为纯数值 npz，重新加载后的预测概率完全一致，禁用 pickle。
- 准确率、宏平均 F1 和混淆矩阵与 scikit-learn 独立实现一致，包含评价集合缺少某个类别的情况。

## 四 小样本学习能力检查

从训练集中每类选择 2 个样本，共 20 个。固定网络 64→32→10、种子 123、学习率 0.1、批量 20、最多 800 轮；此检查仅在同一小集合上监控损失，不使用验证集或测试集判断泛化。

初始交叉熵为 {checks['small_sample']['initial_loss']:.6f}，恢复检查点后为 {checks['small_sample']['final_loss']:.6f}；小集合准确率由 {checks['small_sample']['initial_accuracy'] * 100:.2f}% 上升至 {checks['small_sample']['final_accuracy'] * 100:.2f}%。达到预先规定的 100% 准确率且交叉熵低于 0.05 的检查要求。

这组配置用于验证学习能力，不改变主实验的学习率和训练规则。所选原始样本编号及训练曲线均已保存。

## 五 原始 64 维初始试运行

沿用固定训练 1257、验证 270 个样本。网络为 64→32→10，初始化种子 42，打乱种子 {summary['shuffle_seed']}，学习率 0.05，批量 32，最多 300 轮。验证交叉熵改善阈值为 0.0001，连续 20 轮未达到改善阈值则早停。

第 0 轮记录初始化状态；从完成第 1 轮训练后开始选择检查点。停止后恢复达到改善阈值时保存的参数，检查点不必等于忽略阈值时曲线上的严格最小值。

| 指标 | 初始化训练集 | 恢复后训练集 | 初始化验证集 | 恢复后验证集 |
|---|---|---|---|---|
| 交叉熵 | {initial['train']['loss']:.6f} | {final['train']['loss']:.6f} | {initial['validation']['loss']:.6f} | {final['validation']['loss']:.6f} |
| 准确率 | {initial['train']['accuracy'] * 100:.2f}% | {final['train']['accuracy'] * 100:.2f}% | {initial['validation']['accuracy'] * 100:.2f}% | {final['validation']['accuracy'] * 100:.2f}% |
| 宏平均 F1 | {initial['train']['macro_f1']:.6f} | {final['train']['macro_f1']:.6f} | {initial['validation']['macro_f1']:.6f} | {final['validation']['macro_f1']:.6f} |

实际训练至第 {summary['stopped_epoch']} 轮，保存检查点为第 {summary['best_epoch']} 轮，{'触发早停' if summary['stopped_early'] else '达到预设最大轮数'}。从参数初始化到训练结束并恢复检查点耗时约 {summary['training_seconds']:.3f} 秒，包含逐轮验证，未包含读取数据、绘图和保存文件。为固定单线程环境下的一次开发运行耗时。

训练曲线证明流程能够降低损失，验证集结果用于检查训练行为。该准确率不是测试准确率，也不作为正式对比结论；泛化分析和重复实验留到阶段 4。

## 六 输出与下一阶段

- `实验代码/bp_network.py`：可复用网络、梯度、SGD、早停、预测和保存读取入口。
- `实验代码/validate_bp.py`：可重复执行的正确性检查。
- `实验代码/run_stage3.py`：本阶段统一入口，支持 `--checks-only`。
- `结果/阶段3/validation_report.json`：梯度、数值、复现、早停与小样本检查。
- `结果/阶段3/initial_run_summary.json`：完整初始配置、指标、环境、数据与代码指纹。
- `结果/阶段3/initial_training_history.csv`、`small_sample_history.csv`：逐轮曲线数据。
- `结果/阶段3/bp_initial_model.npz`：恢复后的初始试运行模型。
- `结果/阶段3/initial_predictions_train_validation.npz`：训练与验证概率、预测、标签和原始编号。
- `图表/阶段3/`：初始训练曲线及小样本学习检查图。

下一阶段为阶段 4 对比实验与 PCA 实验。正式实验直接复用同一 BP 实现及固定数据规则，分别记录逻辑回归、宽度选择及三种子 PCA 对比。
"""
    (PROJECT_DIR / "BP网络实现与验证.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checks-only", action="store_true", help="仅运行正确性检查，不运行 1257 样本的初始试验")
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = DATA_DIR / "prepared_train_validation.npz"
    cache_digest = file_sha256(cache_path)
    data = load_prepared_data()
    if any("test" in key for key in data):
        raise AssertionError("阶段 3 数据入口不应包含测试数据")
    X_train, y_train = data["X_train_centered64"], data["y_train"]
    X_validation, y_validation = data["X_validation_centered64"], data["y_validation"]
    if X_train.shape != (1257, 64) or X_validation.shape != (270, 64):
        raise AssertionError("训练/验证缓存与固定方案不符")

    with threadpool_limits(limits=1):
        print("[1/4] 检查全部小网络参数的中心差分梯度", flush=True)
        gradients = gradient_check()
        write_csv(OUTPUT_DIR / "gradient_check.csv", list(gradients[0]), [list(row.values()) for row in gradients])
        print("[2/4] 检查数值、末批、复现、保存读取和早停恢复", flush=True)
        numerical = numerical_and_training_checks(OUTPUT_DIR)
        print("[3/4] 检查 20 个训练样本的学习能力", flush=True)
        small, small_history = small_sample_check(X_train, y_train, data["indices_train"])
        write_history(OUTPUT_DIR / "small_sample_history.csv", small_history)
        checks = {"status": "passed", "gradient_checks": gradients,
                  "numerical_and_training": numerical, "small_sample": small,
                  "test_data_loaded": False, "loaded_cache": cache_path.name}
        write_json(OUTPUT_DIR / "validation_report.json", checks)
        if args.checks_only:
            print(json.dumps({"status": "passed", "mode": "checks_only", "small_sample": small}, ensure_ascii=False, indent=2))
            return

        print("[4/4] 按版本 1.0 执行 64→32→10 的初始试运行", flush=True)
        config = TrainingConfig()
        started = perf_counter()
        model = BPNetwork(64, 32, 10, seed=42)
        initial_train, initial_validation = model.evaluate(X_train, y_train), model.evaluate(X_validation, y_validation)
        result = model.fit(X_train, y_train, X_validation, y_validation, config, verbose=True)
        training_seconds = perf_counter() - started
        final_train, final_validation = model.evaluate(X_train, y_train), model.evaluate(X_validation, y_validation)
        np.testing.assert_allclose(final_validation["loss"], result.best_validation_loss, rtol=0, atol=1e-13)
        if final_train["loss"] >= initial_train["loss"]:
            raise AssertionError("完整训练未降低训练损失")
        model_file = OUTPUT_DIR / "bp_initial_model.npz"
        model.save(model_file)
        loaded = BPNetwork.load(model_file)
        np.testing.assert_array_equal(loaded.predict_proba(X_validation), model.predict_proba(X_validation))
        write_history(OUTPUT_DIR / "initial_training_history.csv", result.history)
        predictions = {}
        for name in ("train", "validation"):
            probabilities = model.predict_proba(data[f"X_{name}_centered64"])
            predictions[f"probabilities_{name}"] = probabilities
            predictions[f"predictions_{name}"] = probabilities.argmax(axis=1)
            predictions[f"y_{name}"] = data[f"y_{name}"]
            predictions[f"indices_{name}"] = data[f"indices_{name}"]
        np.savez_compressed(OUTPUT_DIR / "initial_predictions_train_validation.npz", **predictions)

        def serialize(metrics: dict) -> dict:
            return {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in metrics.items()}

        manifest = json.loads((DATA_DIR / "data_manifest.json").read_text(encoding="utf-8"))
        summary = {
            "stage": 3, "purpose": "development_initial_run", "research_protocol_version": "1.0",
            "generated_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
            "architecture": [64, 32, 10], "n_parameters": model.n_parameters,
            "initialization_seed": 42, "shuffle_seed": result.shuffle_seed, "config": asdict(config),
            "best_epoch": result.best_epoch, "stopped_epoch": result.stopped_epoch,
            "stopped_early": result.stopped_early, "training_seconds": training_seconds,
            "initial_metrics": {"train": serialize(initial_train), "validation": serialize(initial_validation)},
            "restored_checkpoint_metrics": {"train": serialize(final_train), "validation": serialize(final_validation)},
            "checkpoint_validation_loss_matches_history": True,
            "saved_model_probabilities_exactly_match": True,
            "test_data_loaded": False, "test_metrics_computed": False,
            "dataset_content_sha256": manifest["dataset_content_sha256"],
            "split_content_sha256": manifest["split_content_sha256"],
            "train_validation_cache_sha256": cache_digest,
            "model_parameters_sha256": array_digest(*model.parameters.values()),
            "python": platform.python_version(), "platform": platform.platform(),
            "thread_limit": 1, "threadpools_during_run": threadpool_info(),
            "code_sha256": {name: file_sha256(Path(__file__).parent / name) for name in ["bp_network.py", "validate_bp.py", "run_stage3.py", "data_utils.py"]},
        }
        if cache_digest != file_sha256(cache_path):
            raise AssertionError("阶段 3 运行期间数据缓存被改变")
        write_json(OUTPUT_DIR / "initial_run_summary.json", summary)
        write_implementation_report(summary, checks, gradients)

    setup_plot_style()
    plot_history(result.history, result.best_epoch, "01_BP初始训练曲线.png", "BP 网络初始试运行  64→32→10")
    plot_history(small_history, small["best_epoch"], "02_小样本学习检查.png", "20 个训练样本的学习检查  两条曲线来自同一小集合")
    print(json.dumps({"stage": 3, "status": "passed", "gradient_entries_checked": sum(row["checked_entries"] for row in gradients),
                      "small_sample_accuracy": small["final_accuracy"], "best_epoch": result.best_epoch,
                      "stopped_epoch": result.stopped_epoch, "train_accuracy": final_train["accuracy"],
                      "validation_accuracy": final_validation["accuracy"], "validation_macro_f1": final_validation["macro_f1"],
                      "test_data_loaded": False, "report": str(PROJECT_DIR / "BP网络实现与验证.md")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
