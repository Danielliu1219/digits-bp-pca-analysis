"""执行阶段 2：数据审查、固定划分、训练集预处理、探索图表和说明。"""

from __future__ import annotations

import csv
import importlib.metadata
import json
import os
import platform
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from sklearn.datasets import load_digits
from threadpoolctl import threadpool_info, threadpool_limits

from data_utils import (
    DATA_DIR, PCA_DIMENSIONS, PIXEL_DIVISOR, PROJECT_DIR,
    array_digest, check_split, fit_preprocessor, load_npz,
    make_fixed_split, transform_inputs,
)


FIGURE_DIR = PROJECT_DIR / "图表" / "阶段2"
SPLIT_NAMES = ("train", "validation", "test")
CONFIG = {
    "schema_version": 1,
    "dataset": "sklearn.datasets.load_digits",
    "n_samples": 1797,
    "test_size": 270,
    "test_split_seed": 42,
    "validation_size": 270,
    "validation_split_seed": 43,
    "stratify": True,
    "pixel_divisor": PIXEL_DIVISOR,
    "pca_dimensions": list(PCA_DIMENSIONS),
    "pca_solver": "full",
    "pca_whiten": False,
    "preprocessing_fit_subset": "train",
    "exploratory_image_and_pixel_subset": "train",
}


def write_json(path: Path, content: dict) -> None:
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, header: list[str], rows: list) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def setup_plot_style() -> None:
    available = {font.name for font in font_manager.fontManager.ttflist}
    selected = next((name for name in ["Microsoft YaHei", "SimHei", "SimSun"] if name in available), None)
    if selected is None:
        raise RuntimeError("未找到中文字体，请安装微软雅黑、黑体或宋体后重试")
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": [selected],
        "axes.unicode_minus": False, "font.size": 10,
        "axes.titlesize": 12, "figure.titlesize": 15,
        "axes.spines.top": False, "axes.spines.right": False,
        "savefig.dpi": 180, "figure.facecolor": "white", "axes.facecolor": "white",
    })


def save_figure(fig: plt.Figure, filename: str) -> None:
    fig.savefig(FIGURE_DIR / filename, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def audit_data(X: np.ndarray, y: np.ndarray, images: np.ndarray) -> dict:
    _, inverse, counts = np.unique(X, axis=0, return_inverse=True, return_counts=True)
    duplicate_groups = np.flatnonzero(counts > 1)
    conflicting = sum(len(np.unique(y[inverse == group])) > 1 for group in duplicate_groups)
    audit = {
        "n_samples": int(len(y)), "n_features": int(X.shape[1]),
        "image_shape": list(images.shape), "X_dtype": str(X.dtype), "y_dtype": str(y.dtype),
        "pixel_min": float(X.min()), "pixel_max": float(X.max()),
        "nan_count": int(np.isnan(X).sum()), "infinite_count": int(np.isinf(X).sum()),
        "integer_valued_pixels": bool(np.equal(X, np.floor(X)).all()),
        "labels": np.unique(y).tolist(), "class_counts": np.bincount(y, minlength=10).tolist(),
        "exact_duplicate_groups": int(len(duplicate_groups)),
        "extra_duplicate_samples": int((counts - 1).sum()),
        "duplicate_groups_with_conflicting_labels": int(conflicting),
        "flat_image_consistent": bool(np.array_equal(X, images.reshape(len(y), -1))),
        "dataset_content_sha256": array_digest(X, y),
    }
    if (X.shape != (1797, 64) or images.shape != (1797, 8, 8)
            or not np.array_equal(np.unique(y), np.arange(10))
            or not np.isfinite(X).all() or not audit["flat_image_consistent"]
            or not audit["integer_valued_pixels"] or X.min() < 0 or X.max() > 16):
        raise ValueError("数据结构或质量与预期不符，请检查 dataset_audit.json")
    return audit


def save_splits(X: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    splits = make_fixed_split(y)
    check_split(splits, len(y))
    path = DATA_DIR / "split_indices.npz"
    if path.exists():
        previous = load_npz(path)
        if set(previous) != set(splits) or any(not np.array_equal(previous[name], splits[name]) for name in splits):
            raise ValueError("当前生成的划分与已保存索引不同；保留原文件并先核查环境和方案")
    else:
        np.savez_compressed(path, **splits)
    split_column = np.empty(len(y), dtype="U10")
    for name, indices in splits.items():
        split_column[indices] = name
    write_csv(DATA_DIR / "split_manifest.csv", ["sample_index", "label", "split"],
              [[i, int(y[i]), split_column[i]] for i in range(len(y))])
    # 重复图像即使索引不同也可能造成泄漏，按内容检查不同子集。
    hashes = {name: {array_digest(row) for row in X[indices]} for name, indices in splits.items()}
    for first, second in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        if hashes[first] & hashes[second]:
            raise ValueError(f"发现跨 {first}/{second} 的完全重复图像，需要先修订独立性方案")
    return splits


def export_tables(X: np.ndarray, y: np.ndarray, splits: dict, state: dict) -> list:
    counts = [[digit, int((y == digit).sum()), *[int((y[splits[name]] == digit).sum()) for name in SPLIT_NAMES]] for digit in range(10)]
    write_csv(DATA_DIR / "class_distribution.csv", ["digit", "all", *SPLIT_NAMES], counts)
    train = X[splits["train"]]
    pixels = [[i, i // 8, i % 8, float(train[:, i].mean()), float(train[:, i].std(ddof=1)),
               float(train[:, i].min()), float(train[:, i].max()), float((train[:, i] == 0).mean())] for i in range(64)]
    write_csv(DATA_DIR / "pixel_statistics_train.csv", ["feature", "row", "column", "mean", "sample_std", "min", "max", "zero_fraction"], pixels)
    cumulative = np.cumsum(state["explained_variance_ratio"])
    write_csv(DATA_DIR / "pca_variance_train.csv", ["component", "variance", "variance_ratio", "cumulative_ratio"],
              [[i + 1, float(v), float(r), float(cumulative[i])] for i, (v, r) in enumerate(zip(state["explained_variance"], state["explained_variance_ratio"]))])
    return counts


def create_figures(X: np.ndarray, y: np.ndarray, splits: dict, state: dict) -> None:
    setup_plot_style()
    train_indices = splits["train"]
    X_train, y_train = X[train_indices], y[train_indices]
    # 每类按原始样本编号选择最早的 5 个训练样本，便于追溯。
    samples = [np.sort(train_indices[y_train == digit])[:5] for digit in range(10)]
    fig, axes = plt.subplots(5, 10, figsize=(12.5, 7.2), layout="constrained")
    for row in range(5):
        for digit in range(10):
            idx = samples[digit][row]
            ax = axes[row, digit]
            ax.imshow(X[idx].reshape(8, 8), cmap="gray_r", vmin=0, vmax=16, interpolation="nearest")
            ax.set_title(f"数字 {digit}" if row == 0 else "", fontsize=10)
            ax.set_xlabel(f"编号 {idx}", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values(): spine.set_visible(False)
    fig.suptitle("训练集样本示例  每类 5 张")
    save_figure(fig, "01_训练样本.png")

    fig, ax = plt.subplots(figsize=(10, 4.7), layout="constrained")
    bottom = np.zeros(10)
    for name, label, color in [("train", "训练集", "#315A78"), ("validation", "验证集", "#B77A36"), ("test", "测试集", "#5F9089")]:
        heights = np.bincount(y[splits[name]], minlength=10)
        ax.bar(np.arange(10), heights, bottom=bottom, width=0.65, color=color, label=f"{label}  n={len(splits[name])}")
        for digit, value in enumerate(heights):
            ax.text(digit, bottom[digit] + value / 2, str(value), ha="center", va="center", color="white", fontsize=9)
        bottom += heights
    for digit, total in enumerate(bottom): ax.text(digit, total + 2, str(int(total)), ha="center", fontsize=10)
    ax.set(xticks=np.arange(10), xlabel="数字类别", ylabel="样本数", ylim=(0, 215), title="分层划分后的类别分布")
    ax.legend(loc="upper center", ncol=3, frameon=False)
    ax.set_axisbelow(True); ax.grid(axis="y", color="#E6E8EA")
    save_figure(fig, "02_类别与划分分布.png")

    fig, axes = plt.subplots(2, 5, figsize=(11, 5.4), layout="constrained")
    for digit, ax in enumerate(axes.flat):
        image = X_train[y_train == digit].mean(axis=0).reshape(8, 8)
        plotted = ax.imshow(image, cmap="gray_r", vmin=0, vmax=16, interpolation="nearest")
        ax.set_title(f"数字 {digit}  n={(y_train == digit).sum()}", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(plotted, ax=axes.ravel().tolist(), shrink=0.75, label="平均像素值", ticks=[0, 8, 16])
    fig.suptitle("各类别平均图像  仅使用训练集")
    save_figure(fig, "03_训练集类别均值图.png")

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.2), layout="constrained")
    for ax, matrix, title in [(axes[0], X_train.mean(axis=0).reshape(8, 8), "逐位置像素均值"),
                               (axes[1], X_train.std(axis=0, ddof=1).reshape(8, 8), "逐位置像素标准差")]:
        plotted = ax.imshow(matrix, cmap="Blues", interpolation="nearest")
        ax.set(title=title, xlabel="列号", ylabel="行号", xticks=range(8), yticks=range(8))
        fig.colorbar(plotted, ax=ax, shrink=0.85)
    values = np.bincount(X_train.astype(np.int64).ravel(), minlength=17)
    axes[2].bar(np.arange(17), values / values.sum() * 100, color="#315A78")
    axes[2].set(title="像素取值分布", xlabel="原始像素值", ylabel="像素占比（%）", xticks=range(0, 17, 2))
    axes[2].set_axisbelow(True); axes[2].grid(axis="y", color="#E6E8EA")
    fig.suptitle("像素统计  仅使用训练集")
    save_figure(fig, "04_训练集像素统计.png")

    ratios = state["explained_variance_ratio"]
    cumulative = np.cumsum(ratios)
    fig, axes = plt.subplots(1, 2, figsize=(12.1, 4.7), layout="constrained")
    axes[0].bar(np.arange(1, 33), ratios[:32] * 100, color="#315A78")
    axes[0].set(xlabel="主成分序号", ylabel="解释方差比例（%）", title="前 32 个主成分")
    axes[1].plot(np.arange(1, 65), cumulative * 100, color="#315A78", linewidth=2)
    for threshold in (90, 95):
        axes[1].axhline(threshold, color="#9DA5AC", linestyle=":", linewidth=1)
    for dim, offset in [(8, -19), (16, -13), (32, -7)]:
        axes[1].scatter(dim, cumulative[dim - 1] * 100, color="#B77A36", zorder=3)
        axes[1].annotate(f"{dim} 维：{cumulative[dim - 1] * 100:.1f}%", (dim, cumulative[dim - 1] * 100),
                         xytext=(dim + 2, cumulative[dim - 1] * 100 + offset), fontsize=10,
                         arrowprops={"arrowstyle": "-", "color": "#9DA5AC"})
    axes[1].set(xlabel="保留主成分数", ylabel="累计解释方差（%）", title="累计解释方差", xlim=(0, 65), ylim=(0, 103))
    for ax in axes: ax.set_axisbelow(True); ax.grid(axis="y", color="#E6E8EA")
    fig.suptitle("PCA 方差分析  仅在训练集拟合")
    save_figure(fig, "05_训练集PCA解释方差.png")

    fig, axes = plt.subplots(3, 4, figsize=(9, 6.6), layout="constrained")
    for row, digit in enumerate([0, 3, 8]):
        idx = samples[digit][0]
        centered = transform_inputs(X[idx:idx + 1], state)
        for column, dim in enumerate([None, 8, 16, 32]):
            if dim is None:
                image = X[idx]
            else:
                vectors = state["components"][:dim]
                image = ((centered @ vectors.T) @ vectors + state["mean"]) * PIXEL_DIVISOR
            ax = axes[row, column]
            ax.imshow(np.asarray(image).reshape(8, 8), cmap="gray_r", vmin=0, vmax=16, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0: ax.set_title("原始图像" if dim is None else f"PCA {dim} 维重建")
            if column == 0: ax.set_ylabel(f"数字 {digit}\n编号 {idx}")
    fig.suptitle("训练样本的 PCA 重建示例")
    save_figure(fig, "06_训练样本PCA重建.png")


def verify_preprocessing(X: np.ndarray, splits: dict, state: dict, prepared: dict) -> dict:
    scaled_train = X[splits["train"]] / PIXEL_DIVISOR
    centered = scaled_train - state["mean"]
    covariance = centered.T @ centered / (len(centered) - 1)
    vectors = state["components"]
    mean_error = float(np.max(np.abs(state["mean"] - scaled_train.mean(axis=0))))
    centered_mean_error = float(np.max(np.abs(centered.mean(axis=0))))
    orthogonality_error = float(np.max(np.abs(vectors @ vectors.T - np.eye(len(vectors)))))
    eigen_residual = float(np.max(np.abs(covariance @ vectors.T - vectors.T * state["explained_variance"])))
    full_reconstruction_error = float(np.max(np.abs(centered @ vectors.T @ vectors - centered)))
    if max(mean_error, centered_mean_error, orthogonality_error, eigen_residual, full_reconstruction_error) > 1e-10:
        raise ValueError("训练集均值、PCA 正交性、协方差特征关系或完整重建检查失败")
    for name, indices in splits.items():
        for suffix, dim in [("centered64", None), *[(f"pca{d}", d) for d in PCA_DIMENSIONS]]:
            array = prepared[f"X_{name}_{suffix}"]
            expected_width = 64 if dim is None else dim
            if array.shape != (len(indices), expected_width) or not np.isfinite(array).all():
                raise ValueError(f"{name}/{suffix} 形状或有限值检查失败")
    return {
        "status": "passed", "sample_index_coverage_and_disjointness": True,
        "cross_split_exact_duplicate_images": 0,
        "preprocessing_fit_subset": "train", "fit_n_samples": int(state["fit_n_samples"]),
        "training_mean_parameter_max_error": mean_error,
        "centered_training_mean_max_abs": centered_mean_error,
        "pca_orthogonality_max_abs_error": orthogonality_error,
        "training_covariance_eigen_equation_max_abs_residual": eigen_residual,
        "full_pca_reconstruction_max_abs_error": full_reconstruction_error,
        "all_transformed_arrays_finite_and_expected_shape": True,
        "no_classifier_trained": True,
    }


def write_data_report(audit: dict, counts: list, splits: dict, state: dict, validation: dict, versions: dict) -> None:
    train_raw = load_npz(DATA_DIR / "digits_raw.npz")["X"][splits["train"]]
    constant = np.flatnonzero(np.ptp(train_raw, axis=0) == 0).tolist()
    zero_fraction = float((train_raw == 0).mean())
    cumulative = np.cumsum(state["explained_variance_ratio"])
    count_rows = "\n".join("| " + " | ".join(map(str, row)) + " |" for row in counts)
    pca_rows = "\n".join(f"| {d} | {cumulative[d - 1] * 100:.4f}% | {(1 - d / 64) * 100:.1f}% |" for d in PCA_DIMENSIONS)
    shape_rows = "\n".join(f"| {name} | {len(indices)} | ({len(indices)}, 64) | ({len(indices)}, 8/16/32) |" for name, indices in splits.items())
    report = f"""# Digits 数据说明与探索分析

阶段 2 已完成。本报告记录实际读取的数据、固定划分和训练集探索结果，供后续 BP 与基线实验使用；当前尚未训练分类模型。

## 一 数据来源与实际核对

使用 [scikit-learn Digits](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)，原始出处为 [UCI Optical Recognition of Handwritten Digits](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits)。本项目加载 1797 个样本、64 个像素特征和数字 0—9 共 10 个类别，图像形状为 8×8。它是 UCI 原数据测试部分的副本；本项目重新进行样本分层划分。

实际像素范围为 {audit['pixel_min']:g}—{audit['pixel_max']:g}，像素均为整数值，矩阵类型为 {audit['X_dtype']}，标签类型为 {audit['y_dtype']}。NaN 数量为 {audit['nan_count']}，无穷值数量为 {audit['infinite_count']}，完全重复图像的额外样本数量为 {audit['extra_duplicate_samples']}，重复组中的冲突标签数量为 {audit['duplicate_groups_with_conflicting_labels']}。展开矩阵与原始图像逐项一致。

数据不需要缺失值填补或去重。训练集具有 {len(constant)} 个常数像素特征，零基索引为 {constant}；保持 64 维原始输入以遵守研究方案，不执行逐像素标准化，避免零方差除法。

## 二 固定数据划分与类别分布

先以种子 42 分层划出 270 个测试样本，再以种子 43 从剩余数据分层划出 270 个验证样本。训练、验证、测试数量分别为 1257、270、270。索引以生成时的顺序保存，后续训练自行按预设种子打乱。

| 数字 | 全部 | 训练 | 验证 | 测试 |
|---|---|---|---|---|
{count_rows}
| 合计 | 1797 | 1257 | 270 | 270 |

各类样本数介于 {min(audit['class_counts'])}—{max(audit['class_counts'])}，最大与最小类别数量比约为 {max(audit['class_counts']) / min(audit['class_counts']):.3f}，整体接近平衡。验证集和测试集每类约 27 个样本。本项目保留自然分布，不进行过采样或类别重加权。

三个子集的样本索引无重叠、完整覆盖数据，且跨子集完全重复图像数量为 0。此协议不具备独立书写者分组信息，不能据此证明对未见书写者的泛化能力。

## 三 预处理流程与数据接口

固定除以 16 进行归一化，仅在训练集上拟合逐像素均值和完整 SVD PCA。原始路径为 x/16−μ；降维路径为 (x/16−μ)V_d。PCA 不白化，维数保持预定的 8、16、32。所有子集复用同一训练集均值和主成分，不在验证或测试数据上重新拟合。

| 子集 | 样本数 | 中心化原始输入 | 各 PCA 输入 |
|---|---|---|---|
{shape_rows}

训练集中心化后各维均值的最大绝对值为 {validation['centered_training_mean_max_abs']:.3e}；验证集和测试集的均值通常不为零，这是复用训练参数后的正常现象，不应分别重新中心化。

`实验代码/data_utils.py` 提供 `fit_preprocessor`、`transform_inputs` 和 `load_prepared_data`。默认加载入口只读取训练和验证缓存；最终测试评价时显式选择包含测试数据。标签均保持 0—9 的整数格式，后续 BP 实现时再转换为独热表示。

## 四 训练集探索发现

图像、像素、类别均值和 PCA 探索只使用训练集；全体数据仅用于结构审查、重复检查和类别数量核对。训练样本展示按原始编号选取每类最早的 5 个样本，便于复现和追溯。

训练图像中零值像素占比为 {zero_fraction * 100:.2f}%。像素均值和变化主要由数字笔画位置决定，边缘部分位置变化较小。类别均值图展示了各数字的典型空间结构，具体易混淆类别仍需后续模型的混淆矩阵验证。

| 保留维数 | 训练集累计解释方差 | 相比 64 维减少的特征比例 |
|---|---|---|
{pca_rows}

达到 90% 和 95% 累计解释方差分别需要 {int(np.searchsorted(cumulative, 0.90) + 1)}、{int(np.searchsorted(cumulative, 0.95) + 1)} 个主成分。这些数值来自训练集；仅说明方差保留程度，不代表分类准确率或最优维数。继续保持阶段 1 预定的 8、16、32 维实验，不根据此图增加参数搜索。

PCA 重建图仅展示训练样本的表示变化，不是分类模型输出。显示时将色阶固定为 0—16，重建数组未通过显示裁剪改变。

## 五 图表与文件

图表存放于 `图表/阶段2/`，依次为：01 训练样本、02 类别与划分分布、03 训练集类别均值图、04 训练集像素统计、05 训练集 PCA 解释方差、06 训练样本 PCA 重建。

`数据/` 中包含原始数值快照、固定索引、逐样本划分清单、预处理参数、训练验证缓存、单独的测试缓存、类别统计、训练集像素统计、PCA 方差表、数据审查记录和验证报告。`data_manifest.json` 记录内容指纹、配置及环境。

阶段 2 的缓存便于后续开发。阶段 4 的正式耗时实验应调用训练集拟合入口重新计时，不把读取现成 PCA 缓存的时间当作完整预处理耗时。

## 六 验证与环境

- 固定划分：数量、唯一索引、完整覆盖及交集检查通过。
- 跨子集完全重复图像检查通过。
- 均值确由训练集计算，参数与独立均值计算的最大差为 {validation['training_mean_parameter_max_error']:.3e}。
- PCA 正交性最大误差为 {validation['pca_orthogonality_max_abs_error']:.3e}。
- 训练集协方差特征方程最大残差为 {validation['training_covariance_eigen_equation_max_abs_residual']:.3e}，用于核对主成分确实对应训练集协方差。
- 完整 PCA 重建最大误差为 {validation['full_pca_reconstruction_max_abs_error']:.3e}。
- 所有预处理数组形状和有限值检查通过。

Python {platform.python_version()}；NumPy {versions['numpy']}；SciPy {versions['scipy']}；scikit-learn {versions['scikit-learn']}；Matplotlib {versions['matplotlib']}。依赖在作业独立虚拟环境中安装，锁定文件与复现命令见《运行说明.md》。

下一阶段为阶段 3：BP 网络实现与正确性验证，使用训练与验证数据进行开发。
"""
    (PROJECT_DIR / "数据说明与探索分析.md").write_text(report, encoding="utf-8")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    dataset = load_digits()
    X, y = np.asarray(dataset.data, dtype=np.float64), np.asarray(dataset.target, dtype=np.int64)
    audit = audit_data(X, y, dataset.images)
    write_json(DATA_DIR / "dataset_audit.json", audit)
    splits = save_splits(X, y)
    with threadpool_limits(limits=1):
        state = fit_preprocessor(X[splits["train"]])
    prepared = {}
    for name, indices in splits.items():
        prepared[f"y_{name}"] = y[indices]
        prepared[f"indices_{name}"] = indices
        prepared[f"X_{name}_centered64"] = transform_inputs(X[indices], state)
        for dim in PCA_DIMENSIONS:
            prepared[f"X_{name}_pca{dim}"] = transform_inputs(X[indices], state, dim)
    validation = verify_preprocessing(X, splits, state, prepared)
    np.savez_compressed(DATA_DIR / "digits_raw.npz", X=X, y=y, images=dataset.images)
    np.savez_compressed(DATA_DIR / "preprocessor.npz", **state)
    for archive, subset_names in [("prepared_train_validation.npz", ("train", "validation")), ("prepared_test.npz", ("test",))]:
        arrays = {key: value for key, value in prepared.items() if any(key.startswith(f"X_{name}_") or key in (f"y_{name}", f"indices_{name}") for name in subset_names)}
        np.savez_compressed(DATA_DIR / archive, **arrays)
    counts = export_tables(X, y, splits, state)
    versions = {name: importlib.metadata.version(name) for name in ["numpy", "scipy", "scikit-learn", "matplotlib"]}
    manifest = {
        "generated_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "config": CONFIG, "dataset_content_sha256": audit["dataset_content_sha256"],
        "split_content_sha256": array_digest(*[splits[name] for name in SPLIT_NAMES]),
        "preprocessor_content_sha256": array_digest(state["mean"], state["components"]),
        "split_sizes": {name: len(indices) for name, indices in splits.items()},
        "versions": versions, "python": sys.version, "platform": platform.platform(),
        "processor": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", ""),
        "pca_fit_thread_limit": 1, "available_threadpools": threadpool_info(),
        "sources": ["https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html", "https://doi.org/10.24432/C50P49"],
        "prepared_array_shapes": {key: list(value.shape) for key, value in prepared.items()},
    }
    write_json(DATA_DIR / "data_config.json", CONFIG)
    write_json(DATA_DIR / "data_manifest.json", manifest)
    write_json(DATA_DIR / "validation_report.json", validation)
    create_figures(X, y, splits, state)
    write_data_report(audit, counts, splits, state, validation, versions)
    print(json.dumps({"stage": 2, "validation": validation["status"], "samples": len(y),
                      "class_counts": audit["class_counts"], "duplicates": audit["extra_duplicate_samples"],
                      "split_sizes": manifest["split_sizes"],
                      "pca_cumulative_variance": {str(d): float(state["explained_variance_ratio"][:d].sum()) for d in PCA_DIMENSIONS},
                      "figures": len(list(FIGURE_DIR.glob("*.png"))),
                      "report": str(PROJECT_DIR / "数据说明与探索分析.md")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
