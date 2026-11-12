"""Digits 数据划分与预处理；供后续 BP 和基线实验共用。"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "数据"
PIXEL_DIVISOR = 16.0
PCA_DIMENSIONS = (8, 16, 32)


def array_digest(*arrays: np.ndarray) -> str:
    """对数组内容计算指纹，不受压缩文件时间戳影响。"""
    digest = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(array)
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def make_fixed_split(y: np.ndarray) -> dict[str, np.ndarray]:
    """两次分层划分，严格采用研究方案中的数量、种子和调用顺序。"""
    indices = np.arange(len(y), dtype=np.int64)
    remaining, test = train_test_split(
        indices, test_size=270, stratify=y, random_state=42
    )
    train, validation = train_test_split(
        remaining, test_size=270, stratify=y[remaining], random_state=43
    )
    return {"train": train, "validation": validation, "test": test}


def check_split(splits: dict[str, np.ndarray], n_samples: int) -> None:
    expected_sizes = {"train": 1257, "validation": 270, "test": 270}
    for name, expected in expected_sizes.items():
        indices = splits[name]
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError(f"{name} 索引必须是一维整数数组")
        if len(indices) != expected or len(np.unique(indices)) != expected:
            raise ValueError(f"{name} 数量不符或存在重复索引")
    combined = np.concatenate([splits[name] for name in expected_sizes])
    if not np.array_equal(np.sort(combined), np.arange(n_samples)):
        raise ValueError("三个子集必须无重叠且完整覆盖原始数据")


def _validate_pixels(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != 64 or len(X) == 0:
        raise ValueError("输入应为非空的 n×64 原始像素矩阵")
    if not np.isfinite(X).all() or X.min() < 0 or X.max() > PIXEL_DIVISOR:
        raise ValueError("原始像素必须为 0—16 范围内的有限数值")
    return X


def fit_preprocessor(
    X_train_raw: np.ndarray, n_components: int | None = None
) -> dict[str, np.ndarray]:
    """仅传入训练集进行拟合。默认保留完整 PCA，便于阶段 2 的方差分析。

    阶段 4 若需对某个维数单独计时，可传入 n_components=d，重新拟合。
    返回纯数值状态，可保存为 npz，无须 pickle。
    """
    X_train_raw = _validate_pixels(X_train_raw)
    pca = PCA(n_components=n_components, svd_solver="full", whiten=False)
    pca.fit(X_train_raw / PIXEL_DIVISOR)
    return {
        "pixel_divisor": np.asarray(PIXEL_DIVISOR, dtype=np.float64),
        "mean": pca.mean_.copy(),
        "components": pca.components_.copy(),
        "explained_variance": pca.explained_variance_.copy(),
        "explained_variance_ratio": pca.explained_variance_ratio_.copy(),
        "singular_values": pca.singular_values_.copy(),
        "fit_n_samples": np.asarray(len(X_train_raw), dtype=np.int64),
    }


def transform_inputs(
    X_raw: np.ndarray,
    state: dict[str, np.ndarray],
    pca_dim: int | None = None,
) -> np.ndarray:
    """只做变换；pca_dim=None 返回中心化后的原始 64 维输入。"""
    X_raw = _validate_pixels(X_raw)
    centered = X_raw / float(state["pixel_divisor"]) - state["mean"]
    if pca_dim is None:
        return np.ascontiguousarray(centered)
    if pca_dim not in PCA_DIMENSIONS:
        raise ValueError(f"预设 PCA 维数为 {PCA_DIMENSIONS}")
    if pca_dim > len(state["components"]):
        raise ValueError("保存的主成分数量不足")
    return np.ascontiguousarray(centered @ state["components"][:pca_dim].T)


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def load_prepared_data(
    *, include_test: bool = False, data_dir: Path = DATA_DIR
) -> dict[str, np.ndarray]:
    """默认只读取训练与验证数据，阶段 3 直接使用此默认入口。

    X_train_centered64 / X_validation_centered64 为原始维数的输入；
    X_train_pca8 等为降维输入；y_train 等为整数标签。
    方案冻结后的最终评价可显式 include_test=True。
    """
    prepared = load_npz(data_dir / "prepared_train_validation.npz")
    if include_test:
        prepared.update(load_npz(data_dir / "prepared_test.npz"))
    return prepared
