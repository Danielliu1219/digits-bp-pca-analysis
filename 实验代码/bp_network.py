"""单隐藏层多层感知机：NumPy 实现 ReLU、Softmax、交叉熵和 BP。

变量采用行样本约定：X=(m,d)，W1=(d,h)，W2=(h,c)。
BP 计算梯度，小批量 SGD 更新权重；本模块不读取任何数据文件。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _labels(y: np.ndarray, n_samples: int, n_classes: int) -> np.ndarray:
    y = np.asarray(y)
    if y.shape != (n_samples,) or not np.issubdtype(y.dtype, np.integer):
        raise ValueError("标签应为与样本一一对应的一维整数数组")
    if np.any(y < 0) or np.any(y >= n_classes):
        raise ValueError("标签超出模型类别范围")
    return y.astype(np.int64, copy=False)


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    """使用平移后的 logits 直接计算 log-softmax，避免 log(0)。"""
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2 or len(logits) == 0 or logits.shape[1] < 2:
        raise ValueError("logits 应为非空的 m×c 矩阵，且 c≥2")
    if not np.isfinite(logits).all():
        raise FloatingPointError("logits 出现 NaN 或无穷值")
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp_shifted = np.exp(shifted)
        return shifted - np.log(exp_shifted.sum(axis=1, keepdims=True))


def softmax_cross_entropy(logits: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    log_probs = _log_softmax(logits)
    y = _labels(y, len(log_probs), log_probs.shape[1])
    with np.errstate(under="ignore"):
        probabilities = np.exp(log_probs)
        loss = float(-log_probs[np.arange(len(y)), y].mean())
    if not np.isfinite(loss):
        raise FloatingPointError("交叉熵出现非有限值")
    return probabilities, loss


def classification_metrics(y: np.ndarray, prediction: np.ndarray, n_classes: int) -> dict:
    """固定类别集合的准确率、宏平均 F1 和混淆矩阵。"""
    y = _labels(y, len(y), n_classes)
    prediction = _labels(prediction, len(y), n_classes)
    if len(y) == 0:
        raise ValueError("评价样本不能为空")
    matrix = np.bincount(y * n_classes + prediction, minlength=n_classes ** 2).reshape(n_classes, n_classes)
    true_positive = np.diag(matrix).astype(np.float64)
    denominator = matrix.sum(axis=0) + matrix.sum(axis=1)
    class_f1 = np.divide(2 * true_positive, denominator, out=np.zeros(n_classes), where=denominator != 0)
    return {
        "accuracy": float(true_positive.sum() / len(y)),
        "macro_f1": float(class_f1.mean()),
        "confusion_matrix": matrix,
    }


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 0.05
    batch_size: int = 32
    max_epochs: int = 300
    patience: int = 20
    min_delta: float = 1e-4
    shuffle_seed: int | None = None

    def validate(self) -> None:
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("学习率必须为有限正数")
        if any(not isinstance(value, int) or value <= 0 for value in (self.batch_size, self.max_epochs, self.patience)):
            raise ValueError("批量大小、最大轮数及早停耐心值必须为正整数")
        if not np.isfinite(self.min_delta) or self.min_delta < 0:
            raise ValueError("最小改善阈值必须为有限非负数")


@dataclass
class TrainingResult:
    history: list[dict]
    best_epoch: int
    stopped_epoch: int
    stopped_early: bool
    best_validation_loss: float
    shuffle_seed: int


class BPNetwork:
    """d→h→c 网络。参数字典的四项分别对应研究方案中的 W1、b1、W2、b2。"""

    def __init__(self, input_dim: int = 64, hidden_dim: int = 32, n_classes: int = 10, seed: int = 42):
        if input_dim <= 0 or hidden_dim <= 0 or n_classes < 2:
            raise ValueError("输入/隐藏维数应为正，类别数至少为 2")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.n_classes = int(n_classes)
        self.seed = int(seed)
        rng = np.random.default_rng(self.seed)
        self.parameters = {
            "W1": rng.normal(0, np.sqrt(2 / self.input_dim), (self.input_dim, self.hidden_dim)),
            "b1": np.zeros(self.hidden_dim, dtype=np.float64),
            "W2": rng.normal(0, np.sqrt(1 / self.hidden_dim), (self.hidden_dim, self.n_classes)),
            "b2": np.zeros(self.n_classes, dtype=np.float64),
        }

    @property
    def n_parameters(self) -> int:
        return sum(array.size for array in self.parameters.values())

    def _features(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or len(X) == 0 or X.shape[1] != self.input_dim:
            raise ValueError(f"输入应为非空 n×{self.input_dim} 矩阵")
        if not np.isfinite(X).all():
            raise ValueError("输入包含 NaN 或无穷值")
        return X

    def _forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        p = self.parameters
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            Z1 = X @ p["W1"] + p["b1"]
            H = np.maximum(Z1, 0.0)
            logits = H @ p["W2"] + p["b2"]
        if not np.isfinite(logits).all():
            raise FloatingPointError("前向传播产生非有限值")
        return Z1, H, logits

    def forward(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """返回中间变量，便于教学解释与数值验证。"""
        X = self._features(X)
        Z1, H, logits = self._forward(X)
        with np.errstate(under="ignore"):
            probabilities = np.exp(_log_softmax(logits))
        return {"Z1": Z1, "H": H, "logits": logits, "probabilities": probabilities}

    def loss_and_gradients(self, X: np.ndarray, y: np.ndarray) -> tuple[float, dict[str, np.ndarray]]:
        X = self._features(X)
        y = _labels(y, len(X), self.n_classes)
        Z1, H, logits = self._forward(X)
        probabilities, loss = softmax_cross_entropy(logits, y)
        # Softmax + 平均交叉熵：D2=(P-Y)/m，末批也除以实际样本数。
        D2 = probabilities.copy()
        D2[np.arange(len(y)), y] -= 1.0
        D2 /= len(y)
        dW2 = H.T @ D2
        db2 = D2.sum(axis=0)
        # 这里仍然使用旧 W2，四组梯度计算完成后才允许统一更新。
        D1 = (D2 @ self.parameters["W2"].T) * (Z1 > 0)
        gradients = {"W1": X.T @ D1, "b1": D1.sum(axis=0), "W2": dW2, "b2": db2}
        if not all(np.isfinite(array).all() for array in gradients.values()):
            raise FloatingPointError("反向传播产生非有限梯度")
        return loss, gradients

    def step(self, gradients: dict[str, np.ndarray], learning_rate: float) -> None:
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("学习率必须为有限正数")
        if set(gradients) != set(self.parameters):
            raise ValueError("梯度必须包含全部四组参数")
        updated = {}
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            for name, old in self.parameters.items():
                gradient = np.asarray(gradients[name], dtype=np.float64)
                if gradient.shape != old.shape or not np.isfinite(gradient).all():
                    raise ValueError(f"{name} 梯度形状不符或包含非有限值")
                updated[name] = old - learning_rate * gradient
        if not all(np.isfinite(array).all() for array in updated.values()):
            raise FloatingPointError("参数更新产生非有限值")
        self.parameters = updated

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)["probabilities"]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        X = self._features(X)
        y = _labels(y, len(X), self.n_classes)
        _, _, logits = self._forward(X)
        probabilities, loss = softmax_cross_entropy(logits, y)
        result = classification_metrics(y, probabilities.argmax(axis=1), self.n_classes)
        return {"loss": loss, **result}

    def get_parameters(self) -> dict[str, np.ndarray]:
        return {name: value.copy() for name, value in self.parameters.items()}

    def set_parameters(self, values: dict[str, np.ndarray]) -> None:
        if set(values) != set(self.parameters):
            raise ValueError("参数名称不完整")
        copied = {}
        for name, old in self.parameters.items():
            value = np.asarray(values[name], dtype=np.float64)
            if value.shape != old.shape or not np.isfinite(value).all():
                raise ValueError(f"{name} 参数形状不符或包含非有限值")
            copied[name] = value.copy()
        self.parameters = copied

    def fit(
        self, X_train: np.ndarray, y_train: np.ndarray,
        X_validation: np.ndarray, y_validation: np.ndarray,
        config: TrainingConfig | None = None, *, verbose: bool = False,
    ) -> TrainingResult:
        config = config or TrainingConfig()
        config.validate()
        X_train, X_validation = self._features(X_train), self._features(X_validation)
        y_train = _labels(y_train, len(X_train), self.n_classes)
        y_validation = _labels(y_validation, len(X_validation), self.n_classes)
        shuffle_seed = self.seed + 1000 if config.shuffle_seed is None else config.shuffle_seed
        rng = np.random.default_rng(shuffle_seed)
        history = []

        def record(epoch: int, checkpoint_saved: bool = False) -> tuple[dict, dict]:
            train = self.evaluate(X_train, y_train)
            validation = self.evaluate(X_validation, y_validation)
            history.append({
                "epoch": epoch, "train_loss": train["loss"], "validation_loss": validation["loss"],
                "train_accuracy": train["accuracy"], "validation_accuracy": validation["accuracy"],
                "checkpoint_saved": int(checkpoint_saved),
            })
            return train, validation

        record(0)  # 第 0 轮只记录随机初始化，检查点从完成第 1 轮训练后开始选择。
        best_loss, best_epoch, wait = float("inf"), 0, 0
        best_parameters = None
        for epoch in range(1, config.max_epochs + 1):
            order = rng.permutation(len(X_train))
            for start in range(0, len(order), config.batch_size):
                batch = order[start:start + config.batch_size]
                _, gradients = self.loss_and_gradients(X_train[batch], y_train[batch])
                self.step(gradients, config.learning_rate)
            train, validation = record(epoch)
            if validation["loss"] < best_loss - config.min_delta:
                best_loss, best_epoch, wait = validation["loss"], epoch, 0
                best_parameters = self.get_parameters()
                history[-1]["checkpoint_saved"] = 1
            else:
                wait += 1
            if verbose and (epoch == 1 or epoch % 50 == 0 or wait >= config.patience):
                print(f"epoch={epoch:3d} train_loss={train['loss']:.6f} val_loss={validation['loss']:.6f} "
                      f"train_acc={train['accuracy']:.4f} val_acc={validation['accuracy']:.4f}", flush=True)
            if wait >= config.patience:
                break
        self.set_parameters(best_parameters)
        return TrainingResult(history, best_epoch, epoch, wait >= config.patience, best_loss, int(shuffle_seed))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, schema_version=np.asarray(1, dtype=np.int64),
            input_dim=np.asarray(self.input_dim), hidden_dim=np.asarray(self.hidden_dim),
            n_classes=np.asarray(self.n_classes), seed=np.asarray(self.seed), **self.parameters,
        )

    @classmethod
    def load(cls, path: str | Path) -> BPNetwork:
        with np.load(path, allow_pickle=False) as archive:
            if int(archive["schema_version"]) != 1:
                raise ValueError("不支持的模型文件版本")
            model = cls(int(archive["input_dim"]), int(archive["hidden_dim"]), int(archive["n_classes"]), int(archive["seed"]))
            model.set_parameters({name: archive[name] for name in model.parameters})
        return model
