"""BP 正确性检查：独立数值梯度、极端 logits、早停、复现与小样本学习。"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from bp_network import BPNetwork, TrainingConfig, classification_metrics, softmax_cross_entropy


def gradient_check() -> list[dict]:
    """逐项扰动全部参数，用中心差分估计损失导数。"""
    records = []
    epsilon = 1e-6
    for case, (d, h, c, m) in enumerate([(4, 5, 3, 7), (8, 4, 10, 5)]):
        model = BPNetwork(d, h, c, seed=29 + case)
        rng = np.random.default_rng(117 + case)
        # 中心差分不跨过 ReLU 的零点，避免把不可导问题误判为 BP 错误。
        for _ in range(100):
            X = rng.normal(size=(m, d))
            activation_margin = float(np.abs(model.forward(X)["Z1"]).min())
            if activation_margin > 1e-3:
                break
        else:
            raise AssertionError("无法构造远离 ReLU 零点的检查输入")
        y = np.arange(m, dtype=np.int64) % c
        _, analytic = model.loss_and_gradients(X, y)
        before = model.get_parameters()
        for name, array in model.parameters.items():
            numeric = np.empty_like(array)
            for index in np.ndindex(array.shape):
                original = float(array[index])
                try:
                    array[index] = original + epsilon
                    plus = model.evaluate(X, y)["loss"]
                    array[index] = original - epsilon
                    minus = model.evaluate(X, y)["loss"]
                finally:
                    array[index] = original
                numeric[index] = (plus - minus) / (2 * epsilon)
            difference = np.abs(analytic[name] - numeric)
            relative = difference / np.maximum(1e-8, np.abs(analytic[name]) + np.abs(numeric))
            np.testing.assert_allclose(analytic[name], numeric, rtol=1e-5, atol=1e-7)
            records.append({
                "case": f"{d}-{h}-{c}", "batch_size": m, "parameter": name,
                "checked_entries": array.size, "epsilon": epsilon,
                "max_absolute_error": float(difference.max()),
                "max_relative_error": float(relative.max()),
                "relu_min_activation_margin": activation_margin, "passed": True,
            })
        for name in before:
            np.testing.assert_array_equal(model.parameters[name], before[name])
    return records


def numerical_and_training_checks(output_dir: Path) -> dict:
    # 错误类别与最大 logits 相差很大，直接 log(softmax) 会遇到 log(0)。
    extreme = np.asarray([[1000.0, -1000.0, 0.0], [-10000.0, 10000.0, 0.0]])
    probabilities, loss = softmax_cross_entropy(extreme, np.asarray([1, 0]))
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=1e-15)
    np.testing.assert_allclose(loss, 11000.0, rtol=0, atol=1e-12)

    rng = np.random.default_rng(99)
    X = rng.normal(size=(7, 4))
    y = np.arange(7, dtype=np.int64) % 3
    model = BPNetwork(4, 5, 3, seed=17)
    base_loss, gradients = model.loss_and_gradients(X, y)
    repeated_loss, repeated_gradients = model.loss_and_gradients(np.tile(X, (3, 1)), np.tile(y, 3))
    np.testing.assert_allclose(base_loss, repeated_loss, rtol=0, atol=1e-14)
    for name in gradients:
        np.testing.assert_allclose(gradients[name], repeated_gradients[name], rtol=0, atol=1e-14)
    shifted, shifted_loss = softmax_cross_entropy(model.forward(X)["logits"] + 10000, y)
    np.testing.assert_allclose(shifted, model.predict_proba(X), rtol=0, atol=1e-11)
    np.testing.assert_allclose(shifted_loss, base_loss, rtol=0, atol=1e-11)
    before = model.get_parameters()
    model.step(gradients, learning_rate=0.05)
    for name in before:
        np.testing.assert_allclose(model.parameters[name], before[name] - 0.05 * gradients[name], rtol=0, atol=0)
    updated_loss = model.evaluate(X, y)["loss"]
    if updated_loss >= base_loss:
        raise AssertionError("小步梯度更新没有降低人工数据上的损失")

    # 7 个样本配批量 3，实际执行 3/3/1，覆盖末批只有一个样本的路径。
    config = TrainingConfig(max_epochs=12, batch_size=3, patience=20)
    first, second = BPNetwork(4, 5, 3, seed=17), BPNetwork(4, 5, 3, seed=17)
    first_result = first.fit(X, y, X, y, config)
    second_result = second.fit(X, y, X, y, config)
    if first_result.history != second_result.history:
        raise AssertionError("相同配置与种子的训练曲线不一致")
    for name in first.parameters:
        np.testing.assert_array_equal(first.parameters[name], second.parameters[name])
    model_file = output_dir / "validation_roundtrip.npz"
    first.save(model_file)
    loaded = BPNetwork.load(model_file)
    np.testing.assert_array_equal(loaded.predict_proba(X), first.predict_proba(X))

    # 人工构造训练/验证标签相反的情况，确保早停被触发且恢复第 1 轮参数。
    stop_model = BPNetwork(2, 3, 2, seed=8)
    stop_model.set_parameters({"W1": np.zeros((2, 3)), "b1": np.ones(3),
                               "W2": np.zeros((3, 2)), "b2": np.zeros(2)})
    initial = stop_model.get_parameters()
    stop_X = np.ones((4, 2))
    train_y, validation_y = np.zeros(4, dtype=np.int64), np.ones(4, dtype=np.int64)
    stopped = stop_model.fit(stop_X, train_y, stop_X, validation_y,
                             TrainingConfig(learning_rate=0.1, batch_size=4, max_epochs=12, patience=3, min_delta=0))
    if not stopped.stopped_early or stopped.best_epoch != 1 or stopped.stopped_epoch != 4:
        raise AssertionError("早停轮数或保存检查点不符合预期")
    one_epoch = BPNetwork(2, 3, 2, seed=8)
    one_epoch.set_parameters(initial)
    one_epoch.fit(stop_X, train_y, stop_X, validation_y,
                  TrainingConfig(learning_rate=0.1, batch_size=4, max_epochs=1, patience=3, min_delta=0))
    for name in one_epoch.parameters:
        np.testing.assert_array_equal(stop_model.parameters[name], one_epoch.parameters[name])

    truth = np.asarray([0, 1, 2, 2, 0, 1])
    prediction = np.asarray([0, 1, 1, 2, 0, 2])
    own = classification_metrics(truth, prediction, n_classes=4)
    np.testing.assert_allclose(own["accuracy"], accuracy_score(truth, prediction), rtol=0, atol=1e-15)
    np.testing.assert_allclose(own["macro_f1"], f1_score(truth, prediction, labels=range(4), average="macro", zero_division=0), rtol=0, atol=1e-15)
    np.testing.assert_array_equal(own["confusion_matrix"], confusion_matrix(truth, prediction, labels=range(4)))
    return {
        "status": "passed", "extreme_logits_loss": loss,
        "probability_normalization": True, "logit_shift_invariance": True,
        "mean_gradient_batch_replication_invariance": True,
        "simultaneous_sgd_parameter_update": True, "single_step_loss_decreases": True,
        "short_last_batch_and_single_sample_batch": True,
        "same_seed_training_bitwise_reproducible": True,
        "model_save_load_predictions_exactly_equal": True,
        "early_stopping": {"best_epoch": stopped.best_epoch, "stopped_epoch": stopped.stopped_epoch,
                           "checkpoint_matches_one_epoch_model": True},
        "metrics_match_sklearn_including_absent_class": True,
    }


def small_sample_check(X_train: np.ndarray, y_train: np.ndarray, original_indices: np.ndarray) -> tuple[dict, list[dict]]:
    # 每类选 2 个训练样本，共 20 个，用于检查模型能否记住一个极小训练集。
    selected = np.concatenate([np.flatnonzero(y_train == digit)[:2] for digit in range(10)])
    X, y = X_train[selected], y_train[selected]
    model = BPNetwork(64, 32, 10, seed=123)
    initial = model.evaluate(X, y)
    config = TrainingConfig(learning_rate=0.1, batch_size=20, max_epochs=800, patience=80, min_delta=1e-5)
    # 此处监控同一小集合的损失，仅检查学习能力，不作泛化或早停选择结论。
    result = model.fit(X, y, X, y, config)
    final = model.evaluate(X, y)
    if final["accuracy"] != 1.0 or final["loss"] >= 0.05:
        raise AssertionError(f"20 样本学习检查失败：accuracy={final['accuracy']}, loss={final['loss']}")
    return {
        "status": "passed", "n_samples": len(selected), "samples_per_class": 2,
        "source": "training_subset_only", "original_indices": original_indices[selected].tolist(),
        "initial_loss": initial["loss"], "final_loss": final["loss"],
        "initial_accuracy": initial["accuracy"], "final_accuracy": final["accuracy"],
        "best_epoch": result.best_epoch, "stopped_epoch": result.stopped_epoch,
        "config": asdict(config), "generalization_evaluation": False,
    }, result.history
