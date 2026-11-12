"""阶段 5：只读取冻结结果，计算可追溯的汇总、误差分析和正式图表。"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score, f1_score

from data_utils import PROJECT_DIR, DATA_DIR, load_npz
from run_experiments import validate_freeze

SOURCE = PROJECT_DIR / "结果" / "阶段4"
OUT = PROJECT_DIR / "结果" / "阶段5"
FIG = PROJECT_DIR / "图表" / "阶段5"
MODES = ["raw64", "pca8", "pca16", "pca32"]
LABELS = ["原始64维", "PCA 8维", "PCA 16维", "PCA 32维"]
COLORS = ["#364f6b", "#c46b4a", "#669787", "#8a769f"]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=210, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / f"{name}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def load_history(mode):
    path = SOURCE / "runs" / f"E3_{mode}_seed0" / "history.csv"
    return np.genfromtxt(path, delimiter=",", names=True, encoding="utf-8-sig")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    frozen = validate_freeze(SOURCE)
    summary = read_json(SOURCE / "experiment_summary.json")
    assert frozen["representative_seed_for_stage5"] == 0
    digest = hashlib.sha256((SOURCE / "frozen_protocol.json").read_bytes()).hexdigest()
    assert digest == summary["frozen_protocol_sha256"]
    with (SOURCE / "test_results.csv").open(encoding="utf-8-sig", newline="") as stream:
        test_rows = list(csv.DictReader(stream))
    by_id = {r["run_id"]: r for r in test_rows}
    images = load_npz(DATA_DIR / "digits_raw.npz")["images"]
    groups = summary["bp_groups"]
    table_rows = []
    baseline = summary["baseline"]
    table_rows.append({"model": "逻辑回归", "n_runs": 1, "input_dim": 64,
                       "test_accuracy_mean": baseline["test_accuracy"], "test_accuracy_std": 0.,
                       "test_macro_f1_mean": baseline["test_macro_f1"], "test_macro_f1_std": 0.,
                       "n_parameters": baseline["n_parameters"], "pipeline_parameter_count": baseline["pipeline_parameter_count"],
                       "pipeline_training_seconds_mean": baseline["pipeline_training_seconds"], "pipeline_training_seconds_std": 0.})
    for g, label in zip(groups, LABELS):
        table_rows.append({"model": "BP " + label, **{k: g[k] for k in list(table_rows[0])[1:]}})
    # 每个最终预测都重新计算准确率、F1；汇总和样本标准差独立重算。
    for r in test_rows:
        pred = load_npz(SOURCE / "runs" / r["run_id"] / "predictions_test.npz")
        assert len(pred["y"]) == 270
        np.testing.assert_allclose(accuracy_score(pred["y"], pred["predictions"]), float(r["test_accuracy"]), atol=1e-14)
        np.testing.assert_allclose(f1_score(pred["y"], pred["predictions"], average="macro"), float(r["test_macro_f1"]), atol=1e-14)
    for mode, g in zip(MODES, groups):
        rows = [by_id[f"E3_{mode}_seed{s}"] for s in (0, 1, 2)]
        for metric in ["test_accuracy", "test_macro_f1", "pipeline_training_seconds"]:
            values = np.array([float(r[metric]) for r in rows])
            np.testing.assert_allclose(values.mean(), g[metric + "_mean"], atol=1e-14)
            np.testing.assert_allclose(values.std(ddof=1), g[metric + "_std"], atol=1e-14)
    write_csv(OUT / "报告主结果表.csv", table_rows)

    representatives = [("E2_C1", "逻辑回归"), ("E3_raw64_seed0", "BP 原始64维"), ("E3_pca32_seed0", "BP PCA32维")]
    per_class, errors, rep_summary, pairs = [], [], [], []
    prediction_sets = {}
    for run_id, label in representatives:
        p = load_npz(SOURCE / "runs" / run_id / "predictions_test.npz")
        prediction_sets[run_id] = p
        cm = confusion_matrix(p["y"], p["predictions"], labels=np.arange(10))
        assert cm.sum() == 270
        np.savetxt(OUT / f"混淆矩阵_{run_id}.csv", cm, delimiter=",", fmt="%d", header="predicted_0,1,2,3,4,5,6,7,8,9", comments="")
        precision, recall, f1, support = precision_recall_fscore_support(p["y"], p["predictions"], labels=np.arange(10), zero_division=0)
        for c in range(10):
            per_class.append({"run_id": run_id, "class": c, "support": int(support[c]), "precision": float(precision[c]), "recall": float(recall[c]), "f1": float(f1[c])})
        wrong = np.flatnonzero(p["y"] != p["predictions"])
        for i in sorted(wrong, key=lambda j: int(p["indices"][j])):
            errors.append({"run_id": run_id, "original_index": int(p["indices"][i]), "true_label": int(p["y"][i]), "prediction": int(p["predictions"][i]), "max_probability": float(p["probabilities"][i].max()), "test_position": int(i)})
        error_pairs = sorted([(int(cm[a, b]), a, b) for a in range(10) for b in range(10) if a != b and cm[a, b]], key=lambda x: (-x[0], x[1], x[2]))
        for count, a, b in error_pairs:
            pairs.append({"run_id": run_id, "true_label": a, "prediction": b, "count": count})
        rep_summary.append({"run_id": run_id, "label": label, "n_errors": len(wrong), "accuracy": float(np.trace(cm) / cm.sum()), "error_pairs": [{"true": a, "pred": b, "count": count} for count, a, b in error_pairs]})
    write_csv(OUT / "逐类指标.csv", per_class)
    write_csv(OUT / "全部误分类样本.csv", errors)
    write_csv(OUT / "误分类类别对.csv", pairs)
    raw_errors = {r["original_index"] for r in errors if r["run_id"] == "E3_raw64_seed0"}
    pca_errors = {r["original_index"] for r in errors if r["run_id"] == "E3_pca32_seed0"}
    overlap = {"both_wrong_indices": sorted(raw_errors & pca_errors), "raw_only_wrong_indices": sorted(raw_errors - pca_errors), "pca_only_wrong_indices": sorted(pca_errors - raw_errors)}

    available = {f.name for f in font_manager.fontManager.ttflist}
    font = next(n for n in ("Microsoft YaHei", "SimHei", "SimSun") if n in available)
    plt.rcParams.update({"font.family": font, "font.size": 11, "axes.unicode_minus": False,
                         "axes.spines.top": False, "axes.spines.right": False, "figure.constrained_layout.use": True,
                         "axes.titlepad": 10, "axes.labelpad": 6, "legend.frameon": False})

    fig, axs = plt.subplots(2, 2, figsize=(10, 6.2), sharex=False, sharey=True)
    for ax, mode, label in zip(axs.flat, MODES, LABELS):
        h = load_history(mode)
        record = read_json(SOURCE / "runs" / f"E3_{mode}_seed0" / "record.json")
        ax.plot(h["epoch"], h["train_loss"], color=COLORS[0], label="训练集")
        ax.plot(h["epoch"], h["validation_loss"], color=COLORS[1], label="验证集")
        ax.axvline(record["best_epoch"], color="#777777", ls="--", lw=1, label="恢复检查点")
        ax.set(title=f'{label}  检查点{record["best_epoch"]}轮 / 停止{record["stopped_epoch"]}轮', xlabel="训练轮次", ylabel="交叉熵损失")
        ax.set_ylim(0, 2.7)
        ax.grid(alpha=.18)
    axs[0, 0].legend(fontsize=9)
    save(fig, "01_训练与验证损失")

    h = load_history("raw64")
    r = read_json(SOURCE / "runs" / "E3_raw64_seed0" / "record.json")
    fig, ax = plt.subplots(figsize=(8.6, 3.4))
    ax.plot(h["epoch"], h["train_accuracy"] * 100, label="训练集", color=COLORS[0])
    ax.plot(h["epoch"], h["validation_accuracy"] * 100, label="验证集", color=COLORS[1])
    ax.axvline(r["best_epoch"], ls="--", lw=1, color="#777777", label="恢复第136轮检查点")
    ax.set(xlabel="训练轮次", ylabel="准确率 %", title="原始64维 BP 训练过程  固定代表种子0", ylim=(40, 101))
    ax.legend(loc="lower right", ncol=3, fontsize=10)
    ax.grid(alpha=.2)
    save(fig, "02_原始BP准确率曲线")

    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    labels = ["逻辑回归", "BP\n原始64维", "BP\nPCA8维", "BP\nPCA16维", "BP\nPCA32维"]
    for ax, metric, title in zip(axs, ["test_accuracy", "test_macro_f1"], ["测试准确率 %", "测试宏平均F1 ×100"]):
        for i, row in enumerate(table_rows):
            mu, sd = row[metric + "_mean"] * 100, row[metric + "_std"] * 100
            ax.errorbar(i, mu, yerr=sd if i else None, fmt="s" if i == 0 else "o", color="#252525" if i == 0 else COLORS[i - 1], capsize=4, ms=6)
            if i:
                vals = [float(by_id[f"E3_{MODES[i - 1]}_seed{s}"][metric]) * 100 for s in (0, 1, 2)]
                ax.scatter(np.array([i - .12, i, i + .12]), vals, color=COLORS[i - 1], alpha=.5, s=16)
            ax.text(i, mu + sd + .45, f"{mu:.2f}", ha="center", fontsize=9)
        ax.set(xticks=range(5), xticklabels=labels, ylabel=title, ylim=(89, 100.5))
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("同一270样本测试集  BP为三次均值及样本标准差  浅色点为单次结果", fontsize=11)
    save(fig, "03_测试指标比较")

    order = [1, 2, 3, 0]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    x = np.arange(4)
    labs = ["PCA8", "PCA16", "PCA32", "原始64"]
    axs[0].bar(x, [groups[i]["cumulative_explained_variance"] * 100 for i in order], color=[COLORS[i] for i in order], width=.55)
    for j, i in enumerate(order):
        axs[0].text(j, groups[i]["cumulative_explained_variance"] * 100 + 1.5, f'{groups[i]["cumulative_explained_variance"] * 100:.2f}%', ha="center", fontsize=10)
    axs[0].set(ylabel="训练集累计解释方差 %", ylim=(0, 112), xticks=x, xticklabels=labs)
    for metric, label, color in [("validation_accuracy", "验证集", COLORS[0]), ("test_accuracy", "测试集", COLORS[1])]:
        axs[1].errorbar(x, [groups[i][metric + "_mean"] * 100 for i in order], yerr=[groups[i][metric + "_std"] * 100 for i in order], marker="o", capsize=4, label=label, color=color)
    axs[1].set(ylabel="三次准确率均值及标准差 %", ylim=(89, 100), xticks=x, xticklabels=labs)
    axs[1].legend()
    axs[1].grid(alpha=.2)
    save(fig, "04_PCA维数与识别效果")

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    xx = np.arange(5)
    pre = [baseline["preprocessing_seconds"]] + [g["preprocessing_seconds_mean"] for g in groups]
    train = [baseline["model_training_seconds"]] + [g["model_training_seconds_mean"] for g in groups]
    axs[0].bar(xx, train, color=COLORS[0], label="模型训练 含逐轮验证")
    axs[0].bar(xx, pre, bottom=train, color=COLORS[1], label="预处理拟合及变换")
    axs[0].errorbar(xx[1:], [g["pipeline_training_seconds_mean"] for g in groups], yerr=[g["pipeline_training_seconds_std"] for g in groups], fmt="none", color="#333333", capsize=3)
    axs[0].set(ylabel="训练流程耗时 秒", ylim=(0, 2.8), xticks=xx, xticklabels=labels)
    axs[0].legend(fontsize=8)
    for i, row in enumerate(table_rows):
        axs[0].text(i, .08, f'{row["pipeline_training_seconds_mean"]:.3f}', ha="center", fontsize=8, color="#222222", bbox=dict(facecolor="white", alpha=.8, edgecolor="none", pad=1))
    axs[1].bar(xx - .18, [r["n_parameters"] for r in table_rows], width=.36, label="分类器参数", color=COLORS[0])
    axs[1].bar(xx + .18, [r["pipeline_parameter_count"] for r in table_rows], width=.36, label="含预处理状态的参数", color=COLORS[1])
    axs[1].set(ylabel="需保存的浮点参数个数", xticks=xx, xticklabels=labels, ylim=(0, 3400))
    axs[1].legend(fontsize=8)
    for i, row in enumerate(table_rows):
        axs[1].text(i + .18, row["pipeline_parameter_count"] + 65, str(row["pipeline_parameter_count"]), ha="center", fontsize=8)
    save(fig, "05_计算耗时与参数量")

    fig, axs = plt.subplots(1, 2, figsize=(10, 4.6))
    for ax, (run_id, label) in zip(axs, representatives[1:]):
        p = prediction_sets[run_id]
        cm = confusion_matrix(p["y"], p["predictions"], labels=np.arange(10))
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=29)
        for a in range(10):
            for b in range(10):
                ax.text(b, a, str(cm[a, b]), ha="center", va="center", fontsize=9, color="white" if cm[a, b] > 15 else ("#aa3d26" if a != b and cm[a, b] else "#333333"))
        ax.set(title=f'{label}  种子0  错{np.sum(p["y"] != p["predictions"])}例', xlabel="预测标签", ylabel="真实标签", xticks=range(10), yticks=range(10))
        for spine in ax.spines.values():
            spine.set_visible(False)
    save(fig, "06_代表模型混淆矩阵")

    fig, axs = plt.subplots(2, 6, figsize=(10, 4.1))
    selected_indices = {}
    for row_idx, (run_id, label) in enumerate(representatives[1:]):
        records = [e for e in errors if e["run_id"] == run_id][:6]
        selected_indices[run_id] = [r["original_index"] for r in records]
        for ax, e in zip(axs[row_idx], records):
            ax.imshow(images[e["original_index"]], cmap="gray_r", vmin=0, vmax=16, interpolation="nearest")
            ax.set_title(f'索引{e["original_index"]}\n真{e["true_label"]} → 预测{e["prediction"]}\npmax={e["max_probability"]:.2f}', fontsize=9)
            ax.set(xticks=[], yticks=[])
        axs[row_idx, 0].set_ylabel("原始64维" if row_idx == 0 else "PCA32维", fontsize=11)
    fig.suptitle("固定种子0  各模型错误按原始索引升序展示前6例", fontsize=11)
    save(fig, "07_误分类样本")

    details = {"frozen_protocol_sha256": digest, "n_test": 270, "representative_seed": 0,
               "error_selection_rule": "each model errors sorted by original_index, first 6",
               "representatives": rep_summary, "error_overlap": overlap, "displayed_error_indices": selected_indices,
               "main_table": table_rows, "bp_groups": groups,
               "accuracy_difference_bp_raw_minus_lr_percentage_points": (groups[0]["test_accuracy_mean"] - baseline["test_accuracy"]) * 100,
               "pca32_accuracy_difference_from_raw_percentage_points": (groups[3]["test_accuracy_mean"] - groups[0]["test_accuracy_mean"]) * 100,
               "pca32_network_parameter_reduction_percent": (1 - 698 / 1210) * 100}
    write_json(OUT / "analysis_summary.json", details)
    figure_captions = [
        ("01_训练与验证损失", "四种输入的训练与验证交叉熵曲线。均为预先指定的种子0；虚线为实际恢复的检查点，曲线包含停止前全部轮次。PCA8达到300轮上限。"),
        ("02_原始BP准确率曲线", "原始64维BP种子0的训练及验证准确率。第156轮停止并恢复第136轮；最终指标对应恢复后的模型。"),
        ("03_测试指标比较", "统一270例测试集的准确率与宏平均F1。BP为种子0、1、2的均值及样本标准差，浅色点为各次值；逻辑回归只拟合一次，无重复标准差。"),
        ("04_PCA维数与识别效果", "左为训练集累计解释方差，右为验证及测试准确率。原始64维列表示全部输入，100%仅为总方差参照；四种输入的BP隐藏宽度均为16。"),
        ("05_计算耗时与参数量", "左为预处理拟合变换加模型训练的实测耗时，BP误差线为三次样本标准差；右分别统计分类器参数和加上均值及PCA投影的参数。耗时不包含读写、绘图和测试。"),
        ("06_代表模型混淆矩阵", "固定代表种子0的两种BP模型，行是真实标签，列是预测标签，数字为样本数。每个矩阵共270例；此图不表示三次均值。"),
        ("07_误分类样本", "固定种子0，分别将原始64维与PCA32维BP的全部错误按原始数据索引排序，展示前6例。pmax为模型输出的最大Softmax概率，未作校准，不能视为真实正确概率。")]
    write_csv(OUT / "图表清单与图注.csv", [{"file": name + ".png", "caption": cap} for name, cap in figure_captions])
    lines = ["# BP 神经网络实验结果分析", "", "分析只读取阶段4冻结模型及预测，没有新增拟合或根据测试集调参。冻结协议 SHA256：`" + digest + "`。", "", "## 主要结果", "", "|方法|测试准确率|宏平均F1|分类器参数|完整流程参数|流程耗时 秒|", "|---|---:|---:|---:|---:|---:|"]
    for row in table_rows:
        repeated = row["n_runs"] == 3
        acc = f'{row["test_accuracy_mean"] * 100:.2f}%' + (f' ± {row["test_accuracy_std"] * 100:.2f}个百分点' if repeated else "")
        f1 = f'{row["test_macro_f1_mean"]:.4f}' + (f' ± {row["test_macro_f1_std"]:.4f}' if repeated else "")
        time = f'{row["pipeline_training_seconds_mean"]:.3f}' + (f' ± {row["pipeline_training_seconds_std"]:.3f}' if repeated else "")
        lines.append(f'|{row["model"]}|{acc}|{f1}|{row["n_parameters"]}|{row["pipeline_parameter_count"]}|{time}|')
    lines.extend(["", "BP均值与样本标准差来自同一划分的种子0、1、2，ddof=1；误差范围反映初始化及批次顺序变化，不代表独立数据抽样的置信区间。逻辑回归只有一次确定性拟合。每个测试样本对应0.3704个百分点。", "", "## 三个研究问题的回答", "", "1. **简单BP能否完成识别**：原始64→16→10网络使用1210个可训练参数，测试均值97.53%，说明本任务下单隐藏层模型已经能学习有效分类。种子0的训练/验证/测试准确率为99.76%/96.67%/97.41%，训练与验证存在差距，早停仅限制继续拟合，不能据此认定过拟合已经消失。", "", "2. **PCA维数与效果及开销**：8/16/32维分别保留训练方差67.22%/84.90%/96.68%，输入维数分别减少87.5%/75%/50%。8维测试均值降至92.10%；16与32维均值分别为96.79%与96.91%，接近但低于原始输入。训练方差大不直接保证类别区分信息充足。8维表现下降可能与丢失判别细节有关，本实验没有单独验证因果，也可能受统一学习率、固定隐藏宽度和300轮上限影响。PCA8种子0触及300轮，需在局限中说明。", "", "PCA32的网络参数由1210降为698，减少42.31%，但预处理还要保存64个均值和32×64个投影系数，总计2810个浮点参数，超过原始流程的1274。该维数的平均流程耗时1.719秒也高于原始输入1.526秒。PCA16平均1.247秒，存在较大波动，不能据此认定维数越少必然越快。参数个数不是压缩文件字节数，不包含元数据。", "", "3. **与逻辑回归的差别**：逻辑回归263/270例正确，准确率97.41%、宏平均F1为0.9738，流程耗时约0.030秒。BP原始输入均值仅高0.12个百分点，且来自三次而非一次，未作显著性检验，不能声称稳定优于逻辑回归。两者结果表明数据在现有表示下已具较强的线性可分性，但不能由高准确率推断严格线性可分。", "", "## 混淆与误分类", "", "代表种子0在查看测试结果前已冻结，不按准确率选取最好一次。下面均为单次模型，与前述三次均值分开。"])
    for rep in rep_summary:
        ps = "、".join(f'{p["true"]}→{p["pred"]}（{p["count"]}例）' for p in rep["error_pairs"])
        lines.append(f'\n- {rep["label"]}：{rep["accuracy"] * 100:.2f}%，错误{rep["n_errors"]}例；{ps}。')
    lines.extend(["", f'原始BP与PCA32共同错误索引为{overlap["both_wrong_indices"]}；只在原始BP错误为{overlap["raw_only_wrong_indices"]}；只在PCA32错误为{overlap["pca_only_wrong_indices"]}。这些差异说明两种表示改变了部分样本的决策，不能把所有新增错误直接归因为PCA。', "", "误分类展示遵循固定索引排序，各模型前6例，完整记录见结果/阶段5/全部误分类样本.csv。图像仅8×8，笔画连接、倾斜及局部形状差异可能造成视觉相似；具体原因属于解释假设，未做针对笔画的干预实验。Softmax最大概率没有校准，较高分数也可能对应错误。", "", "## 结论适用范围", "", "隐藏宽度16及逻辑回归C=1由验证集选择，PCA32是三个候选降维方案中验证集均值最高者；不能因原始64维测试均值最高而重选超参数。18次开发拟合、冻结后13次测试评价均沿用原计划。", "", "固定划分的270例测试集较小，三次种子只考察优化随机性；没有重复划分、交叉验证、书写者独立分组或外部数据，结论仅适用于当前Digits子集与实验协议。PCA没有逐维调整学习率和隐藏宽度，耗时也不是硬件无关的算法复杂度证明。后续若扩展，应在新的训练/验证协议下重新冻结方案，并保留独立测试集。", "", "## 图表与来源", ""])
    lines.extend(f'- `图表/阶段5/{name}.png`：{cap}' for name, cap in figure_captions)
    lines.extend(["", "全部图另存SVG便于排版。数据与逐类指标位于`结果/阶段5/`，生成入口为`实验代码/analyze_results.py`。核心模型、开发脚本与冻结记录未修改。", ""])
    (PROJECT_DIR / "结果分析.md").write_text("\n".join(lines), encoding="utf-8")
    artifacts = {str(p.relative_to(PROJECT_DIR)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [*sorted(OUT.glob("*.csv")), OUT / "analysis_summary.json", PROJECT_DIR / "结果分析.md", *sorted(FIG.glob("*"))]}
    validate_freeze(SOURCE)
    write_json(OUT / "verification_report.json", {"status": "passed", "n_test_predictions_verified": 13, "n_group_summaries_verified": 4, "n_figures": len(figure_captions), "frozen_protocol_sha256": digest, "representative_seed": 0, "source_models_unchanged": True, "artifact_sha256": artifacts})
    print(json.dumps({"status": "passed", "figures": len(figure_captions), "representatives": rep_summary, "error_overlap": overlap}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
