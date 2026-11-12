# 手写数字识别的 BP 与多基线比较及 PCA 降维影响分析

《数据分析的机器智能方法》期末作业复现仓库。

- 学生：刘添屹　2024141460484
- 最终报告：[Word 版](报告/2024141460484_刘添屹_BP神经网络手写数字识别课程报告.docx) ｜ [PDF 版](报告/2024141460484_刘添屹_BP神经网络手写数字识别课程报告.pdf)
- 数据：scikit-learn 自带 Digits（1797 例 8×8 手写数字图像），固定划分为 1257 例训练、270 例验证、270 例测试
- 方法：NumPy 单隐藏层 BP、逻辑回归、kNN、RBF-SVM；PCA 8/16/32 维对照；精确 McNemar 检验、Wilson 区间与功效分析

## 主要结果

同一 270 例测试集上的准确率：

| 模型 | 测试准确率 |
|---|---:|
| 逻辑回归（64 维） | 97.41% |
| BP 原始 64 维（3 种子） | 97.53% ± 0.57 |
| BP + PCA 8 维（3 种子） | 92.10% ± 1.13 |
| BP + PCA 16 维（3 种子） | 96.79% ± 0.86 |
| BP + PCA 32 维（3 种子） | 96.91% ± 0.57 |
| kNN，k=1 / k=3 | 98.15% / 98.52% |
| RBF-SVM（验证集选参） | 99.26% |

完整实验设置、图表、混淆矩阵、误分类分析与统计检验见最终报告。

## 仓库内容

```text
实验代码/     实验复现代码与依赖
报告/         最终 Word / PDF 报告
```

本仓库只保留复现所需的代码与最终报告。数据由代码自动生成；运行产生的 `数据/`、`结果/`、`图表/` 及阶段说明文档已加入 `.gitignore`，不纳入版本控制。

## 环境

- Python 3.12（报告实验在 Python 3.12.14 上完成）
- NumPy 2.5.4、SciPy 1.18.1、scikit-learn 1.9.1、Matplotlib 3.11.2
- 绘制中文图表需要微软雅黑、黑体或宋体之一
- 不需要下载数据集，也无需 GPU

创建并启用虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r .\实验代码\requirements-lock.txt
```

Linux / macOS：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r 实验代码/requirements-lock.txt
```

## 复现步骤

在本仓库根目录依次执行。所有脚本都根据自身位置定位项目根目录，输出分别写入 `数据/`、`结果/`、`图表/`。

```powershell
# 1. 数据审查、固定划分、训练集预处理与探索图
python -X utf8 实验代码/prepare_data.py

# 2. BP 实现检查、梯度验证与初始试运行
python -X utf8 实验代码/run_stage3.py

# 3. 18 次开发拟合、冻结配置，再评价 13 个测试模型
python -X utf8 实验代码/run_experiments.py

# 4. 独立核验阶段 4 的模型、预测与汇总
python -X utf8 实验代码/verify_experiments.py

# 5. 正式图表、汇总与误分类分析
python -X utf8 实验代码/analyze_results.py

# 6. 模型缺陷复查与参数敏感性检查
python -X utf8 实验代码/review_model.py
python -X utf8 实验代码/verify_model_review.py

# 7. kNN / RBF-SVM 追加基线、配对统计与功效分析
python -X utf8 实验代码/supplement_experiments.py
python -X utf8 实验代码/supplement_experiments.py --verify
python -X utf8 实验代码/plot_supplement.py
python -X utf8 实验代码/report_power_analysis.py
```

完整流程在普通笔记本电脑上通常需要数分钟。训练耗时会被重新测量，允许与报告中的原始机器耗时不同；数据数组、固定划分、验证选参、模型参数、预测、准确率、F1 和损失应保持一致。

## 代码说明

| 文件 | 作用 |
|---|---|
| `prepare_data.py` | Digits 审查、固定划分、训练集均值与 PCA、探索图 |
| `bp_network.py` | ReLU / Softmax / 交叉熵 / BP 梯度 / 小批量 SGD / 早停 |
| `validate_bp.py`、`run_stage3.py` | 梯度与训练流程检查、初始试运行 |
| `run_experiments.py` | 隐藏宽度与逻辑回归选参、冻结后测试评价 |
| `verify_experiments.py` | 基于保存模型和预测的独立核验 |
| `analyze_results.py` | 汇总表、混淆矩阵、误分类分析与正式图表 |
| `review_model.py`、`verify_model_review.py` | 复查网络宽度、学习率与 PCA8 训练预算 |
| `supplement_experiments.py` | 追加 kNN / RBF-SVM 基线、Wilson 区间和精确 McNemar 检验 |
| `plot_supplement.py`、`report_power_analysis.py` | 扩展比较图、检验功效与最小可检出差异 |
| `requirements*.txt` | 直接依赖与完整锁定依赖 |

## 说明

- 测试集只在方案冻结后读取；BP、逻辑回归、kNN 与 SVM 使用同一固定划分。
- 报告中的训练耗时来自最初一次运行；复现时只要求指标与预测一致，不要求计时相同。
- 代码在 Windows 上验证。Linux / macOS 若缺少上述中文字体，请先安装可用中文字体，或调整绘图脚本中的字体设置；这不影响数值实验。
- 最终报告的 Word 可编辑，PDF 为 A4 打印版。仓库不包含报告生成、排版与提交包打包脚本。
