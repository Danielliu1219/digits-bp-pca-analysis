"""绘制追加基线与原模型的测试指标比较图。"""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def load_rows():
    report=ROOT/"结果/补充实验_20261111/report_summary.json"
    if report.exists():
        return json.loads(report.read_text(encoding="utf-8"))["main_table"]
    analysis=json.loads((ROOT/"结果/阶段5/analysis_summary.json").read_text(encoding="utf-8"))
    supplement=json.loads((ROOT/"结果/补充实验_20261111/supplement_summary.json").read_text(encoding="utf-8"))
    baselines=[{"model":row["model"],"n_runs":1,"input_dim":64,
                "test_accuracy_mean":row["accuracy"],"test_accuracy_std":0,
                "test_macro_f1_mean":row["macro_f1"],"test_macro_f1_std":0}
               for row in supplement["baselines"]]
    return [analysis["main_table"][0],*baselines,*analysis["main_table"][1:]]


def main():
    rows=load_rows()
    out=ROOT/"图表/补充实验_20261111"
    out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.family":"Microsoft YaHei","axes.unicode_minus":False,"font.size":9})
    labels=["逻辑\n回归","kNN\nk=1","kNN\nk=3","RBF\nSVM","BP\n原始64维","BP\nPCA8维","BP\nPCA16维","BP\nPCA32维"]
    colors=["#555555","#529181","#28776c","#79558d","#364f6b","#c46b4a","#669787","#8a769f"]
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout="constrained")
    for ax,metric,label in zip(axes,["test_accuracy","test_macro_f1"],["测试准确率 %","测试宏平均F1 ×100"]):
        for index,row in enumerate(rows):
            mean=row[metric+"_mean"]*100
            std=row[metric+"_std"]*100
            ax.errorbar(index,mean,yerr=std,fmt="o" if row["n_runs"]==3 else "s",color=colors[index],capsize=4,markersize=5)
            ax.text(index,mean+std+0.3,f"{mean:.2f}",ha="center",va="bottom",fontsize=8)
        ax.set_xticks(np.arange(len(rows)),labels,fontsize=8)
        ax.set_ylim(90,101)
        ax.set_ylabel(label)
        ax.grid(axis="y",alpha=.2)
        ax.spines[["top","right"]].set_visible(False)
    fig.suptitle("同一270例测试集  BP为三次均值及样本标准差",fontsize=11)
    for extension in ("png","svg"):
        fig.savefig(out/("01_扩展测试指标比较."+extension),dpi=210,bbox_inches="tight",facecolor="white")
    plt.close(fig)
    print(str(out/"01_扩展测试指标比较.png"))


if __name__=="__main__":
    main()
