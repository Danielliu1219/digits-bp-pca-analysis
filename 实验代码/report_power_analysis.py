"""精确McNemar检验的设计敏感性计算，不拟合模型或改动原始预测。"""
from pathlib import Path
import csv
import json
from math import comb

import numpy as np
from scipy.optimize import brentq
from scipy.special import gammaln, xlogy
from scipy.stats import binom

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "结果/补充实验_20261111"
N, ALPHA, TARGET = 270, 0.05, 0.80


def rejection_regions(n=N):
    regions = []
    for m in range(n + 1):
        regions.append(np.array([b for b in range(m + 1)
                                if min(1.0, 2 * sum(comb(m, j) for j in range(min(b, m-b) + 1)) / 2**m) <= ALPHA], dtype=int))
    assert all(len(regions[m]) == 0 for m in range(6))
    assert regions[6].tolist() == [0, 6]
    return regions


REGIONS = rejection_regions()


def power(q, delta):
    """M~Bin(N,q), B|M~Bin(M,(1+delta/q)/2)，在精确双侧拒绝域求和。"""
    assert 0 < q <= 1 and abs(delta) <= q
    theta = (1 + delta / q) / 2
    return float(sum(binom.pmf(m, N, q) * np.sum(binom.pmf(region, m, theta))
                     for m, region in enumerate(REGIONS)))


def multinomial_cross_check(q, delta):
    """独立按三项分布枚举仅前者正确、仅后者正确、其余样本。"""
    p10, p01 = (q + delta) / 2, (q - delta) / 2
    probability = 0.0
    for m, region in enumerate(REGIONS):
        for b in region:
            c, same = m - b, N - m
            log_coefficient = gammaln(N + 1) - gammaln(b + 1) - gammaln(c + 1) - gammaln(same + 1)
            probability += np.exp(log_coefficient + xlogy(b, p10) + xlogy(c, p01) + xlogy(same, 1-q))
    return float(probability)


def main():
    rows = []
    for expected in range(2, 9):
        q = expected / N
        maximum = power(q, q)
        # 所有分歧同向时，拒绝恰好要求至少6个分歧。
        np.testing.assert_allclose(maximum, binom.sf(5, N, q), rtol=0, atol=1e-13)
        mde = brentq(lambda delta: power(q, delta) - TARGET, 0, q, xtol=1e-14) if maximum >= TARGET else None
        rows.append({"expected_discordant_count": expected, "q": q,
                     "maximum_power_all_disagreements_one_direction": maximum,
                     "mde_accuracy_difference_pp_at_80pct_power": mde * 100 if mde is not None else None})
    delta = rows[-1]["mde_accuracy_difference_pp_at_80pct_power"] / 100
    independent = multinomial_cross_check(8 / N, delta)
    np.testing.assert_allclose(independent, TARGET, rtol=0, atol=1e-11)
    result = {"status": "passed", "n_independent_paired_examples": N, "two_sided_alpha": ALPHA,
              "target_power": TARGET, "method": "exact binomial-mixture enumeration; independent multinomial check",
              "interpretation": "假定总分歧率的设计敏感性情景；不是由本次p值反推真实效应或实际功效。",
              "observed_primary_table_discordant_range": [2, 8],
              "minimum_observed_significant_pattern": {"b": 6, "c": 0, "p": 0.03125,
                                                         "accuracy_difference_pp": 6 / N * 100},
              "svm_seed0_pattern": {"b": 5, "c": 0, "p": 0.0625},
              "scenarios": rows, "independent_multinomial_power_at_mde": independent,
              "statistical_method_reference": "https://support.sas.com/documentation/cdl/en/statug/67523/HTML/default/statug_power_details55.htm"}
    (OUT / "power_sensitivity.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (OUT / "power_sensitivity.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"status": "passed", "mde_pp_q8_over_270": rows[-1]["mde_accuracy_difference_pp_at_80pct_power"],
                      "max_power_q7_over_270": rows[-2]["maximum_power_all_disagreements_one_direction"],
                      "multinomial_cross_check": independent}, ensure_ascii=False))


if __name__ == "__main__":
    main()
