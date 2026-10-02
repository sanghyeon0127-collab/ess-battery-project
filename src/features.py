"""초기 100 사이클 → 셀 단위 피처 (Severson et al. 2019 기반).

ΔQ(V) = Q_100(V) - Q_10(V)  : 같은 전압에서의 방전 용량 차이. 열화가 빠른 셀일수록 곡선이 크게 벌어짐
"""
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

from preprocess import OUT

# cycle 번호 기준 (cycles[j] = cycle j+1)
C_LO, C_HI = 10, 100


def delta_q(qdlin, lo=C_LO, hi=C_HI):
    return qdlin[:, hi - 1] - qdlin[:, lo - 1]


def _fit(x, y):
    slope, intercept = np.polyfit(x, y, 1)
    return slope, intercept


def build(cells, summary, qdlin, cell_ids):
    dq = delta_q(qdlin)
    dq_df = pd.DataFrame({
        "cell_id": cell_ids,
        "dQ_log_var": np.log10(np.var(dq, axis=1)),
        "dQ_log_min": np.log10(np.abs(np.min(dq, axis=1))),
        "dQ_log_mean": np.log10(np.abs(np.mean(dq, axis=1))),
        "dQ_log_skew": np.log10(np.abs(skew(dq, axis=1))),
        "dQ_log_kurt": np.log10(np.abs(kurtosis(dq, axis=1))),
    })

    rows = []
    for cid, g in summary[summary["cycle"].between(2, C_HI)].groupby("cell_id"):
        g = g.set_index("cycle")
        g["IR"] = g["IR"].replace(0, np.nan)   # 0 = 미측정 (b2 일부 셀은 전 구간 0)
        qd = g["QD"]
        rows.append({
            "cell_id": cid,
            "QD_2": qd.loc[2],
            "QD_max_minus_2": qd.max() - qd.loc[2],
            "QD_100_minus_10": qd.loc[C_HI] - qd.loc[C_LO],
            **dict(zip(["fade_slope_2_100", "fade_icpt_2_100"], _fit(qd.index, qd.values))),
            **dict(zip(["fade_slope_91_100", "fade_icpt_91_100"], _fit(qd.loc[91:].index, qd.loc[91:].values))),
            "chargetime_2_6": g.loc[2:6, "chargetime"].mean(),
            "Tavg_mean": g["Tavg"].mean(),
            "Tmax_max": g["Tmax"].max(),
            "IR_min": g["IR"].min(),
            "IR_100_minus_2": g.loc[C_HI, "IR"] - g.loc[2, "IR"],
        })
    return cells.merge(dq_df, on="cell_id").merge(pd.DataFrame(rows), on="cell_id")


def load():
    """processed 캐시 → (cells, summary, qdlin, vdlin, cell_ids)"""
    z = np.load(OUT / "qdlin.npz")
    cells = pd.read_csv(OUT / "cells.csv")
    return cells, pd.read_parquet(OUT / "summary.parquet"), z["qdlin"], z["vdlin"], z["cell_id"]


def main():
    cells, summary, qdlin, _, ids = load()
    feats = build(cells, summary, qdlin, ids)
    feats.to_csv(OUT / "features.csv", index=False)
    print(feats.groupby("batch").size(), feats.shape)


if __name__ == "__main__":
    main()
