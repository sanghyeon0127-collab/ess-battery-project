"""Raw .mat (MATLAB v7.3 / HDF5) → data/processed/ 분석용 캐시.

    cells.csv        셀 단위 메타 (batch, cell_id, policy, cycle_life, censored ...)
    summary.parquet  사이클 단위 요약 (QD, QC, IR, Tavg, Tmin, Tmax, chargetime)
    qdlin.npz        초기 N_CYCLES 사이클의 Qdlin (n_cells, N_CYCLES, 1000) + 전압축 Vdlin

struct 메모
- cycles[j] = cycle j+1. cycle 1은 비어 있음(원본 노트 : 샘플링 과다로 제외)
- cycle_life = 기록된 사이클 수 + 1 → EOL(0.88Ah) 미도달 셀은 '중단 시점'이 라벨로 들어가 있음(censored)
"""
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data" / "raw", ROOT / "data" / "processed"
BATCHES = {
    "b1": "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
    "b2": "2018-02-20_batchdata_updated_struct_errorcorrect.mat",
    "b3": "2018-04-12_batchdata_updated_struct_errorcorrect.mat",
}
N_CYCLES = 100          # Regression 입력 = 초기 100 사이클
CENSOR_QD = 0.90        # EOL(0.88Ah) 도달 셀의 말단 QD는 0.881~0.886, 미도달 셀은 0.915+ → 사이 값으로 구분
SUMMARY = {"cycle": "cycle", "QDischarge": "QD", "QCharge": "QC", "IR": "IR",
           "Tavg": "Tavg", "Tmin": "Tmin", "Tmax": "Tmax", "chargetime": "chargetime"}


def _str(f, ref):
    return f[ref][()].tobytes()[::2].decode()


def load_batch(name, path):
    cells, summaries, qdlin = [], [], []
    with h5py.File(path, "r") as f:
        b = f["batch"]
        vdlin = f[b["Vdlin"][0, 0]][()].ravel()
        for i in range(b["summary"].shape[0]):
            cell_id = f"{name}c{i}"
            s = f[b["summary"][i, 0]]
            summ = pd.DataFrame({new: s[old][()].ravel() for old, new in SUMMARY.items()})
            summ.insert(0, "cell_id", cell_id)
            summaries.append(summ)

            cyc = f[b["cycles"][i, 0]]
            q = np.full((N_CYCLES, 1000), np.nan, np.float32)
            for j in range(min(N_CYCLES, cyc["Qdlin"].shape[0])):
                arr = f[cyc["Qdlin"][j, 0]][()].ravel()
                if arr.size == 1000:
                    q[j] = arr
            qdlin.append(q)

            qd = summ["QD"].to_numpy()
            cells.append({
                "batch": name, "cell_id": cell_id,
                "policy": _str(f, b["policy_readable"][i, 0]).replace("-newstructure", ""),
                "cycle_life": float(f[b["cycle_life"][i, 0]][()].ravel()[0]),
                "n_cycles": len(summ),
                "final_QD": float(np.median(qd[-5:])),
            })
    cells = pd.DataFrame(cells)
    # 마지막 5사이클 중앙값이 EOL보다 확실히 위 = 수명 종료 전에 시험이 끝난 셀 (라벨 = 중단 시점)
    cells["censored"] = (cells["final_QD"] > CENSOR_QD) | cells["cycle_life"].isna()
    return cells, pd.concat(summaries, ignore_index=True), np.stack(qdlin), vdlin


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cells, summaries, qdlins, ids = [], [], [], []
    for name, fname in BATCHES.items():
        c, s, q, vdlin = load_batch(name, RAW / fname)
        print(f"{name}: {len(c)} cells, censored={c['censored'].sum()}, "
              f"cycle_life {c['cycle_life'].min():.0f}~{c['cycle_life'].max():.0f}")
        cells.append(c)
        summaries.append(s)
        qdlins.append(q)
        ids += c["cell_id"].tolist()
    pd.concat(cells, ignore_index=True).to_csv(OUT / "cells.csv", index=False)
    pd.concat(summaries, ignore_index=True).to_parquet(OUT / "summary.parquet", index=False)
    np.savez_compressed(OUT / "qdlin.npz", qdlin=np.concatenate(qdlins), cell_id=np.array(ids), vdlin=vdlin)


if __name__ == "__main__":
    main()
