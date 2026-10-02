"""Batch 2 성능 개선 실험 — README '개선 방향' 구현. 최종 성능표(train.py)는 그대로 둠.

    python src/improve_b2.py

E0 기준        : B1로 학습한 최종 모델(Ridge, dQ_log_var, log target)
E1 학습 배치 확장 : B1+B3로 학습 → B2 평가 (B2 라벨 미사용)
E2 소량 재보정    : B2 셀 k개의 실제 수명으로 예측 편향(log 오프셋)만 보정 → 나머지 B2 셀로 평가
                  근거 : log 수명–dQ_log_var 기울기는 배치 간 거의 같고(B1 -0.30 / B2 -0.33) B2 절편만 낮음 = 평행 이동
                  보정 셀 선택에 따른 편차를 보려고 무작위 N_REPEAT회 반복
주의 : E2는 '새 배치 셀 일부의 실제 수명을 안다'는 다른 시나리오라 원논문 Target과 직접 비교 불가
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from preprocess import OUT, ROOT
from train import make_model, mape

FEATS = ["dQ_log_var"]
ALPHA = 1.0               # train.py mine/log 1-SE 선택 결과
KS = [1, 3, 5, 10]
N_REPEAT = 500
SEED = 42


def fit(d):
    return make_model(Ridge(alpha=ALPHA), log_target=True).fit(d[FEATS], d["cycle_life"])


def recalibrate(true, pred, k, rng):
    """k개로 log 오프셋 추정 → 나머지에 적용. (보정 전 MAPE, 보정 후 MAPE, 평가 셀 index)"""
    cal = rng.choice(len(true), k, replace=False)
    rest = np.setdiff1d(np.arange(len(true)), cal)
    offset = np.mean(np.log10(true[cal]) - np.log10(pred[cal]))
    return mape(true[rest], pred[rest]), mape(true[rest], pred[rest] * 10 ** offset), rest


def main():
    df = pd.read_csv(OUT / "features.csv")
    df = df[~df["censored"]]
    b1, b2, b3 = (df[df["batch"] == b] for b in ["b1", "b2", "b3"])
    true = b2["cycle_life"].to_numpy()
    slow = (b2["chargetime_2_6"] >= 10.1).to_numpy()   # B2 숨은 그룹 (10.17분 = 단수명)

    rows = []
    p0 = fit(b1).predict(b2[FEATS])
    rows.append({"실험": "E0 기준 (B1 학습)", "B2 MAPE": mape(true, p0), "std": 0.0, "평가 셀": len(true)})
    p1 = fit(pd.concat([b1, b3])).predict(b2[FEATS])
    rows.append({"실험": "E1 B1+B3 학습", "B2 MAPE": mape(true, p1), "std": 0.0, "평가 셀": len(true)})

    rng = np.random.default_rng(SEED)
    for k in KS:
        res = [recalibrate(true, p0, k, rng) for _ in range(N_REPEAT)]
        after = np.array([r[1] for r in res])
        rows.append({"실험": f"E2 재보정 k={k}", "B2 MAPE": after.mean(), "std": after.std(),
                     "평가 셀": len(true) - k, "같은 셀 보정 전": np.mean([r[0] for r in res])})

    # k=5 그룹별 (보정 셀이 대부분 단수명 그룹에서 뽑히는 영향 확인)
    grp = {"slow": [], "fast": []}
    for _ in range(N_REPEAT):
        cal = rng.choice(len(true), 5, replace=False)
        rest = np.setdiff1d(np.arange(len(true)), cal)
        corr = p0 * 10 ** np.mean(np.log10(true[cal]) - np.log10(p0[cal]))
        for name, mask in [("slow", slow[rest]), ("fast", ~slow[rest])]:
            if mask.any():
                grp[name].append(mape(true[rest][mask], corr[rest][mask]))

    out = pd.DataFrame(rows).round(2)
    out.to_csv(ROOT / "results" / "b2_improvement.csv", index=False)
    pd.set_option("display.width", 200)
    print(out.to_string(index=False))
    print(f"\nE2 k=5 그룹별 : 10.17분 그룹 {np.mean(grp['slow']):.1f}% (보정 전 {mape(true[slow], p0[slow]):.1f}%) / "
          f"10.04분 그룹 {np.mean(grp['fast']):.1f}% (보정 전 {mape(true[~slow], p0[~slow]):.1f}%)")


if __name__ == "__main__":
    main()
