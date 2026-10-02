"""Batch 1 학습 → Batch 2(필수) / Batch 3(추가) 평가, 포맷에 맞춘 성능표 생성.

    python src/train.py [feature_set] [log|raw]     # 기본 : mine log

분할 (누수 방지)
- Valid (Batch 1 hold-out) : 충전 정책 단위로 분리. 같은 정책 셀이 train/valid에 나뉘면 정책 정보가 새어 들어감
- Train (Batch 1 CV)       : hold-out을 뺀 나머지로 GroupKFold(정책 단위). 결측대체·스케일링은 Pipeline 안에서 fold별 fit
- Test                     : Batch 1 전체로 재학습한 최종 모델을 Batch 2 / 3에 한 번만 적용

모델 선택 (1-SE 규칙) : CV 최저 모델의 'CV MAPE + 1 표준오차' 안에 드는 모델 중 가장 단순한 것 (MODELS 순서 = 단순한 순)
                       학습 셀이 29개뿐이라 CV 차이가 표준오차보다 작으면 우연과 구분할 수 없음
"""
import sys

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from preprocess import OUT, ROOT

# ============================ CONFIG (직접 결정) ============================
SEED = 42
VALID_SIZE = 0.2          # Batch 1 중 hold-out 비율 (정책 단위)
CV_FOLDS = 5
TARGET_MAPE = 9.1         # 원논문 Test 오차 (%)

FEATURE_SETS = {
    # 원논문 3개 모델의 피처 구성 (비교 기준)
    "variance": ["dQ_log_var"],
    "discharge": ["dQ_log_var", "dQ_log_min", "dQ_log_skew", "dQ_log_kurt", "QD_2", "QD_max_minus_2"],
    "full": ["dQ_log_var", "dQ_log_min", "fade_slope_2_100", "fade_icpt_2_100", "QD_2",
             "chargetime_2_6", "Tavg_mean", "IR_min", "IR_100_minus_2"],
    # 본인 전략 : 세 배치 모두 수명과 같은 방향의 관계를 유지하는 ΔQ 1개
    "mine": ["dQ_log_var"],
}

MODELS = {   # 후보 모델 (단순한 순서) : (estimator, 하이퍼파라미터 grid)
    "Ridge": (Ridge(), {"alpha": np.logspace(-3, 2, 11)}),
    "ElasticNet": (ElasticNet(max_iter=50_000), {"alpha": np.logspace(-4, 0, 9), "l1_ratio": [0.1, 0.5, 0.9]}),
    "SVR": (SVR(), {"C": [0.1, 1, 10], "epsilon": [0.01, 0.05, 0.1]}),
    "RandomForest": (RandomForestRegressor(random_state=SEED), {"n_estimators": [300], "max_depth": [2, 3, None]}),
    "GBR": (GradientBoostingRegressor(random_state=SEED), {"n_estimators": [100, 300], "max_depth": [1, 2], "learning_rate": [0.05, 0.1]}),
}
# ===========================================================================


def mape(y, p):
    return mean_absolute_percentage_error(y, p) * 100


def make_model(est, log_target):
    pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), est)
    if log_target:
        return TransformedTargetRegressor(pipe, func=np.log10, inverse_func=lambda z: 10 ** z)
    return pipe


def grid_prefix(est, grid, log_target):
    # make_pipeline 스텝 이름 = 소문자 클래스명, TransformedTargetRegressor 안이면 regressor__ 추가
    prefix = ("regressor__" if log_target else "") + type(est).__name__.lower() + "__"
    return {prefix + k: v for k, v in grid.items()}


def split_b1(b1):
    tr, va = next(GroupShuffleSplit(1, test_size=VALID_SIZE, random_state=SEED).split(b1, groups=b1["policy"]))
    return b1.iloc[tr], b1.iloc[va]


def select_1se(comp):
    """comp는 MODELS 순서(단순한 순). CV 최저 + 1SE 이하인 첫 모델"""
    best = comp.loc[comp["train_cv"].idxmin()]
    return comp[comp["train_cv"] <= best["train_cv"] + best["cv_se"]].iloc[0]


def run(feature_set, log_target):
    feats = FEATURE_SETS[feature_set]
    df = pd.read_csv(OUT / "features.csv")
    df = df[~df["censored"]]                       # EOL 미도달 셀 = 라벨이 실제 수명 아님
    b1, b2, b3 = (df[df["batch"] == b] for b in ["b1", "b2", "b3"])
    train, valid = split_b1(b1)
    assert not set(train["policy"]) & set(valid["policy"])
    X, y = (lambda d: d[feats]), (lambda d: d["cycle_life"])

    rows, preds = [], []
    for name, (est, grid) in MODELS.items():
        gs = GridSearchCV(make_model(est, log_target), grid_prefix(est, grid, log_target), cv=GroupKFold(CV_FOLDS),
                          scoring="neg_mean_absolute_percentage_error")
        gs.fit(X(train), y(train), groups=train["policy"])
        cv_std = gs.cv_results_["std_test_score"][gs.best_index_] * 100

        final = make_model(est, log_target).set_params(**gs.best_params_).fit(X(b1), y(b1))   # Batch 1 전체로 재학습
        r = {"model": name, "best_params": {k.split("__")[-1]: (round(float(v), 4) if isinstance(v, float) else v)
                                            for k, v in gs.best_params_.items()},
             "train_cv": -gs.best_score_ * 100, "cv_se": cv_std / np.sqrt(CV_FOLDS),
             "valid": mape(y(valid), gs.predict(X(valid)))}
        for b, d in [("b1_valid", valid), ("b2", b2), ("b3", b3)]:
            p = gs.predict(X(d)) if b == "b1_valid" else final.predict(X(d))
            if b != "b1_valid":
                r[f"test_{b}"] = mape(y(d), p)
            preds.append(pd.DataFrame({"model": name, "cell_id": d["cell_id"], "batch": b, "policy": d["policy"],
                                       "true": y(d), "pred": p}))
        rows.append(r)

    comp = pd.DataFrame(rows)
    preds = pd.concat(preds, ignore_index=True)
    preds["ape"] = (preds["pred"] - preds["true"]).abs() / preds["true"] * 100
    return feats, train, valid, comp, preds


def report(best):
    """포맷 표. Gap은 '오차 증가' 방향 → (+)면 악화"""
    rows = [
        ("Train (Batch 1 CV)", best.train_cv, ""),
        ("Valid (Batch 1 Hold-out)", best.valid, ""),
        ("Test (Batch 2)", best.test_b2, ""),
        ("Gap (Train-Valid)", best.valid - best.train_cv, "(+) : 과적합 의심"),
        ("Gap (Valid-Test)", best.test_b2 - best.valid, "(+) : 배치간 일반화 저하 의심"),
        ("Gap (Target-Test)", best.test_b2 - TARGET_MAPE, f"Target : 원논문 {TARGET_MAPE}%"),
        ("Test (Batch 3)", best.test_b3, ""),
        ("Gap (Batch2-Batch3)", best.test_b3 - best.test_b2, "Test 성능 간 비교"),
        ("Gap (Target-Test, Batch 3)", best.test_b3 - TARGET_MAPE, "Batch 3 기준, 원논문 성능 비교"),
    ]
    return pd.DataFrame(rows, columns=["구분", "MAPE (%)", "비고"]).round({"MAPE (%)": 2})


def main():
    feature_set = sys.argv[1] if len(sys.argv) > 1 else "mine"
    log_target = (sys.argv[2] if len(sys.argv) > 2 else "log") == "log"
    tag = f"{feature_set}_{'log' if log_target else 'raw'}"
    feats, train, valid, comp, preds = run(feature_set, log_target)
    best = select_1se(comp)
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    report(best).to_csv(out / f"model_performance_{tag}.csv", index=False)
    comp.to_csv(out / f"model_comparison_{tag}.csv", index=False)
    preds[preds["model"] == best.model].to_csv(out / f"predictions_{tag}.csv", index=False)

    pd.set_option("display.width", 200)
    print(f"[{tag}] features={feats}")
    print(f"split : train {len(train)} cells / valid {len(valid)} cells (valid policies : {sorted(valid['policy'].unique())})")
    print(comp.round(2).to_string(index=False))
    print(f"\nselected (1-SE) : {best.model}\n{report(best).to_string(index=False)}")


if __name__ == "__main__":
    main()
