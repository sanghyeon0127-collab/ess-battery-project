# Data

원본 `.mat`(MATLAB v7.3 / HDF5)은 용량(약 8.3GB) 때문에 저장소에 포함하지 않음. 아래 명령으로 `data/raw/`에 내려받음.

| 과제 배치 | 파일 | 크기 | 비고 |
|---|---|---|---|
| Batch 1 (Train) | `2017-05-12_batchdata_updated_struct_errorcorrect.mat` | 3.03GB | 원논문 학습 배치 |
| Batch 2 (Test) | `2018-02-20_batchdata_updated_struct_errorcorrect.mat` | 2.02GB | 원출처 분류상 Fig.4용 low-rate 데이터 (원논문 test 배치 아님) |
| Batch 3 (Test, 추가) | `2018-04-12_batchdata_updated_struct_errorcorrect.mat` | 3.24GB | 원논문 2차 테스트 배치 |

```bash
mkdir -p data/raw && cd data/raw
curl -L -o 2017-05-12_batchdata_updated_struct_errorcorrect.mat https://data.matr.io/1/api/v1/file/5c86c0b5fa2ede00015ddf66/download
curl -L -o 2018-02-20_batchdata_updated_struct_errorcorrect.mat https://s3.amazonaws.com/publications.matr.io/1/final_data/2018-02-20_batchdata_updated_struct_errorcorrect.mat
curl -L -o 2018-04-12_batchdata_updated_struct_errorcorrect.mat https://data.matr.io/1/api/v1/file/5c86bd64fa2ede00015ddbb2/download
```

- 출처 : https://data.matr.io/1/projects/5c48dd2bc625d700019f3204 (CC BY 4.0)
- Kaggle 미러 : https://www.kaggle.com/datasets/itshpark/data-driven-prediction-of-battery-cycle
- `python src/preprocess.py` 실행 시 `data/processed/`에 분석용 캐시 생성
