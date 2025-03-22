# ML-table-embedding

테이블 데이터의 조인 가치를 평가하기 위한 실험 프로젝트입니다. BERT와 DistilBERT를 활용하여 두 테이블 간의 IVS(Integration Value Score)를 예측하고 분석합니다.

## 프로젝트 소개

이 프로젝트는 다음과 같은 문제를 해결합니다:

- 두 개의 테이블(.csv)이 있을 때, 이들을 병합했을 때 얻을 수 있는 가치를 미리 예측
- IVS(Integration Value Score)를 통한 병합 가치 평가
- BERT/DistilBERT 기반의 테이블 임베딩을 활용한 데이터 분석

## 주요 기능

### 1. 테이블 임베딩 (model_BERT, model_distilBERT)

- BERT와 DistilBERT를 활용한 테이블 구조 및 의미적 특성 추출
- 테이블 간의 의미적 유사도 분석
- 각 모델별 커스텀 구현 및 최적화

### 2. IVS(Integration Value Score) 계산

- 두 테이블 간의 병합 가치를 평가하는 종합 지표
- 테이블의 구조적, 의미적 특성을 고려한 0~1 사이의 점수
- 컬럼 이름, 데이터 타입, 값의 분포 등 다양한 특성 반영

### 3. 모델 학습 및 예측

- `train_BERT/`, `train_distilBERT/`: IVS 예측을 위한 모델 학습
- `predict_BERT/`, `predict_distilBERT/`: 새로운 테이블 쌍의 IVS 예측
- `find.py`: Top-K 유사 테이블 검색

### 4. HPC 클러스터 지원

- OpenHPC 슈퍼컴퓨터 클러스터에서의 분산 학습 지원
- `sbatch.sh`를 통한 대규모 데이터셋 처리
- 효율적인 학습 및 예측 파이프라인 구축

## 디렉토리 구조

```
ML-table-embedding/
├── model_BERT/          # BERT 기반 모델 구현
├── model_distilBERT/    # DistilBERT 기반 모델 구현
├── train_BERT/         # BERT 모델 학습 코드
├── train_distilBERT/   # DistilBERT 모델 학습 코드
├── predict_BERT/       # BERT 기반 IVS 예측
├── predict_distilBERT/ # DistilBERT 기반 IVS 예측
├── find.py            # Top-K 테이블 검색
└── sbatch.sh          # HPC 클러스터 배치 스크립트
```

## 사용 방법

1. 모델 학습

```bash
# BERT 모델 학습
python3 train_BERT/train.py

# DistilBERT 모델 학습
python3 train_distilBERT/train.py
```

2. IVS 예측

```bash
# BERT 기반 IVS 예측
python3 predict_BERT/predict.py

# DistilBERT 기반 IVS 예측
python3 predict_distilBERT/predict.py
```

3. 유사 테이블 검색

```bash
# k개의 가장 유사한 테이블 검색
python3 find.py --k 15
```

## 의존성

- Python 3.7 이상
- PyTorch
- Transformers
- pandas
- numpy

## 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다.
