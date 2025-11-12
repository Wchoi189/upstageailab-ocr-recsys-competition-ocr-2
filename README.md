<!-- Github Decorative Badges -->
<div align="center">

[![CI](https://github.com/AIBootcamp13/upstageailab-ocr-recsys-competition-ocr-2/actions/workflows/ci.yml/badge.svg)](https://github.com/AIBootcamp13/upstageailab-ocr-recsys-competition-ocr-2/actions)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8+-red.svg)](https://pytorch.org)
[![UV](https://img.shields.io/badge/UV-0.8+-purple.svg)](https://github.com/astral-sh/uv)
[![Hydra](https://img.shields.io/badge/Hydra-1.3+-green.svg)](https://hydra.cc)
[![PyTorch Lightning](https://img.shields.io/badge/PyTorch_Lightning-2.1+-orange.svg)](https://lightning.ai)
[![Competition](https://img.shields.io/badge/Competition-Upstage_AI_Lab-blue.svg)](https://upstage.ai)
</div>

# AI 경진대회: 영수증 텍스트 검출

이 프로젝트는 영수증 이미지에서 텍스트 위치를 추출하는 것에 중점을 둡니다. 주어진 영수증 이미지에서 텍스트 요소 주변에 경계 다각형을 정확하게 식별하고 생성할 수 있는 모델을 구축하는 것이 목표입니다.

* **대회 기간:** 2025년 9월 22일 (10:00) - 2025년 10월 16일 (19:00)
* **주요 과제:** 영수증 이미지 내 텍스트 영역 식별 및 윤곽 그리기

## 📋 목차

- [0. 개요](#0-개요)
- [1. 대회 정보](#1-대회-정보)
- [2. 구성 요소](#2-구성-요소)
- [3. 데이터 설명](#3-데이터-설명)
- [4. 모델링](#4-모델링)
- [5. 결과](#5-결과)
- [6. 결론 및 향후 과제](#6-결론-및-향후-과제)


## 👥 팀 소개
<table>
    <tr>
        <td align="center"><img src="https://avatars.githubusercontent.com/u/156163982?v=4" width="180" height="180"/></td>
        <td align="center"><img src="https://github.com/AIBootcamp13/upstageailab-ir-competition-ir-2/blob/main/docs/assets/images/team/hskimh1982.png" width="180" height="180"/></td>
        <td align="center"><img src="https://github.com/Wchoi189/document-classifier/blob/dev-hydra/docs/images/team/AI13_%EC%B5%9C%EC%9A%A9%EB%B9%84.png?raw=true" width="180" height="180"/>
        <td align="center"><img src="https://github.com/AIBootcamp13/upstageailab-ir-competition-ir-2/blob/main/docs/assets/images/team/YeonkyungKang.png" width="180" height="180"/></td>
        <td align="center"><img src="https://github.com/AIBootcamp13/upstageailab-ir-competition-ir-2/blob/main/docs/assets/images/team/jungjaehoon.jpg" width="180" height="180"/></td>
    </tr>
    <tr>
        <td align="center"><a href="https://github.com/SuWuKIM">AI13_이상원</a></td>
        <td align="center"><a href="https://github.com/YOUR_GITHUB">AI13_김효석</a></td>
        <td align="center"><a href="https://github.com/Wchoi189">AI13_최용비</a></td>
        <td align="center"><a href="https://github.com/YeonkyungKang">AI13_강연경</a></td>
        <td align="center"><a href="https://github.com/YOUR_GITHUB">AI13_정재훈</a></td>
    </tr>
    <tr>
        <td align="center">팀장, 일정관리, 성능 최적화</td>
        <td align="center">EDA, 데이터셋 증강</td>
        <td align="center">베이스라인, CI</td>
        <td align="center">연구/실험 설계 및 추론 분석</td>
        <td align="center">아키텍처 리뷰 및 설계</td>
    </tr>
 </table>

## 0. 개요

### 개발 환경
- **Python:** 3.10+
- **패키지 관리자:** UV 0.8+
- **딥러닝:** PyTorch 2.8+, PyTorch Lightning 2.1+
- **구성 관리:** Hydra 1.3+

### 요구사항
- Python 3.10 이상
- UV 패키지 관리자
- CUDA 호환 GPU (훈련 시 권장)

## 1. 대회 정보

### 개요
영수증 텍스트 검출에 중점을 둔 AI 경진대회입니다. 참가자들은 경계 다각형을 사용하여 영수증 이미지에서 텍스트 영역을 정확하게 검출하고 위치를 파악할 수 있는 모델을 개발해야 합니다.

### 일정
- **시작일:** 2025년 9월 22일 (10:00)
- **최종 제출 마감일:** 2025년 10월 16일 (19:00)

## 2. 구성 요소

### 디렉토리 구조

```
├── augmentation-patterns.yaml
├── configs/
│   ├── predict.yaml
│   ├── test.yaml
│   ├── train.yaml
│   └── preset/
│       ├── base.yaml
│       ├── example.yaml
│       ├── datasets/
│       │   └── db.yaml
│       ├── lightning_modules/
│       │   └── base.yaml
│       └── models/
│           ├── model_example.yaml
│           ├── decoder/
│           ├── encoder/
│           ├── head/
│           └── loss/
├── data/
│   ├── datasets/
│   │   ├── sample_submission.csv
│   │   └── images/
│   │       ├── test/
│   │       └── ...
│   └── jsons/
├── docs/
│   ├── ai_handbook/
│   │   ├── index.md
│   │   ├── 02_protocols/
│   │   ├── 04_experiments/
│   │   └── ...
│   ├── pipeline/
│   │   └── data_contracts.md
│   ├── bug_reports/
│   ├── CHANGELOG.md
│   ├── QUICK_FIXES.md
│   ├── api-reference.md
│   ├── architecture-overview.md
│   ├── process-management-guide.md
│   ├── component-diagrams.md
│   ├── workflow-diagram.md
│   ├── maintenance/
│   │   └── project-state.md
│   └── development/
│       ├── coding-standards.md
│       ├── naming-conventions.md
│       └── testing-guide.md
├── ocr/
│   ├── datasets/
│   ├── lightning_modules/
│   ├── metrics/
│   ├── models/
│   └── utils/
├── ablation_study/
├── outputs/
├── runners/
│   ├── predict.py
│   ├── test.py
│   └── train.py
├── scripts/
│   ├── agent_tools/
│   └── process_monitor.py
├── ui/
│   ├── command_builder.py
│   ├── evaluation_viewer.py
│   ├── inference_ui.py
│   ├── resource_monitor.py
│   ├── components/
│   ├── utils/
│   └── README.md
└── tests/
```

### UI 도구

프로젝트에는 명령어 구축과 결과 분석을 위한 Streamlit 기반 UI 도구가 포함되어 있습니다.

#### Command Builder (`ui/command_builder.py`)
훈련, 테스트, 예측 명령어를 직관적인 UI로 구축하고 실행할 수 있는 도구입니다.

**주요 기능:**
- 모델 아키텍처 선택 (인코더, 디코더, 헤드, 손실 함수)
- 학습 파라미터 조정 (학습률, 배치 크기, 에폭 수)
- 실험 설정 (W&B 통합, 체크포인트 재개)
- 실시간 명령어 검증 및 미리보기
- 원클릭 명령어 실행 및 진행 상황 모니터링

**사용법:**
```bash
# 명령어 구축 UI 실행
python run_ui.py command_builder

# 또는 직접 실행
uv run streamlit run ui/command_builder.py
```

#### Evaluation Viewer (`ui/evaluation_viewer.py`)
평가 결과를 시각화하고 분석하는 도구입니다.

### UI 데모 스크린샷

<div align="center">

<table style="border-collapse: collapse; border: none;">
<tr>
<td style="border: none; padding: 20px; text-align: center; vertical-align: top;">
<strong>Command Builder</strong><br><br>
<img src="docs/assets/images/demo/command-builder-predict-command-generate.png" alt="Command Builder Interface" style="width: 800px; height: 500px; object-fit: contain; border: 1px solid #ddd; border-radius: 8px;"><br><br>
<small>훈련 및 예측 명령어를 직관적인 UI로 구축하고 실행할 수 있는 인터페이스</small>
</td>
</tr>
<tr>
<td style="border: none; padding: 20px; text-align: center; vertical-align: top;">
<strong>Real-time OCR Inference</strong><br><br>
<img src="docs/assets/images/demo/real-time-ocr-inference-select-img.png.jpg" alt="Real-time OCR Inference Interface" style="width: 800px; height: 500px; object-fit: contain; border: 1px solid #ddd; border-radius: 8px;"><br><br>
<small>실시간으로 영수증 이미지에서 텍스트를 검출하고 결과를 확인할 수 있는 인터페이스</small>
</td>
</tr>
<tr>
<td style="border: none; padding: 20px; text-align: center; vertical-align: top;">
<strong>Evaluation Viewer</strong><br><br>
<img src="docs/assets/images/demo/ocr-eval-results-viewer-gallery.png" alt="Evaluation Viewer Interface" style="width: 800px; height: 500px; object-fit: contain; border: 1px solid #ddd; border-radius: 8px;"><br><br>
<small>모델 평가 결과를 시각화하고 상세하게 분석할 수 있는 갤러리 뷰 인터페이스</small>
</td>
</tr>
</table>

</div>

### 주요 구성 파일

- `train.yaml`, `test.yaml`, `predict.yaml`: 러너 실행 설정 (훈련, 테스트, 예측용 기본 구성)
- `configs/preset/example.yaml`: 각 모듈의 구성 파일 지정 및 기본 실험 설정
- `configs/preset/datasets/db.yaml`: DBNet 데이터셋, Transform, 데이터 관련 설정
- `configs/preset/datasets/preprocessing.yaml`: 전처리 파이프라인 설정
- `configs/preset/lightning_modules/base.yaml`: PyTorch Lightning 모듈 실행 설정
- `configs/preset/models/model_example.yaml`: 각 모델 모듈과 Optimizer의 구성 파일 지정
- `configs/preset/models/encoder/`: 다양한 인코더 설정 (MobileNetV3, ResNet 등)
- `configs/preset/models/decoder/`: 다양한 디코더 설정 (PAN, DBNet++ 등)
- `configs/preset/models/head/`: 모델 헤드 구성
- `configs/preset/models/loss/`: 손실 함수 설정

## 3. 데이터 설명

### 데이터셋 개요

데이터는 이미지 폴더와 주석을 위한 해당 JSON 파일로 구성됩니다. 데이터셋은 영수증 이미지와 텍스트 영역 주석을 포함하는 train/validation/test 분할로 구성되어 있습니다.

### 디렉토리 구조

```
.
├── images/
│   ├── train/
│   │   └── ...jpg
│   ├── val/
│   │   └── ...jpg
│   └── test/
│       └── ...jpg
└── jsons/
     ├── train.json
     ├── val.json
     └── test.json
```

### JSON 주석 형식

JSON 파일은 이미지 파일명을 텍스트 경계 상자의 좌표에 매핑합니다.

* **IMAGE_FILENAME**: 각 이미지 레코드의 키
* **words**: 이미지에 대해 감지된 모든 텍스트 인스턴스를 포함하는 객체
* **nnnn**: 각 단어 인스턴스의 고유한 4자리 인덱스 (0001부터 시작)
* **points**: 텍스트 주변의 다각형을 정의하는 [X, Y] 좌표 쌍의 배열. 원점 (0,0)은 이미지의 왼쪽 상단 모서리. 평가에 유효한 다각형이 되려면 최소 4개의 점이 필요

### EDA

- _EDA 과정과 단계별 결론을 설명하세요_

### 데이터 처리

- 이미지는 JPG 형식으로 저장
- 주석은 다각형 좌표가 포함된 JSON 형식으로 제공
- 텍스트 영역은 정확한 경계 다각형으로 주석 처리
- 데이터셋은 train, validation, test 분할을 포함

### 데이터 전처리 (Pre-processing)

이 프로젝트는 훈련 성능을 크게 향상시키는 오프라인 전처리 시스템을 사용합니다.

#### 전처리가 필요한 이유

DBNet 모델은 확률 맵(probability map)과 임계값 맵(threshold map)을 필요로 합니다. 이전에는 이러한 맵을 훈련 중 실시간으로 생성했으나, 다음과 같은 문제가 있었습니다:

- 계산 비용이 높은 pyclipper 연산과 거리 계산
- 에포크마다 동일한 맵을 반복 계산
- 효과적이지 못한 캐싱 메커니즘

오프라인 전처리를 통해 **5-8배 빠른 검증 속도**를 달성했습니다.

#### 전처리 실행 방법

전체 데이터셋을 전처리하려면 프로젝트 루트에서 다음 명령을 실행하세요:

```bash
uv run python scripts/preprocess_maps.py
```

샘플 수를 제한하여 테스트하려면:

```bash
uv run python scripts/preprocess_maps.py data.train_num_samples=100 data.val_num_samples=20
```

전처리 스크립트는 다음을 생성합니다:
- `data/datasets/images/train_maps/`: 훈련 데이터의 전처리된 맵
- `data/datasets/images_val_canonical_maps/`: 검증 데이터의 전처리된 맵

각 이미지에 대해 압축된 `.npz` 파일이 생성되며, 확률 맵과 임계값 맵이 포함됩니다.

#### 자동 폴백 (Fallback)

전처리된 맵이 없어도 훈련은 정상적으로 작동합니다. 시스템이 자동으로 실시간 맵 생성으로 전환되지만, 속도가 느려집니다.

더 자세한 내용은 [데이터 전처리 데이터 컨트랙트](docs/preprocessing-data-contracts.md)와 [파이프라인 데이터 컨트랙트](docs/pipeline/data_contracts.md)를 참조하세요.

#### 데이터 증강 및 전처리 (Data Augmentation and Preprocessing)
- **이미지 향상**: Doctr 라이브러리를 사용하여 텍스트 검출, 크롭핑, 그리고 이미지 향상을 수행하여 검출 성능을 개선했습니다. CamScanner 스타일의 전처리를 적용하여 이미지 품질을 최적화했습니다.
- **회전 보정**: 검증 및 테스트 데이터셋에서 발견된 회전 불일치를 표준 방향으로 수정했습니다. 이는 데이터 누출이 아니며, 일관된 전처리 단계로 간주됩니다 (훈련 데이터에 영향을 미치지 않음).

## 4. 모델링

### 모델 설명

이 프로젝트는 모듈형 OCR 시스템으로, PyTorch Lightning과 Hydra를 기반으로 구축되었습니다. 컴포넌트 기반 아키텍처를 통해 인코더, 디코더, 헤드, 손실 함수를 플러그 앤 플레이 방식으로 교체할 수 있습니다.

#### 아키텍처 모듈화 (Architecture Modularization)
- **레지스트리 기반 시스템**: 컴포넌트들은 `architectures/registry.py`에 등록되며, Hydra 설정을 통해 동적으로 조립됩니다. 추상 인터페이스 (`BaseEncoder`, `BaseDecoder`, `BaseHead`, `BaseLoss`)를 상속하여 일관성을 유지합니다.
- **팩토리 패턴**: `ModelFactory`가 등록된 컴포넌트를 사용하여 완전한 모델을 생성합니다. 이는 빠른 실험을 위한 플러그 앤 플레이 교체를 지원합니다.
- **최종 모델**: DBNet 아키텍처를 기반으로, MobileNetV3 인코더와 PAN 디코더를 사용했습니다. 이는 실시간 텍스트 검출을 위한 미분 가능한 이진화를 활용합니다.

#### DBNet: 미분 가능한 이진화를 통한 실시간 장면 텍스트 검출

![DBNet](docs/assets/images/banner/flow-chart-of-the-dbnet.png)

### 베이스라인 성능

V100 GPU에서 10 에포크 훈련 후, 베이스라인 모델은 공개 테스트 세트에서 다음과 같은 성능을 달성했습니다:

* **훈련 시간:** 약 22분
* **H-Mean:** 0.8818
* **정밀도:** 0.9651
* **재현율:** 0.8194

### 평가 지표

이 대회는 텍스트 검출 결과 평가를 위해 **CLEval**을 사용합니다.

#### CLEval: 텍스트 검출 및 인식 작업을 위한 문자 수준 평가

![CLEval](https://github.com/clovaai/CLEval/raw/master/resources/screenshots/explanation.gif)

### 모델링 과정

#### 훈련
```bash
uv run python runners/train.py preset=example
```

#### 테스트
```bash
# 사용 예시
uv run python runners/test.py preset=example checkpoint_path=\"outputs/ocr_training/checkpoints/epoch-9-step-1030.ckpt\"
```

#### 예측
```bash
# 사용 예시
uv run python runners/predict.py preset=example checkpoint_path=\"outputs/ocr_training/checkpoints/epoch-8-step-1845.ckpt\"
```

#### 사용 예시
```bash
# 1. Run unit tests
uv run pytest tests/ -v

# 2. Train model (adjust epochs as needed)
uv run python runners/train.py preset=example trainer.max_epochs=10 dataset_base_path="/path/to/data/datasets/"

# 3. Generate predictions
uv run python runners/predict.py preset=example checkpoint_path="outputs/ocr_training/checkpoints/best.ckpt" dataset_base_path="/path/to/data/datasets/"

# 4. Convert to submission format
uv run python ocr/utils/convert_submission.py --json_path outputs/ocr_training/submissions/latest.json --output_path submission.csv


```

### 모델 개선 사항
- **아키텍처 변경**: 베이스라인 DBNet을 모듈형 컴포넌트 시스템으로 변환하여 유연성을 높였습니다. MobileNetV3 인코더는 경량화로 효율성을, PAN 디코더는 정확한 텍스트 영역 검출을 제공합니다.
- **하이퍼파라미터 튜닝**: 학습률, 배치 크기, 에폭을 조정하여 최적화했습니다 (예: `trainer.max_epochs=10`, `model.optimizer.lr=0.0005`).
- **데이터 증강**: Doctr 기반 전처리와 CamScanner 스타일 향상을 적용하여 이미지 품질을 개선하고 검출 성능을 향상시켰습니다.


## 5. 결과

### 최종 성능

- **공개 리더보드 (Public Leaderboard)**: H-Mean: 0.9751, 정밀도: 0.9739, 재현율: 0.9773 (3/4위).
- **비공개 리더보드 (Private Leaderboard)**: H-Mean: 0.9787, 정밀도: 0.9771, 재현율: 0.9809.
- **베이스라인 대비 개선**: 베이스라인 H-Mean 0.8818에서 크게 향상되었으며, 모듈화와 전처리로 인한 안정성 증가.

### 제출 과정

#### 제출 파일 생성

예측 스크립트는 JSON 파일을 생성합니다. 이 파일은 제출하기 전에 제공된 유틸리티 스크립트를 사용하여 필요한 CSV 형식으로 변환해야 합니다.

```bash
# 사용 예시
uv run python ocr/utils/convert_submission.py --json_path outputs/ocr_training/submissions/your_submission.json --output_path submission.csv
```

#### CSV 형식

제출 파일은 `filename`과 `polygons` 두 열이 있는 CSV여야 합니다.

* **filename**: 테스트 세트의 이미지 파일명
* **polygons**: 해당 이미지에서 예측된 모든 텍스트 영역의 좌표를 포함하는 단일 문자열
  * 하나의 다각형에 대한 좌표는 공백으로 구분 (예: `X1 Y1 X2 Y2 X3 Y3 X4 Y4`)
  * 같은 이미지의 다른 다각형들은 파이프 문자(`|`)로 구분

#### 출력 파일 구조

```
└─── outputs
    └── {exp_name}
        ├── .hydra
        │   ├── overrides.yaml
        │   ├── config.yaml
        │   └── hydra.yaml
        ├── checkpoints
        │   └── epoch={epoch}-step={step}.ckpt
        ├── logs
        │   └── {exp_name}
        │       └── {exp_version}
        │           └── events.out.tfevents.{timestamp}.{hostname}.{pid}.v2
        └── submissions
            └── {timestamp}.json
```

### 평가 기준

대회 리더보드는 공개 및 비공개 순위로 나뉩니다. 대회 기간 중에는 공개 세트에 대한 점수가 표시됩니다. 최종 우승자는 대회 종료 후 공개되는 비공개 테스트 세트에서의 모델 성능으로 결정됩니다. 테스트 데이터는 공개 및 비공개 세트 간에 동등하게(50/50) 분할됩니다.

### 실험 결과 분석

- 다양한 실험에서 PAN 디코더가 DBNet의 정확도를 높였으나, 복잡한 배경에서 재현율이 낮은 경우가 있었습니다. 실패 케이스는 주로 저해상도 이미지였습니다.
- 모델 한계: 고해상도 이미지에서 계산 비용 증가; 향후 개선으로 EfficientNet 인코더 고려.

## 6. 결론 및 향후 과제

### 주요 성과

- 베이스라인 점수를 크게 개선했습니다 (H-Mean 0.8818 → 0.9751+).
- 중앙 집중식 구성 관리로 설정 관리를 용이하게 했습니다.
- 실시간 OCR 추론을 위한 Streamlit UI, 복잡한 훈련 명령어 생성을 위한 Command Builder, 예측 파일 보기 및 낮은 예측 결과 분석을 위한 Evaluation View를 포함한 데이터 분석 도구를 구현했습니다.
- 성능 병목 현상 분석을 위한 성능 프로파일링을 포함한 고급 성능 기능(.npz 사전 컴파일 이미지, 변환, 캐싱)을 구현했습니다.
- 실험 추적을 위한 Wandb / Hydra 통합.
- 10개 이상의 모놀리식 구조를 모듈형 컴포넌트 시스템으로 리팩토링했습니다.
- 페어 프로그래밍 협업을 개선하기 위한 AI 문서화 및 개발 프로토콜.
- 버그 보고서 생성 프로토콜 구현.

#### 팀원별 기여도
- **최웅비**: 오류 분석 도구(Streamlit Inference UI, Evaluation Viewer) 및 기타 UI 개발로 개발을 용이하게 했습니다. 고급 성능 기능 구현, AI 핸드북 개발을 통한 개발 효율성 개선 연구, 데이터 계약 및 전처리 모듈 개발.

<!-- **NEW:** Unified OCR Development Studio 구현 (Preprocessing, Inference 모드 통합, YAML 기반 설정 시스템, 80% 완료). -->

### 향후 개선 방향

- **모델 성능 개선**: Nvidia DALI 구현으로 훈련/추론 파이프라인 속도를 5~8배 개선하고, EfficientNet 인코더와 같은 더 강력한 아키텍처 탐색.
- **데이터 품질 향상**: 프로덕션급 전처리 및 합성 데이터 생성 파이프라인 구현으로 데이터 다양성 및 품질 향상.
- **시스템 최적화**: 분산 훈련 지원, 메모리 최적화, 그리고 실시간 추론을 위한 모델 양자화 구현.

## 설치 및 설정

### 🚨 환경 설정 (중요)

이 프로젝트는 **UV** 패키지 매니저를 사용합니다. 다른 패키지 매니저(pip, conda, poetry)를 사용하지 마세요.

```bash
# 자동 환경 설정 (권장)
./scripts/setup/00_setup-environment.sh


### VS Code 설정

프로젝트를 VS Code에서 열면 자동으로 다음 설정이 적용됩니다:
- Python 인터프리터: `./.venv/bin/python`
- 터미널: 자동으로 가상환경 활성화
- 모든 Python 명령어는 `uv run` 접두사 사용

### 모든 명령어는 `uv run` 사용

```bash
# ❌ 잘못된 사용
python runners/train.py
pytest tests/

# ✅ 올바른 사용
uv run python runners/train.py
uv run pytest tests/
```

### 로컬 테스트

```bash
# 모든 테스트 실행
uv run pytest tests/

```markdown
# 특정 테스트 파일 실행
uv run pytest tests/test_metrics.py

# 커버리지와 함께 실행 (선택사항)
uv run pytest tests/ --cov=ocr
```

## 라이선스

이 프로젝트는 [MIT 라이선스](LICENSE) 하에 배포됩니다.

## 참고 자료

- [프로세스 관리 가이드](docs/process-management-guide.md) - 훈련 프로세스 관리 및 고아 프로세스 방지
- [DBNet](https://github.com/MhLiao/DB)
- [Hydra](https://hydra.cc/docs/intro/)
- [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/latest/)
- [CLEval](https://github.com/clovaai/CLEval)
- [UV 패키지 관리자](https://github.com/astral-sh/uv)

## 참고 논문:
- CLEval: Character-Level Evaluation for Text Detection and Recognition Tasks
  https://arxiv.org/pdf/2006.06244.pdf


## 연락처

프로젝트에 대한 질문이나 제안사항이 있으시면 다음을 통해 연락해 주세요:

- **이슈 트래커**: [GitHub Issues](https://github.com/AIBootcamp13/upstageailab-ocr-recsys-competition-ocr-2/issues)
<!-- - **이메일**: [팀 대표 이메일 주소]
- **디스코드**: [팀 디스코드 채널] -->

---
