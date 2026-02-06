---
ads_version: '1.0'
type: guide
experiment_id: 20251220_154834_zero_prediction_images_debug
title: 세피아 테스트 빠른 시작
created: '2025-12-21T02:23:00+09:00'
tags:
- sepia
- quickstart
status: complete
updated: '2025-12-21T02:23:00+09:00'
commands: []
prerequisites: []
---
# 세피아 테스트 빠른 시작

## 디렉터리 구조

```
20251220_154834_zero_prediction_images_debug/
├── scripts/
│   ├── sepia_enhancement.py          # 5가지 세피아 방법 (클래식, 적응형, 웜, CLAHE, 선형 대비)
│   ├── compare_sepia_methods.py     # 비교 프레임워크 (아티팩트 검증 포함)
│   ├── sepia_perspective_pipeline.py # 전체 파이프라인 (세피아 + 원근법)
│   └── vlm_validate_sepia.sh        # VLM 품질 평가
├── artifacts/
│   └── reference_images/            # 테스트 샘플
│       ├── 000712.jpg
│       ├── 000732.jpg
│       └── 000732_REMBG.jpg
└── outputs/
    ├── sepia_tests/                 # 개별 방법 결과
    ├── sepia_comparison/            # 비교 그리드
    ├── sepia_pipeline/              # 파이프라인 결과
    └── sepia_vlm_reports/           # VLM 평가 보고서
```

## 빠른 테스트

**개별 세피아 테스트 실행**:
```bash
cd scripts/
python sepia_enhancement.py \
  --input ../artifacts/reference_images/drp.en_ko.in_house.selectstar_000732_REMBG.jpg \
  --method all \
  --output ../outputs/sepia_tests/
```

**예상 출력**:
- `*_sepia_classic.jpg`
- `*_sepia_adaptive.jpg`
- `*_sepia_warm.jpg`
- `*_sepia_clahe.jpg`
- `*_sepia_linear_contrast.jpg`
- `*_metrics.json`

## 방법 비교

```bash
python compare_sepia_methods.py \
  --input ../artifacts/reference_images/drp.en_ko.in_house.selectstar_000732_REMBG.jpg \
  --output ../outputs/sepia_comparison/ \
  --save-metrics
```

**비교 항목**:
1. 원본 이미지
2. 그레이스케일
3. 그레이-월드 정규화
4. 세피아 클래식
5. 세피아 적응형
6. 세피아 웜
7. 세피아 CLAHE (적응형 대비)
8. 세피아 선형 대비 (글로벌 게인)

## 전체 파이프라인 테스트

```bash
python sepia_perspective_pipeline.py \
  --input ../artifacts/reference_images/drp.en_ko.in_house.selectstar_000732.jpg \
  --sepia-method clahe \
  --output ../outputs/sepia_pipeline/
```

## VLM 검증

```bash
export DASHSCOPE_API_KEY="your_key"
./vlm_validate_sepia.sh ../outputs/sepia_comparison/
```

## 문제 해결

**background_normalization.py 누락**:
```bash
# 프로젝트 루트에서 복사
cp ../../../ocr/datasets/background_normalization.py scripts/
```

**vlm_validate_sepia.sh 권한 거부**:
```bash
chmod +x scripts/vlm_validate_sepia.sh
```

## 다음 단계

1. 개별 테스트 실행 → 모든 방법 작동 확인
2. 비교 실행 → 최적의 세피아 방법 식별
3. 파이프라인 실행 → 통합 테스트
4. VLM 검증 → 시각적 품질 확인
5. OCR 추론 → 정확도 개선 측정
