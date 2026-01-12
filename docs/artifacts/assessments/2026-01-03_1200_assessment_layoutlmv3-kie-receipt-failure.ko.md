---
title: "영수증 데이터에 대한 LayoutLMv3 KIE 학습"
author: "AI Agent"
date: "2026-01-03 00:00 (KST)"
type: "assessment"
category: "troubleshooting"
version: "1.0"
ads_version: "1.0"
status: "completed"
tags: ["kie", "layoutlmv3", "receipt", "document-parse", "failure-analysis"]
severity: "blocker"
outcome: "abandoned"
wandb_run: "https://wandb.ai/ocr-team2/ocr-kie/runs/e7l6f9k4"
---

## 1. 요약

Upstage KIE API 엔티티 라벨과 Document Parse API 경계 상자를 병합하여 한국어 영수증에 대한 LayoutLMv3 키 정보 추출(KIE) 학습을 시도했습니다.

| 메트릭 | 값 |
|--------|-------|
| 최종 에폭 | 10 |
| val_F1 | 0.623 |
| val_loss | 1.170 |
| train_loss | 0.441 |
| 결과 | **중단** |

> [!CAUTION]
> **근본 원인:** Document Parse는 영수증이 아닌 구조화된 문서(양식, 보고서)용으로 설계되었습니다. 개별 텍스트 블록 대신 영수증 전체를 포함하는 단일 테이블 경계 상자를 반환합니다.

## 2. 평가

### 2.1 문제 정의

- Upstage KIE API는 엔티티 라벨을 제공하지만 경계 상자는 제공하지 않음
- Upstage Document Parse API는 경계 상자를 제공하지만 영수증을 테이블로 처리함
- 병합된 데이터셋에는 HTML 오염과 과도하게 큰 경계 상자가 포함됨

### 2.2 증거

**샘플 0 (검증 세트):**
```python
Text: "<br><table id='1' style='font-size:18px'><thead>..."
Polygon: x=[0.000-0.965], y=[0.014-0.999]  # 이미지의 96.5% × 98.5%
```

**데이터셋 품질:**
- 검증 샘플의 ~40%가 5개 미만의 텍스트 블록 포함 (예상: 20-50+)
- 텍스트 필드에 HTML 아티팩트 존재 (정제되지 않은 DP 출력)
- 이미지당 1-2개의 거대한 경계 상자 vs 예상된 20-50개의 작은 경계 상자

### 2.3 타임라인

| 날짜 | 이벤트 |
|------|-------|
| 2026-01-02 | 초기 학습: val_F1=0.0 |
| 2026-01-02 | 라벨 불일치 수정 (BIO→단순) |
| 2026-01-03 | 에폭 10: val_F1=0.623 |
| 2026-01-03 | 수동 DP 콘솔 테스트로 문제 확인 |
| 2026-01-03 | **결정: 접근 방식 중단** |

## 3. 권장 사항

### 즉시: 텍스트 인식으로 전환

AI Hub 공공행정문서 데이터셋에 집중:
- 5467개의 최적화된 이미지 준비 완료
- 깨끗한 단어 수준 경계 상자 + 텍스트 라벨
- HTML 오염 없음

### 영수증 특화

1. OCR + LLM 추출 사용 (레이아웃 불필요)
2. Upstage KIE 출력에 텍스트 전용 NER 학습
3. 레이아웃 기반 모델 완전히 생략

### 장기 (레이아웃 KIE 필요 시)

1. AI Hub(한국어 문서 레이아웃)에서 LayoutLMv3 사전 학습
2. 수동으로 라벨링된 영수증 데이터로 미세 조정
3. Document Parse 대신 텍스트 감지 출력(DBNet/CRAFT) 사용

## 4. 수정된 파일

| 파일 | 변경 사항 |
|------|--------|
| `configs/train_kie.yaml` | num_labels=32, 단순 라벨, 워밍업 |
| `runners/train_kie.py` | 트레이너에서 warmup_steps 필터링 |
| `configs/train_kie_baseline_optimized_v2.yaml` | 경로 및 라벨 업데이트 |

## 5. 교훈

1. **파이프라인 구축 전 API 출력 검증** — 5분간의 수동 테스트로 며칠을 절약할 수 있었음
2. **Document Parse ≠ 범용 레이아웃 추출기** — 양식 최적화, 영수증에는 부적합
3. **영수증에는 레이아웃 분석 불필요** — 본질적으로 선형 구조임
