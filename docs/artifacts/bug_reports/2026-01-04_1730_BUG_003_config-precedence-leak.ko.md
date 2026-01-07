---
ads_version: "1.0"
type: "bug_report"
category: "troubleshooting"
status: "completed"
severity: "중요"
version: "1.0"
tags: ['구성', '하이드라', '레거시', '우선순위']
title: "레거시 구성이 아키텍처 구성 요소를 덮어쓰는 문제"
date: "2026-01-04 17:30 (KST)"
resolved_date: "2026-01-04 19:30 (KST)"
branch: "main"
summary: "PARSeq 훈련 구성이 `FPNDecoder` 대신 `PARSeqDecoder`를 잘못 인스턴스화하는 문제. 구성 병합 과정에서 레거시 기본값(`train_v2` → `_base/model` → `dbnet`)이 아키텍처의 특정 구성 요소 정의보다 우선권을 갖기 때문입니다."
---

# 세부 정보

## 증상
PARSeq에 대해 `fast_dev_run` 실행 시:
```text
ValueError: FPNDecoder는 인코더로부터 최소 두 개의 피처 맵이 필요합니다.
```
이 오류는 `parseq.yaml`에서 `parseq_decoder`를 명시했음에도 `FPNDecoder`가 사용되고 있음을 확인합니다.

## 근본 원인
`train_parseq.yaml` 실험 구성은 `defaults: - train_v2` (레거시)와 `- model/architectures/parseq`를 상속받습니다. `_base/model.yaml`에는 기본값으로 `/model/architectures: dbnet`이 포함되어 있습니다.
`OCRModel._prepare_component_configs` 메서드는 다음을 병합했습니다:
1. `top_level_overrides` (cfg에서)를 `direct_overrides` (아키텍처에서) 이후에 적용하여 레거시가 우선권을 얻음
2. 가장 높은 우선순위의 `cfg.component_overrides`에 레거시 dbnet 구성 요소가 포함됨

## 수정 구현 (BUG_003)
`ocr/models/architecture.py` 업데이트:
1. 아키텍처와 충돌하는 레거시 구성 요소를 제거하는 `_filter_architecture_conflicts()` 메서드 추가
2. 병합 순서 재조정: `filtered_top_level` → `direct_overrides` → `filtered_user_overrides`
3. `top_level_overrides`와 `cfg.component_overrides` 모두 필터링 적용

## 검증
```
INFO ocr.models.architecture - BUG_003: 아키텍처 디코더(parseq_decoder)를 위해 레거시 디코더(fpn_decoder) 필터링
INFO ocr.models.architecture - BUG_003: 아키텍처 헤드(parseq_head)를 위해 레거시 헤드(db_head) 필터링
INFO ocr.models.architecture - BUG_003: 아키텍처 손실(cross_entropy)을 위해 레거시 손실(db_loss) 필터링

```
