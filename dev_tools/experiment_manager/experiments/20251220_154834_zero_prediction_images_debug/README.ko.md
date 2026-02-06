---
ads_version: "1.0"
type: experiment_manifest
experiment_id: "20251220_154834_zero_prediction_images_debug"
status: "completed"
created: "2025-12-20T15:48:34.936608+09:00"
updated: "2025-12-21T03:18:00+09:00"
tags: ["sepia", "image-enhancement", "ocr-preprocessing"]
---

# 실험: 제로 예측 OCR을 위한 세피아 향상

## 개요

이 실험은 OCR 전처리를 위해 그레이스케일 및 그레이월드 정규화의 대안으로 세피아 색상 변환을 조사합니다.

**문제**: 특정 저대비 또는 오래된 문서 이미지(예: 000712, 000732)는 표준 정규화 방법을 사용할 때 OCR 예측이 0개 또는 제한적으로 생성됩니다.
**가설**: 세피아 향상(특히 CLAHE를 통한 적응형 대비)은 문자-배경 구분을 개선하여 더 신뢰할 수 있는 OCR 감지를 제공합니다.

## 결과 요약 🏆

- **최적 방법**: `sepia_clahe` (붉은색 틴트 + 적응형 대비)
- **에지 개선**: **+164.0%** (기준선 0% 대비)
- **대비 향상**: **+8.2** (기준선 0 대비)
- **통찰력**: 올바른 붉은색 틴트 매핑이 중요합니다. 초기 녹색 틴트는 에지 선명도가 23% 낮았습니다.

## 문서 색인

이 실험은 EDS v1.0을 따릅니다. 모든 상세 결과 및 가이드는 `.metadata/` 디렉토리에 있습니다:

| 아티팩트 | 목적 |
| :--- | :--- |
| [최종 보고서](.metadata/reports/20251221_0316_report_final-sepia-vs-clahe-performance-analysis.ko.md) | 상세 성능 분석 및 메트릭. |
| [빠른 시작 가이드](.metadata/guides/sepia_quick_start.ko.md) | 향상 및 비교 스크립트 실행 명령어. |
| [테스트 계획](.metadata/plans/sepia_testing_plan.ko.md) | 원본 워크플로우 및 성공 기준. |
| [상태 로그](.metadata/00-status/2025-12-21_sepia_testing_progress.ko.md) | 최종 진행 상황 업데이트 및 타임라인. |

## 실험 구조

```
.
├── scripts/              # 처리 및 벤치마킹 스크립트
├── artifacts/            # 참조 이미지 및 테스트 샘플
├── outputs/              # 생성된 결과 및 메트릭
└── .metadata/            # EDS 호환 문서
```

---

*Antigravity AI 검증 | ETK v1.0.0 | EDS v1.0*
