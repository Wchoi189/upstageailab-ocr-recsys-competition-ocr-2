<div align="center">

[![CI](https://github.com/Wchoi189/upstageailab-ocr-recsys-competition-ocr-2/actions/workflows/ci.yml/badge.svg)](https://github.com/Wchoi189/upstageailab-ocr-recsys-competition-ocr-2/actions)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-FFD21E.svg)](https://huggingface.co/wchoi189/receipt-text-detection_kr-pan_resnet18)

# OCR 텍스트 인식 및 레이아웃 분석 시스템

**정확한 정보 추출을 위한 레이아웃 분석 기능을 갖춘 AI 최적화 텍스트 인식 시스템**

[English](README.md) • [한국어](README.ko.md)

[기능](#기능) • [진행 상황](#프로젝트-진행-상황) • [문서](#문서)

</div>

---

## 소개

이 프로젝트는 Upstage AI 부트캠프 OCR 경진대회에서 시작되어 고급 레이아웃 분석을 통한 엔드투엔드 텍스트 인식 시스템 구축에 중점을 둔 개인 연구로 발전했습니다. 현재 주요 아키텍처 업그레이드 전 최종 준비 및 안전성 검사를 진행 중입니다.

**저장소:**
- **개인 (연구 지속):** [Wchoi189/upstageailab-ocr-recsys-competition-ocr-2](https://github.com/Wchoi189/upstageailab-ocr-recsys-competition-ocr-2)
- **원본 (부트캠프):** [AIBootcamp13/upstageailab-ocr-recsys-competition-ocr-2](https://github.com/AIBootcamp13/upstageailab-ocr-recsys-competition-ocr-2)

---

## 기능

- **원근 보정**: Rembg의 바이너리 마스크 출력을 사용한 고신뢰도 엣지 검출
- **원근 변환**: 대상 영역의 가시성을 최적화하는 기하학적 변환
- **배경 정규화**: 고품질 이미지에서 조명 변화와 색상 편향으로 인한 검출 실패 해결
- **이미지 분석**: 자동화된 이미지 평가 및 기술적 결함 보고를 위한 전문 VLM 도구

---
## OCR 추론 콘솔

OCR 추론 콘솔은 OCR 웹 서비스를 위한 개념 증명 프론트엔드입니다. 문서 미리보기와 구조화된 출력 분석을 위한 간소화된 인터페이스를 제공합니다.

<div align="center">
  <a href="docs/assets/images/demo/my-app.webp">
    <img src="docs/assets/images/demo/my-app.webp" alt="OCR 추론 콘솔" width="800px" />
  </a>
  <p><em>OCR 추론 콘솔: 문서 미리보기, 레이아웃 분석, 구조화된 JSON 출력을 특징으로 하는 3패널 레이아웃 (클릭하여 확대)</em></p>
</div>

### UX 출처
사용자 인터페이스 디자인은 **Upstage Document OCR Console**에서 영감을 받았습니다. 문서 미리보기와 구조화된 출력을 포함한 3패널 콘솔의 레이아웃 패턴은 Upstage 제품군에서 확립된 상호작용 모델을 따릅니다.

이 저장소의 모든 코드와 구현은 Upstage# OCR & RecSys Competition - OCR Track 기준선을 바탕으로 합니다. 주요 기여사항으로는 구성 현대화, 성능 개선, 개발 워크플로우 향상이 있습니다.

원본: https://console.upstage.ai/playground/document-ocr

---
## 실험 추적기: 체계적인 AI 기반 연구

**해결된 문제**: 빠른 AI 기반 실험은 종종 대량의 아티팩트, 스크립트, 문서를 생성하며, 이를 관리 가능한 상태로 유지하려면 체계적인 조직화가 필요합니다. 실험이 매일 반복되고 디버깅에 신뢰할 수 있는 문서에 대한 즉각적인 접근이 필요할 때 기존 프로젝트 구조는 실패합니다.

**해결책**: `experiment-tracker/` - 인간의 가독성과 AI
**해결책**: `experiment-tracker/` - 인간의 가독성과 AI 소비 모두에 최적화된 실험 아티팩트 조직화를 위한 구조화된 시스템입니다. 일반적인 워크플로우를 위한 표준화된 프로토콜과 아티팩트 출력 형식을 제공합니다.

### 표준화된 기술 보고서 및 문서 예시

**기준선 분석**
- [기준선 메트릭 요약](experiment-tracker/experiments/20251217_024343_image_enhancements_implementation/artifacts/20251218_1415_report_baseline-metrics-summary.md) - 품질의 미묘한 개선을 비교할 때 성능 벤치마크를 설정하는 포괄적인 기준선 메트릭

**사고 해결**
- [데이터 손실 사고 보고서](experiment-tracker/experiments/20251217_024343_image_enhancements_implementation/artifacts/20251220_0130_incident_report_perspective_correction_data_loss.md) - 중요한 데이터 손실 사고 분석 및 해결 전략

**비교 분석**
- [배경 정규화 비교](experiment-tracker/experiments/20251217_024343_image_enhancements_implementation/.metadata/reports/20251218_1458_report_background-normalization-comparison.md) - 정량적 결과를 포함한 배경 정규화 전략 비교

### 시각적 결과 및 데모

<div align="center">

| 맞춤 모서리 | 보정된 출력 |
| :---: | :---: |
| [<img src="docs/assets/images/demo/original-with-fitted-corners.webp" width="700px" />](docs/assets/images/demo/original-with-fitted-corners.webp) | [<img src="experiment-tracker/experiments/20251217_024343_image_enhancements_implementation/outputs/full_pipeline_correct/drp.en_ko.in_house.selectstar_000712_step2_corrected.jpg" width="250px" />](experiment-tracker/experiments/20251217_024343_image_enhancements_implementation/outputs/full_pipeline_correct/drp.en_ko.in_house.selectstar_000712_step2_corrected.jpg) |
| *모서리 검출 및 기하학적 맞춤* | *최종 원근 보정 출력* |

*(이미지를 클릭하여 확대)*

</div>

### 주요 장점

- **AI 최적화**: AI 소비에 효율적으로 설계된 문서 구조
- **표준화된 프로토콜**: 수동 프롬프팅을 줄이고 고품질 결과 생성
- **추적 가능성**: 모든 실험 결과에 대한 완전한 재현 경로
- **확장 가능한 조직화**: 컨텍스트 혼란을 방지하기 위한 격리된 실험 아티팩트

---
## 낮은 예측 해상도

<div align="center">

| 이전: 지속적인 낮은 예측 | 내부 프로세스 | 이후: 성공적인 검출 |
| :---: | :---: | :---: |
| [<img src="docs/assets/images/demo/inference-persistent-empties-before.webp" width="250px" />](docs/assets/images/demo/inference-persistent-empties-before.webp) | [<img src="docs/assets/images/demo/inference-persistent-empties-after.webp" width="250px" />](docs/assets/images/demo/inference-persistent-empties-after.webp) | [<img src="docs/assets/images/demo/inference-persistent-empties-after2.webp" width="250px" />](docs/assets/images/demo/inference-persistent-empties-after2.webp) |
| *빈 패치* | *필터 적용* | *정규화된 기하학* |

*(이미지를 클릭하여 확대)*

</div>

---
## 프로젝트 진행 상황

<div align="center">

| 단계 | 상태 | 진행률 |
|-------|--------|----------|
| **1-4단계: 핵심 개발** | 완료 | 100% |
| **5단계: 업그레이드 전 준비** | 진행 중 | 80% |
| **6단계: 아키텍처 업그레이드** | 계획됨 | 0% |

**전체: 80% 완료**

</div>

**현재 초점:** 최종 안전성 검사, 시스템 검증, 주요 아키텍처 개선을 위한 준비

---

## 기술 스택

| 카테고리 | 기술 |
|----------|-------------|
| **ML/DL** | PyTorch, PyTorch Lightning, Hydra |
| **백엔드** | FastAPI, ONNX Runtime |
| **프론트엔드** | React 19, Next.js 16, Chakra UI, Streamlit |
| **도구** | UV (Python), npm, W&B, Playwright, Vitest |

---

## 모델 저장소

| 모델명 | 아키텍처 | H-Mean | Hugging Face |
|------------|--------------|--------|--------------|
| **Receipt Detection KR** | DBNet + PAN (ResNet18) | 95.37% | [🤗 모델 카드](https://huggingface.co/wchoi189/receipt-text-detection_kr-pan_resnet18) |

---

## 문서

**AI 대상 리소스 (.ai-instructions)**
- [시스템 아키텍처](.ai-instructions/tier1-sst/system-architecture.yaml)
- [API 계약](.ai-instructions/tier2-framework/api-contracts.yaml)
- [AgentQMS 워크플로우](AgentQMS/knowledge/agent/system.md)

**참조**
- [파일 배치 규칙](.ai-instructions/tier1-sst/file-placement-rules.yaml)
- [변경 로그](CHANGELOG.md)

---

## 프로젝트 구조

```
├── AgentQMS/          # AI 문서 및 품질 관리
├── apps/              # 프론트엔드 및 백엔드 애플리케이션
├── configs/           # Hydra 구성 (89개 YAML 파일)
├── docs/              # AI 최적화 문서 및 아티팩트
├── ocr/               # 핵심 OCR Python 패키지
├── runners/           # 훈련/테스트/예측 스크립트
├── scripts/           # 유틸리티 스크립트
├── tests/             # 단위 및 통합 테스트
```

상세 구조: [.ai-instructions/tier1-sst/file-placement-rules.yaml](.ai-instructions/tier1-sst/file-placement-rules.yaml)

---

## 기여하기

기여를 환영합니다! 가이드라인은 [CONTRIBUTING.md](CONTRIBUTING.md)를 참조하세요.

---

## 라이선스

MIT 라이선스 - 자세한 내용은 [LICENSE](LICENSE)를 참조하세요.

---

<div align="center">

[⬆ 맨 위로](#ocr-텍스트-인식-및-레이아웃-분석-시스템)

</div>
