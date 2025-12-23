<div align="center">

[![CI](https://github.com/Wchoi189/upstageailab-ocr-recsys-competition-ocr-2/actions/workflows/ci.yml/badge.svg)](https://github.com/Wchoi189/upstageailab-ocr-recsys-competition-ocr-2/actions)
[![파이썬](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8+-red.svg)](https://pytorch.org)
[![라이센스](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![허깅페이스 모델](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-FFD21E.svg)](https://huggingface.co/wchoi189/receipt-text-detection_kr-pan_resnet18)

# OCR 텍스트 인식 및 레이아웃 분석 시스템

**정확한 정보 추출을 위한 레이아웃 분석 기능을 갖춘 AI에 최적화된 텍스트 인식 시스템**

[영어](README.md) • [한국어](README.ko.md)

[특징](#features) • [진행상황](#project-progress)] • [문서](#documentation)

</div>

---

## 소개

이 프로젝트는 Upstage AI Bootcamp OCR 대회에서 시작되었으며 고급 레이아웃 분석을 갖춘 엔드투엔드 텍스트 인식 시스템 구축에 초점을 맞춘 개인 연속으로 발전했습니다. 현재 주요 아키텍처 업그레이드를 앞두고 최종 준비 및 안전 점검이 진행 중입니다.

**저장소:**
- **개인(계속):** [Wchoi189/upstageailab-ocr-recsys-competition-ocr-2](https://github.com/Wchoi189/upstageailab-ocr-recsys-competition-ocr-2)
- **원본(Bootcamp):** [AIBootcamp13/upstageailab-ocr-recsys-competition-ocr-2](https://github.com/AIBootcamp13/upstageailab-ocr-recsys-competition-ocr-2)

---

## 기능

- **원근 교정**: Rembg의 바이너리 마스크 출력을 사용한 높은 신뢰성의 가장자리 감지.
- **원근 왜곡**: 대상 영역의 가시성을 최적화하기 위한 기하학적 변환입니다.
- **배경 정규화**: 고품질 이미지에서 조명 변화 및 색상 경향성으로 인한 감지 오류를 해결합니다.
- **이미지 분석**: 자동화된 이미지 평가 및 기술 결함 보고를 위한 전문 VLM 도구입니다.

---
## OCR 추론 콘솔

OCR 추론 콘솔은 OCR 웹 서비스에 대한 개념 증명 프런트엔드입니다. 문서 미리보기 및 구조화된 출력 분석을 위한 간소화된 인터페이스를 제공합니다.

<div align="center">
  <a href="docs/assets/images/demo/my-app.webp">
    <img src="docs/assets/images/demo/my-app.webp" alt="OCR Inference Console" width="800px" />
  </a>
  <p><em>OCR 추론 콘솔: 문서 미리보기, 레이아웃 분석 및 구조화된 JSON 출력을 갖춘 3패널 레이아웃입니다. (클릭하시면 확대됩니다)</em></p>
</div>

### UX 속성
사용자 인터페이스 디자인은 **Upstage Document OCR Console**에서 영감을 받았습니다. 문서 미리보기 및 구조화된 출력 기능을 갖춘 3패널 콘솔을 포함한 레이아웃 패턴은 Upstage 제품군에서 확립한 상호 작용 모델을 따릅니다.

이 저장소의 모든 코드와 구현은 Upstage OCR RecSys 경쟁 기준을 기반으로 합니다. 주요 기여에는 구성 현대화, 성능 개선, 개발 워크플로 강화가 포함됩니다.

원본: https://console.upstage.ai/playground/document-ocr

---
## 실험 추적기: 조직화된 AI 기반 연구

**문제 해결**: 신속한 AI 기반 실험은 관리 가능한 상태를 유지하기 위해 체계적인 구성이 필요한 대량의 아티팩트, 스크립트 및 문서를 생성하는 경우가 많습니다. 실험이 매일 반복되고 디버깅을 위해서는 신뢰할 수 있는 문서에 즉시 액세스해야 하는 경우 기존 프로젝트 구조가 실패합니다.

**해결책**: `experiment-tracker/` - 사람의 가독성과 AI 소비 모두에 최적화된 실험 아티팩트를 구성하기 위한 구조화된 시스템입니다. 공통 워크플로에 대한 표준화된 프로토콜과 아티팩트의 출력 형식을 제공합니다.

### 표준화된 기술 보고서 및 문서의 예

**기준 분석**
- [기준 지표 요약](experiment-tracker/experiments/20251217_024343_image_enhancements_implementation/artifacts/20251218_1415_report_baseline-metrics-summary.md) - 미묘한 품질 개선을 비교할 때 성능 벤치마크를 설정하는 포괄적인 기준 지표

**사고 해결**
- [데이터 손실 사고 보고서](experiment-tracker/experiments/20251217_024343_image_enhancements_implementation/artifacts/20251220_0130_incident_report_perspective_correction_data_loss.md) - 중요한 데이터 손실 사고 분석 및 해결 전략

**비교 분석**
- [백그라운드 정규화 비교](experiment-tracker/experiments/20251217_024343_image_enhancements_implementation/.metadata/reports/20251218_1458_report_background-normalization-comparison.md) - 정량적 결과와 백그라운드 정규화 전략 비교

### 시각적 결과 및 데모

<div align="center">| 장착된 코너 | 수정된 출력 |
| :---: | :---: |
| [<img src="docs/assets/images/demo/original-with-fitted-corners.webp" width="700px" />](docs/assets/images/demo/original-with-fitted-corners.webp) | [<img src="experiment-tracker/experiments/20251217_024343_image_enhancements_implementation/outputs/full_pipeline_correct/drp.en_ko.in_house.selectstar_000712_step2_corrected.jpg" width="250px" />](experiment-tracker/experiments/20251217_024343_image_enhancements_implementation/outputs/full_pipeline_correct/drp.en_ko.in_house.selectstar_000712_step2_corrected.jpg) |
| *모서리 감지 및 기하학적 피팅* | *최종 원근 보정 출력* |

*(확대하려면 이미지를 클릭하세요)*

</div>

### 주요 이점

- **AI 최적화**: 효율적인 AI 소비를 위해 설계된 문서 구조입니다.
- **표준화된 프로토콜**: 수동 프롬프트를 줄이고 고품질 결과를 생성합니다.
- **추적성**: 모든 실험 결과에 대한 전체 재현 경로입니다.
- **확장 가능한 조직**: 컨텍스트 혼란을 방지하기 위해 격리된 실험 아티팩트입니다.

---
## 낮은 예측 해상도

<div align="center">

| 이전: 지속적으로 낮은 예측 | 내부 프로세스 | 이후: 탐지 성공 |
| :---: | :---: | :---: |
| [<img src="docs/assets/images/demo/inference-persistent-empties-before.webp" width="250px" />](docs/assets/images/demo/inference-persistent-empties-before.webp) | [<img src="docs/assets/images/demo/inference-persistent-empties-after.webp" width="250px" />](docs/assets/images/demo/inference-persistent-empties-after.webp) | [<img src="docs/assets/images/demo/inference-persistent-empties-after2.webp" width="250px" />](docs/assets/images/demo/inference-persistent-empties-after2.webp) |
| *빈 패치* | *필터 적용* | *정규화된 기하학* |

*(확대하려면 이미지를 클릭하세요)*

</div>

---
## 프로젝트 진행

<div align="center">

| 단계 | 상태 | 진행 |
|-------|---------|----------|
| **1-4단계: 핵심 개발** | 완료 | 100% |
| **5단계: 업그레이드 전 준비** | 진행 중 | 80% |
| **6단계: 아키텍처 업그레이드** | 예정 | 0% |

**전체: 80% 완료**

</div>

**현재 초점:** 최종 안전 점검, 시스템 검증 및 주요 아키텍처 개선 준비.

---

## 기술 스택

| 카테고리 | 기술 |
|----------|-------------|
| **ML/DL** | PyTorch, PyTorch 라이트닝, Hydra |
| **백엔드** | FastAPI, ONNX 런타임 |
| **프런트엔드** | React 19, Next.js 16, 차크라 UI, Streamlit |
| **도구** | UV(Python), npm, W&B, 극작가, Vitest |

---

## 모델 동물원

| 모델명 | 건축 | H-평균 | 포옹하는 얼굴 |
|------------|---------------|---------|-------------|
| **영수증감지KR** | DBNet + PAN(ResNet18) | 95.37% | [🤗 모델카드](https://huggingface.co/wchoi189/receipt-text-detection_kr-pan_resnet18) |

---

## 문서

**AI 대응 리소스(.ai-instructions)**
- [시스템 아키텍처](.ai-instructions/tier1-sst/system-architecture.yaml)
- [API 계약](.ai-instructions/tier2-framework/api-contracts.yaml)
- [AgentQMS 작업 흐름](AgentQMS/knowledge/agent/system.md)

**참고**
- [파일 배치 규칙](.ai-instructions/tier1-sst/file-placement-rules.yaml)
- [변경 내역](CHANGELOG.md)

---

## 프로젝트 구조```
├── AgentQMS/          # AI documentation and quality management
├── apps/              # Frontend & backend applications
├── configs/           # Hydra configuration (89 YAML files)
├── docs/              # AI-optimized documentation & artifacts
├── ocr/               # Core OCR Python package
├── runners/           # Training/testing/prediction scripts
├── scripts/           # Utility scripts
├── tests/             # Unit & integration tests
```
세부 구조: [.ai-instructions/tier1-sst/file-placement-rules.yaml](.ai-instructions/tier1-sst/file-placement-rules.yaml)

---

## 기여

기여를 환영합니다! 지침은 [CONTRIBUTING.md](CONTRIBUTING.md)를 참조하세요.

---

## 라이센스

MIT 라이선스 - 자세한 내용은 [LICENSE](LICENSE)를 참조하세요.

---

<div align="center">

[⬆ 맨 위로 돌아가기](#ocr-text-recognition--layout-analysis-system)

</div>