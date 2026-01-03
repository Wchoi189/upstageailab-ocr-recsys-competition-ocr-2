<div align="center">

[![CI](https://github.com/Wchoi189/upstageailab-ocr-recsys-competition-ocr-2/actions/workflows/ci.yml/badge.svg)](https://github.com/Wchoi189/upstageailab-ocr-recsys-competition-ocr-2/actions)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-FFD21E.svg)](https://huggingface.co/wchoi189/receipt-text-detection_kr-pan_resnet18)

# OCR 텍스트 인식 및 레이아웃 분석 시스템

**레이아웃 분석을 통한 정확한 정보 추출을 위한 AI 최적화 텍스트 인식 시스템**

[English](README.md) • [한국어](README.ko.md)

[기능](#기능) • [프로젝트 컴퍼스](#프로젝트-컴퍼스-ai-네비게이션) • [문서화](#문서화)

</div>

---

## 개요

이 프로젝트는 Upstage AI 부트캠프 OCR 경연에서 시작되었으며, 고급 레이아웃 분석을 포함한 엔드투엔드 텍스트 인식 시스템 구축을 위한 개인 프로젝트로 발전했습니다. 주요 아키텍처 업그레이드 전 마지막 준비와 안전 점검을 진행 중입니다.

**리포지토리:**
- **개인 (지속):** [Wchoi189/upstageailab-ocr-recsys-competition-ocr-2](https://github.com/Wchoi189/upstageailab-ocr-recsys-competition-ocr-2)
- **원본 (부트캠프):** [AIBootcamp13/upstageailab-ocr-recsys-competition-ocr-2](https://github.com/AIBootcamp13/upstageailab-ocr-recsys-competition-ocr-2)

---

## 프로젝트 컴퍼스: AI 네비게이션

**프로젝트 컴퍼스**는 이 프로젝트의 중추 시스템으로, AI 에이전트가 코드베이스를 탐색하고 이해하며 git 히스토리를 파헤치지 않고도 유지할 수 있도록 설계되었습니다.

### MCP 통합
**모델 컨텍스트 프로토콜(MCP)** 을 통해 프로젝트 컴퍼스 내부 상태를 노출시켜 AI 에이전트가 직접 프로젝트 상태와 상호작용할 수 있도록 했습니다.

**사용 가능한 도구:**
- `env_check`: `uv` 환경, 파이썬 버전, CUDA 상태를 락 파일과 비교합니다.
- `session_init --objective [목표]`: 특정 목표에 집중하기 위해 현재 세션 컨텍스트를 원자적으로 업데이트합니다.
- `reconcile`: 실험 메타데이터를 깊이 스캔하고 실제 디스크 내용(`manifest.json`)과 동기화합니다.
- `ocr_convert`: **(신규)** 데이터셋을 LMDB로 변환하는 멀티쓰레드 ETL 파이프라인을 시작합니다.
- `ocr_inspect`: **(신규)** LMDB 데이터셋의 무결성을 검증합니다.

<details>
<summary>📂 프로젝트 컴퍼스 상태 탐색 (클릭하여 확장)</summary>

**프로젝트 컴퍼스**는 프로젝트의 실시간 상태를 유지합니다. AI 에이전트는 이 파일을 읽어 컨텍스트, 장애물, 목표를 이해합니다.

| 카테고리 | 파일 | 설명 |
|----------|------|-------------|
| **🧠 활성 컨텍스트** | [`current_session.yml`](project_compass/active_context/current_session.yml) | 현재 고수준 목표와 락 상태. |
| | [`blockers.yml`](project_compass/active_context/blockers.yml) | 활성 장애물 및 의존성 문제 목록. |
| **🗺️ 로드맵** | [`02_recognition.yml`](project_compass/roadmap/02_recognition.yml) | 텍스트 인식 단계 계획. |
| **🤖 에이전트** | [`AGENTS.yaml`](project_compass/AGENTS.yaml) | 사용 가능한 도구 및 MCP 명령어 레지스트리. |
| **💾 데이터** | [`dataset_registry.yml`](project_compass/environments/dataset_registry.yml) | 데이터셋 경로 및 형식에 대한 단일 소스. |
| **📜 역사** | [`session_handover.md`](project_compass/session_handover.md) | 다음 에이전트를 위한 "랜딩 페이지". |

</details>

---

## OCR ETL 파이프라인 및 데이터 처리

대규모 데이터셋을 효율적으로 처리하기 위해 `ocr-etl-pipeline`이라는 독립형 고성능 데이터 처리 패키지를 개발했습니다.

### 주요 기능
- **제로 클러터**: 수백만 개의 원시 이미지 파일을 단일 **LMDB (Lightning Memory-Mapped Database)** 파일로 변환합니다.
- **재개 가능**: JSON 상태 파일을 사용해 진행 상황을 추적하여 데이터 손실 없이 작업을 일시 중지 및 재개할 수 있습니다.
- **멀티프로세싱**: RTX 3090 워크스테이션에 최적화되어 모든 사용 가능한 CPU 코어를 활용해 이미지 디코딩 및 크롭을 수행합니다.

**성능**:
- AI Hub의 616,366개 샘플을 약 1분 만에 처리.
- 파일 시스템 오버헤드를 대폭 감소시킴 (1개 파일 vs 600,000개 파일).

---

## 연구 인사이트 및 전환

### 영수증 처리를 위해 KIE + 문서 파서 API를 포기한 이유

초기 전략은 키 정보 추출(KIE) 모델과 문서 파서 API를 결합하는 것이었습니다. 그러나 광범위한 테스트 결과 다음과 같은 중대한 불일치가 발견되었습니다:

1.  **근본적 불일치**: 영수증은 주로 선형 또는 반구조화된 텍스트 스트림입니다. 문서 파서 API는 문서를 고도로 구조화된 표/양식으로 처리합니다.
2.  **HTML 오염**: 영수증에 대한 API 출력에는 시각적 레이아웃을 나타내지 않는 과도한 HTML 테이블 태그가 포함되어 임베딩을 오염시켰습니다.
3.  **레이아웃LM 비효율성**: 레이아웃LM 모델은 복잡한 2D 공간 관계(예: 양식)에 최적화되어 있습니다. 영수증 OCR의 경우, **텍스트 인식(PARSeq/CRNN)** 모델이 훨씬 더 효과적이고 견고했습니다.

**결과**: AI Hub 데이터셋에 기반한 전용 텍스트 인식 파이프라인 구축으로 전환했으며, 이 특정 도메인에서는 KIE 접근법을 포기했습니다.

---

## AWS 배치 프로세서: 클라우드 네이티브 데이터 엔지니어링

로컬 리소스 제약과 API 속도 제한을 극복하기 위해 서버리스 배치 처리를 위한 독립 모듈 `aws-batch-processor`를 구축했습니다.

**문제**: 문서 파서 API 무료 티어는 엄격한 속도 제한을 적용하며, 5,000개 이상의 문서를 로컬에서 동기식으로 처리하면 주요 워크플로가 멈춥니다.
**해결**: 서버리스 배치 아키텍처를 사용해 **AWS Fargate**로 처리를 오프로드하여 로컬 머신을 온라인 상태 유지 없이 밤새 처리할 수 있도록 했습니다.

- **아키텍처**: [다이어그램 및 구현 세부 정보 보기](aws-batch-processor/README.md)
- **데이터 카탈로그**: [`aws-batch-processor/data/export/data_catalog.yaml`](aws-batch-processor/data/export/data_catalog.yaml)
- **스크립트 카탈로그**: [`aws-batch-processor/script_catalog.yaml`](aws-batch-processor/script_catalog.yaml)

### 주요 기술
- **AWS Fargate**: 배치 작업을 위한 서버리스 컴퓨팅.
- **S3**: 중간 및 최종 결과를 위한 내구성 있는 저장소.
- **Parquet**: 효율적인 주석 쿼리를 위한 열 기반 저장소.

---

## 품질 보증 및 버그 추적

주요 사고에 대해 상세하고 구조화된 버그 보고서를 생성하여 엄격한 품질 기준을 유지합니다. 이 아티팩트는 지속적인 학습 자료로 활용됩니다.

[📂 **버그 보고서 컬렉션 보기**](docs/artifacts/bug_reports/)

**예시 아티팩트:**
- **중요 오류**: 파이프라인 충돌 근본 원인 문서화.
- **데이터 무결성**: "영수증 파싱 시 HTML 오염"과 같은 문제 추적.
- **해결**: 모든 보고서에는 검증된 수정 전략이 포함되어 있어 회귀를 방지합니다.

---

## 기능

- **원근 교정**: Rembg의 이진 마스크 출력을 사용한 고정밀 엣지 감지.
- **원근 왜곡 보정**: 목표 영역의 가시성을 최적화하기 위한 기하학적 변환.
- **배경 정규화**: 고품질 이미지에서 조명 변동 및 색조 변화로 인한 감지 실패 해결.
- **이미지 분석**: 자동화된 이미지 평가 및 기술적 결함 보고를 위한 특수 VLM 도구.

---

## 실험 추적기: 구조화된 AI 기반 연구

**해결된 문제**: 빠른 AI 기반 실험은 관리하기 어려운 대량의 아티팩트, 스크립트, 문서를 생성합니다.

**해결**: `experiment_manager/` - 인간이 읽기 쉽고 AI 소비에 최적화된 실험 아티팩트 구조화 시스템.

### 표준화된 기술 보고서 및 문서 예시

**기준선 분석**
- [기준선 메트릭 요약](experiment_manager/experiments/20251217_024343_image_enhancements_implementation/artifacts/20251218_1415_report_baseline-metrics-summary.md) - 품질 개선 시 성능 벤치마크 설정

**사건 해결**
- [데이터 손실 사건 보고서](experiment_manager/experiments/20251217_024343_image_enhancements_implementation/artifacts/20251220_0130_incident_report_perspective_correction_data_loss.md) - 데이터 손실 사건 분석 및 해결 전략

---

## 프로젝트 진행 현황

<div align="center">

| 단계 | 상태 | 진행률 |
|-------|--------|----------|
| **단계 1-4: 핵심 개발** | 완료 | 100% |
| **단계 5: 사전 업그레이드 준비** | 진행 중 | 90% |
| **단계 6: 아키텍처 업그레이드** | 계획 | 0% |

**전체: 85% 완료**

</div>

**현재 집중 분야**: 새로 생성된 AI Hub LMDB 데이터셋을 사용한 텍스트 인식 모델(PARSeq/CRNN) 학습.

---

## 기술 스택 및 환경

**엄격한 정책**: 이 프로젝트는 모든 파이썬 패키지 관리 및 실행에 `uv`를 사용합니다.

| 카테고리 | 기술 |
|----------|-------------|
| **ML/DL** | PyTorch, PyTorch Lightning, Hydra |
| **백엔드** | FastAPI, ONNX Runtime |
| **도구** | **UV (필수)**, npm, W&B, Playwright, Vitest |
| **QMS** | AgentQMS (아티팩트, 표준, 컴플라이언스) |

---

## 모델 동물원장

| 모델 이름 | 아키텍처 | H-평균 | 허깅페이스 |
|------------|--------------|--------|--------------|
| **영수증 감지 KR** | DBNet + PAN (ResNet18) | 95.37% | [🤗 모델 카드](https://huggingface.co/wchoi189/receipt-text-detection_kr-pan_resnet18) |

---

## 문서화

**AI 대상 리소스 (AgentQMS 표준)**
- [시스템 아키텍처](AgentQMS/standards/tier1-sst/system-architecture.yaml)
- [API 계약](AgentQMS/standards/tier2-framework/api-contracts.yaml)
- [AgentQMS 워크플로우](AgentQMS/knowledge/agent/system.md)

**참조**
- [파일 배치 규칙](AgentQMS/standards/tier1-sst/file-placement-rules.yaml)
- [변경 로그](CHANGELOG.md)

---

## 기여

기여를 환영합니다! 지침은 [CONTRIBUTING.md](CONTRIBUTING.md)를 참조하세요.

---

## 라이선스

MIT 라이선스 - 자세한 내용은 [LICENSE](LICENSE)를 참조하세요.

<div align="center">

[⬆ 맨 위로](#ocr-텍스트-인식--레이아웃-분석-시스템)

</div>
