# 멀티모달 LLM 기반 로봇의 Long-horizon Task 분해 및 적응형 재계획 프레임워크

> **(Multimodal LLM-based Long-horizon Task Breakdown and Semantic Similarity Matching with Adaptive Re-planning)**

본 프로젝트는 사용자의 추상적이고 긴 호흡의 자연어 명령(Long-horizon Task)을 로봇이 수행 가능한 원자적 동작(Atomic Actions)으로 정교하게 분해하고, 의미론적 유사도 검사(Semantic Similarity Matching)를 통해 실제 동작 자산과 정렬하는 지능형 에이전트 시스템을 구축하는 것을 목표로 합니다.

---

## 0. 웹 UI 실행 방법

### Requirements

```bash
pip install flask python-dotenv numpy matplotlib pandas
```

Flask 기반 웹 UI는 `app.py`로 실행할 수 있습니다.

```bash
python app.py
```

기본 Ollama 엔드포인트는 `http://localhost:11434/api/chat`이며, 필요하면 다음 환경변수로 바꿀 수 있습니다.

- `OLLAMA_URL`: Ollama chat endpoint
- `OLLAMA_MODEL`: 사용할 모델 이름
- `DEFAULT_SYSTEM_PROMPT`: 서버 기본 시스템 프롬프트

정적 파일은 `static/html`, `static/css`, `static/js`로 분리되어 있습니다.

---

## 1. 개요 (Overview)

### 해결하고자 하는 문제 (Problem Statement)
* **Semantic Gap:** 사용자의 추상적 의도와 로봇의 구체적 실행 코드 간의 논리적 단절.
* **Matching Error:** LLM이 생성한 명령어가 실제 로봇의 Skill Library에 존재하지 않을 때 발생하는 시스템 중단 문제.
* **Rigidity:** 환경 변화나 실행 오류 발생 시 스스로 계획을 수정하지 못하는 경직된 제어 구조.

### 핵심 솔루션 (Key Solutions)
* **Hierarchical Planning:** MLLM을 활용하여 복잡한 과업을 논리적 단계로 분해.
* **Semantic Alignment:** 고정 키워드 매칭 대신 벡터 임베딩 유사도를 활용하여 가용 동작에 유연하게 매핑.
* **Closed-loop Re-planning:** 매칭 실패 또는 환경 변화 시 피드백 루프를 통해 실시간 재계획 수행.

---

## 2. 주요 기능 (Key Features)

* **Long-horizon Task Decomposition:** 추상적 명령(예: "거실 정리해줘")을 하위 단계(Sub-tasks)로 세분화.
* **Vision-Language Grounding:** VLM을 통해 현재 환경 정보를 인식하고, 이를 기반으로 상황에 적합한 계획 수립.
* **Semantic Similarity Matching:** Faiss와 SBERT를 이용해 LLM의 제안 동작을 로봇 스킬 라이브러리와 실시간 매칭.
* **Adaptive Re-planning:** 유사도 임계값 미달 시 에러 피드백을 통해 계획을 수정하는 자가 회복 기능.

---

## 3. 기술 스택 (Tech Stack)

### Models & API
* **MLLM:** Llama-3-8B (Local), LLaVA-v1.6 (Local), GPT-4o (API)
* **Embedding:** Sentence-BERT (SBERT)

### Frameworks & Tools
* **Language:** Python 3.10+
* **Vector DB:** Faiss
* **Orchestration:** LangChain / LangGraph, Pydantic
* **Inference Engine:** vLLM / Ollama

### Infrastructure
* **Hardware:** NVIDIA RTX 3090 x 2
* **Server A:** 로컬 LLM/VLM 추론 및 배포 전담
* **Server B:** 전체 시스템 로직 관리 및 임베딩 검색 엔진

---

## 4. 개발 로드맵 (Roadmap)

| 주차 | 주요 개발 내용 |
|:---:|---|
| **Week 11** | Atomic Action 라이브러리 정의 및 로컬 MLLM(vLLM/Ollama) 추론 환경 구축 |
| **Week 12** | Task 분해를 위한 CoT 프롬프트 설계 및 VLM 기반 시각적 상황 인지 모듈 개발 |
| **Week 13** | Faiss를 활용한 의미론적 유사도 매칭 시스템 구축 및 임계값 기반 유효성 검증 |
| **Week 14** | 에러 피드백 기반의 Closed-loop Re-planning 상태 관리 로직 완성 |
| **Week 15** | 시나리오별 벤치마크 수행, 과업 성공률 및 매칭 정확도 정량 측정 및 최종 평가 |

---

## 5. 평가 지표 및 기준선 (Metrics & Baseline)

* **과업 성공률(Success Rate):** 복합 과업 시나리오 최종 목적 달성률 (목표: 75% 이상)
* **매칭 정확도(Semantic Accuracy):** 생성 명령과 실제 스킬 간의 정렬 정확도 (목표: 90% 이상)
* **재계획 성공률(Re-planning SR):** 예외 상황 발생 시 재계획을 통한 과업 완수율 (목표: 60% 이상)

---

## 6. 참고 논문 (References)

* **Code as Policies:** Language Model Programs for Embodied Control ([arXiv:2209.07753](https://arxiv.org/abs/2209.07753))
* **SayCan:** Do As I Can, Not As I Say: Grounding Language Models in Robotic Affordances ([arXiv:2204.01691](https://arxiv.org/abs/2204.01691))
* **Inner Monologue:** Embodied Reasoning through Planning with Language Models ([arXiv:2207.05608](https://arxiv.org/abs/2207.05608))
