# 학부연구생_로드맵

# 🧠 학부 연구생 로드맵

> 서울대 뇌인지과학과 인턴십 | NeuroMamba 프로젝트
> 

---

## 🎯 최종 목표

### 핵심 목표

> 뇌 데이터 + AI 분석 연구실에서 실질적인 연구 경험과 인사이트 획득
> 

### 배우고 싶은 것들

| 카테고리 | 세부 내용 |
| --- | --- |
| 🔬 **연구 인사이트** | 뇌 데이터 분석 분야의 연구 트렌드와 방향성 |
| 🛠️ **AI 도구 활용** | 실제 연구에서 사용하는 AI 도구와 프레임워크 |
| 💻 **코딩 & 분석** | 연구용 코드 작성법, 데이터 분석 파이프라인 |
| 🔧 **문제 해결** | 성능 향상을 위한 접근 방식과 디버깅 스킬 |
| 🖥️ **서버 운용** | 랩서버에서 실제 모델 학습 경험 |
| 🤖 **AI 협업** | AI를 활용한 효율적인 연구 방법론 |

---

## 📚 학습 영역

### 1️⃣ 논문 읽기 & 정리

| 우선순위 | 논문 | 핵심 내용 | 상태 |
| --- | --- | --- | --- |
| ⭐⭐⭐ | **NeuroMamba** | Mamba 기반 fMRI Foundation Model | ✅ 정리 완료 |
| ⭐⭐⭐ | **SwiFT** | Swin 4D fMRI Transformer | 🔄 진행 중 |
| ⭐⭐ | **Mamba basics** | SSM, 선별적 압축, Parallel Scan | ✅ 정리 완료 |
| ⭐ | BrainLM | ROI 기반 Brain Language Model | ⬜ 예정 |
| ⭐ | Brain-JEPA | Joint Embedding 방식 | ⬜ 예정 |
| ⭐ | NeuroSTORM | Mamba 기반 (그리드 방식) | ⬜ 예정 |

### 2️⃣ 기초 지식

| 분야 | 학습 내용 | 리소스 |
| --- | --- | --- |
| 🧠 **fMRI 기초** | 복셀, TR, BOLD 신호, 전처리 | Andy’s Brain Book |
| 🐍 **Mamba/SSM** | 상태 공간 모델, 이산화, HiPPO | 논문 + 블로그 |
| 🔄 **Transformer** | Attention, Swin Transformer | 기존 지식 복습 |
| 📊 **딥러닝 일반** | PyTorch Lightning, 분산 학습 | 겨울방학 코딩 기체단 |

### 3️⃣ 코드 분석 & 실행

```
NeuroMamba 코드베이스
├── 📁 모델 아키텍처 (fmamba.py)
├── 📁 학습 파이프라인 (pl_classifier.py)
├── 📁 데이터 로딩 (datasets.py, data_module.py)
├── 📁 유틸리티 (losses.py, lr_scheduler.py)
└── 📁 전처리 (preprocessing.py)
```

| 단계 | 목표 | 상태 |
| --- | --- | --- |
| 1 | 코드 구조 파악  |  |
| 2 | 각 컴포넌트 역할 이해 |  |
| 3 | 더미 데이터로 실행 테스트 |  |
| 4 | 디버깅 & 수정 경험 |  |
| 5 | 실제 데이터로 학습 |  |

### 4️⃣ 서버 & 인프라

| 스킬 | 세부 내용 |
| --- | --- |
| 🐧 **Linux CLI** | 기본 명령어, 파일 시스템, 권한 |
| 📦 **SLURM** | 작업 제출(sbatch), 모니터링(squeue), 리소스 관리 |
| 🐳 **환경 관리** | Conda/venv, 의존성 관리 |
| 🔥 **GPU 활용** | CUDA, 멀티 GPU 학습, DeepSpeed |

---

## 💡 NeuroMamba 개선 아이디어 (향후 탐구)

### 학습 방법 변경

| 현재 | 대안 아이디어 |
| --- | --- |
| Next-token Prediction |  |

### 모델 아키텍처

| 아이디어 | 설명 |
| --- | --- |
| **Hymba** | Mamba + Transformer 하이브리드 |
| **Jamba** | Mamba + Attention 교차 배치 |

### 손실 함수

| 현재 | 대안 |
| --- | --- |
| MSE Loss |  |

### 데이터 확장

| 데이터 유형 | 특징 |
| --- | --- |
| **Movie-watching fMRI** | 더 긴 시퀀스, 풍부한 맥락 |

---

## 📅 주간 계획

### Week 1: 1월 5일 ~ 9일

### 화요일 (1/7)

| 시간 | 할 일 | 상세 |
| --- | --- | --- |
| 🌅 오전 | **논문 리딩** | SwiFT (필수),  BrainLM, BrainJEPA(시간나면) |
| 🌆 오후 | **fMRI 공부** | Andy’s Brain Book fMRI Short Course |

### 수요일 (1/8)

| 시간 | 할 일 | 상세 |
| --- | --- | --- |
| 🌅 종일 | **코드 리뷰** | NeuroMamba (필수), SwiFT (시간 되면) |
|  |  | Cursor로 코드 분석 방법 익히기 |

### 목요일 (1/9)

| 시간 | 할 일 | 상세 |
| --- | --- | --- |
| 🌅 종일 | **서버 사용법** | 랩서버 매뉴얼, SLURM 사용법 |
|  |  | Linux 기본 명령어, CLI 사용법 |

### 추가 태스크

- [ ]  **NotebookLM 만들기**
    - 논문 PDF 업로드
    - 관련 웹사이트 추가
    - 핵심 코드 `.py` → `.txt` 변환하여 업로드

---

## ✅ 체크리스트

### 📖 논문 & 이론

- [x]  NeuroMamba 논문 읽기
- [x]  NeuroMamba 논문 정리 (마크다운)
- [x]  Mamba/SSM 기초 정리
- [ ]  SwiFT 논문 읽기
- [ ]  SwiFT 논문 정리
- [ ]  fMRI 기초 공부 (Andy’s Brain Book)
- [ ]  BrainLM 논문 읽기
- [ ]  Brain-JEPA 논문 읽기
- [ ]  NeuroSTORM 논문 읽기

### 💻 코드 & 실습

- [ ]  NeuroMamba 코드 분석
- [ ]  SwiFT 코드 분석
- [ ]  더미 데이터로 NeuroMamba 실행
- [ ]  디버깅 연습
- [ ]  실제 데이터로 학습 실행

### 🖥️ 인프라

- [ ]  Linux 기본 명령어 학습
- [ ]  SLURM 사용법 학습
- [ ]  랩서버 접속 & 환경 설정
- [ ]  GPU 할당 및 작업 제출

### 🔧 개선 실험

- [ ]  학습 방법 변경 실험
- [ ]  다른 손실 함수 테스트
- [ ]  긴 시퀀스 데이터 테스트
- [ ]  하이브리드 모델 탐구

---

## 📂 생성된 정리 문서

| 파일명 | 내용 | 상태 |  |
| --- | --- | --- | --- |
| `학부연구생_로드맵.md` | 이 문서 (목표/계획) | ✅ |  |
| `Mamba_basics.md` | Mamba/SSM 이론 정리 | ✅ |  |
| `NeuroMamba.md` | NeuroMamba 논문 요약 | ✅ |  |
| `NeuroMamba_코드분석.md` | 전체 코드 구조 및 파일별 역할 | ✅ |  |

[Mamba_basics/naive](https://www.notion.so/Mamba_basics-naive-2e1e2365873a817781efffbd37d4bfca?pvs=21)

[Mamba_basics](https://www.notion.so/Mamba_basics-2e1e2365873a8171840adea13ee9766c?pvs=21)

[Neuromamba/naive](https://www.notion.so/Neuromamba-naive-2e1e2365873a81a59685fb9ade9a5276?pvs=21)

[NeuroMamba](https://www.notion.so/NeuroMamba-2e1e2365873a8154b1c9f8b17cc58201?pvs=21)

[NeuroMamba_코드분석](https://www.notion.so/NeuroMamba_-2e1e2365873a81b9bdfde54c7c8b0e5d?pvs=21)

---

## 🎓 기대 성과

### 단기 (1~2주)

- [ ]  핵심 논문 이해
- [ ]  코드 실행 환경 구축
- [ ]  서버 사용 익히기

### 중기 (1~2개월)

- [ ]  실험 수행 가능
- [ ]  코드 수정 및 개선 경험
- [ ]  첫 번째 결과 도출

### 장기 (전체)

- [ ]  연구 인사이트 획득
- [ ]  연구 경험 & 스킬 습득
- [ ]  AI 협업 연구 방법론 체득

---

## 💬 메모 & 질문

### 교수님/선생님께 여쭤볼 것

- [ ]  [ ]
- [ ]  [ ]
- [ ]  [ ]

## ### 아이디어 & 인사이트

- 
- 

---

*마지막 업데이트: 2026.01.06*
