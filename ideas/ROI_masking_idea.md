# ROI-based Masking 통합 아이디어 요약

---

## 🎯 핵심 아이디어

### 현재 방식 → ROI 기반 방식

**현재**: 배경 값 기반 필터링 (단순 threshold)
```
패치 → 배경 값 < threshold → 뇌 토큰 선택
```

**제안**: ROI 마스크 기반 필터링 (신경과학적 의미)
```
패치 → ROI 마스크와 매칭 → ROI에 속한 패치만 선택
```

---

## 📊 두 가지 접근 방식

| 방식                              | 설명                                  | 장점                                           | 단점                         |
| --------------------------------- | ------------------------------------- | ---------------------------------------------- | ---------------------------- |
| **HCP-MMP**<br>(Population-based) | 모든 피험자에게 동일한 360개 ROI 적용 | • 일관성<br>• 해석 용이<br>• 사전 학습에 유리  | 개별 뇌의 해부학적 변이 무시 |
| **ICA**<br>(Individualized)       | 각 피험자마다 고유한 ROI 생성         | • 개별 뇌 특성 반영<br>• 더 정확한 기능적 경계 | • 계산 비용<br>• 일관성 문제 |

---

## ⚠️ 구현 전략 단계 부터는 AI agent 제시이므로 추가 탐색 및 검증이 필요함.


## 🏗️ 구현 전략 (개요)

### 핵심 컴포넌트

```
[ROI 마스크 로드] → [패치-ROI 매칭] → [선택된 패치만 Mamba 입력]
```

1. **ROIMaskLoader**: HCP-MMP 또는 ICA 로드
2. **PatchROIMatcher**: 패치 좌표 ↔ ROI 레이블 매칭
3. **ROIPatchEmbed**: 기존 `PatchEmbed` 확장

### 구현 단계

1. **Phase 1**: ROI 로더 + 패치 매칭 로직
2. **Phase 2**: `PatchEmbed` 확장, `FMamba` 통합
3. **Phase 3**: 실험 및 평가 (Baseline vs ROI 마스킹)

---

## 🔬 예상 효과

### 장점
- **해석 가능성 향상**: ROI별 기여도 분석 가능
- **계산 효율성**: 더 정확한 뇌 영역만 선택
- **신경과학적 의미**: 기존 ROI 연구와 연결 가능
- **개별화**: ICA 사용 시 개별 뇌 특성 반영

### 잠재적 이슈
- ROI 정의 의존성: ROI 품질에 성능 의존
- 계산 오버헤드: ROI 로드 및 매칭 비용
- 일관성: Individualized 방식의 경우 피험자 간 비교 어려움

---

## 📝 참고 자료

- **HCP-MMP**: https://mne.tools/stable/generated/mne.datasets.fetch_hcp_mmp_parcellation.html
- **CanICA**: https://github.com/GaelVaroquaux/canica
- **상세 계획**: `ROI_masking_plan.md` 참조

---

**작성일**: 2025년  
**상태**: 아이디어 요약 단계





