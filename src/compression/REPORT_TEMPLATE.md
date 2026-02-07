# EXAONE 경량화 실험 보고서

> 작성자: [이름]  
> 작성일: YYYY-MM-DD  
> 팀명: [팀명]

---

## 1. 실험 개요

### 1.1 목표

EXAONE-4.0-1.2B 모델의 경량화를 통해 추론 속도를 개선하면서 성능 저하를 최소화

### 1.2 평가 지표

$$Score = max(0.5 \times PerfNorm + 0.5 \times SpeedNorm, 0)$$

| 지표      | 수식                                               | 의미                        |
| --------- | -------------------------------------------------- | --------------------------- |
| PerfNorm  | $\frac{Perf_{model}}{Perf_{base}}$                 | 성능 유지율 (높을수록 좋음) |
| SpeedNorm | $1 - \frac{Time/Token_{model}}{Time/Token_{base}}$ | 속도 개선율 (높을수록 좋음) |

### 1.3 실험 환경

| 항목         | 값                            |
| ------------ | ----------------------------- |
| 개발 환경    | macOS (M1 Pro)                |
| 평가 서버    | Ubuntu 22.04, GPU L4 (22.4GB) |
| Python       | 3.11                          |
| PyTorch      | 2.9.0                         |
| Transformers | 4.57.3                        |
| vLLM         | 0.14.1                        |

---

## 2. Baseline 결과

### 2.1 원본 모델 정보

| 항목            | 값                          |
| --------------- | --------------------------- |
| 모델            | LGAI-EXAONE/EXAONE-4.0-1.2B |
| 파라미터 수     | 1.28B                       |
| Hidden Size     | 2,048                       |
| Layers          | 30                          |
| Attention Heads | 32                          |
| KV Heads (GQA)  | 8                           |

### 2.2 Baseline 성능

| 지표       | 값        |
| ---------- | --------- |
| Tokens/sec | 30.87     |
| Time/token | 32.40 ms  |
| Perplexity | 2659.93   |
| PerfNorm   | 1.000     |
| SpeedNorm  | 0.000     |
| **Score**  | **0.500** |

---

## 3. 경량화 실험

### 3.1 실험 방법

#### 3.1.1 Layer Dropping

- **가설**: 상위 레이어는 task-specific하므로 일부 제거해도 성능 유지
- **방법**: Transformer의 상위 N개 레이어 제거

#### 3.1.2 Head Pruning (선택)

- **가설**: 일부 attention head는 redundant
- **방법**: 중요도가 낮은 head 제거

#### 3.1.3 Hidden Dimension 축소 (선택)

- **가설**: 표현 공간이 over-parameterized
- **방법**: FFN 차원 축소

### 3.2 실험 결과

| 실험          | Layers | Params | Tokens/sec | Perplexity | PerfNorm | SpeedNorm | Score |
| ------------- | ------ | ------ | ---------- | ---------- | -------- | --------- | ----- |
| Baseline      | 30     | 1.28B  | 30.87      | 2659.93    | 1.000    | 0.000     | 0.500 |
| Drop 4 layers | 26     | ?      | ?          | ?          | ?        | ?         | ?     |
| Drop 6 layers | 24     | ?      | ?          | ?          | ?        | ?         | ?     |
| Drop 8 layers | 22     | ?      | ?          | ?          | ?        | ?         | ?     |

<!-- 실험 후 결과로 채우기 -->

---

## 4. 결과 분석

### 4.1 성능 vs 속도 Trade-off

<!-- 그래프 삽입 위치 -->

```
[성능-속도 Trade-off 그래프]
- X축: SpeedNorm (속도 개선율)
- Y축: PerfNorm (성능 유지율)
- 점: 각 실험 결과
```

### 4.2 최적 경량화 수준

- **최적 레이어 수**: ? layers
- **파라미터 감소율**: ?%
- **속도 개선율**: ?%
- **성능 유지율**: ?%

### 4.3 주요 발견

1. **발견 1**: ...
2. **발견 2**: ...
3. **발견 3**: ...

---

## 5. 최종 모델 선정

### 5.1 선정 기준

- Score 최대화
- 실제 추론 환경(vLLM) 호환성

### 5.2 최종 선택

| 항목           | 값    |
| -------------- | ----- |
| 경량화 방법    | ?     |
| 레이어 수      | ?     |
| 파라미터 수    | ?     |
| **최종 Score** | **?** |

### 5.3 Baseline 대비 개선

| 지표       | Baseline | 최종 모델 | 변화 |
| ---------- | -------- | --------- | ---- |
| Tokens/sec | 30.87    | ?         | +?%  |
| Perplexity | 2659.93  | ?         | +?%  |
| Score      | 0.500    | ?         | +?   |

---

## 6. 결론

### 6.1 요약

(경량화 실험의 주요 결론 요약)

### 6.2 한계점

(실험의 한계 및 개선 가능성)

### 6.3 향후 과제

(추가로 시도해볼 수 있는 경량화 기법)

---

## 부록

### A. MLflow 실험 기록

- 실험 URL: https://dagshub.com/sthun0211/LGaimers.mlflow/

### B. 제출 파일 정보

- 파일명: submit.zip
- 크기: ? MB
- 구조:
  ```
  submit.zip
  └── model/
      ├── config.json
      ├── model.safetensors
      └── tokenizer files...
  ```

### C. 참고 자료

- [EXAONE-4.0 HuggingFace](https://huggingface.co/LGAI-EXAONE/EXAONE-4.0-1.2B)
- [vLLM Documentation](https://docs.vllm.ai/)
