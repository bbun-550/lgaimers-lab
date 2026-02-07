"""
EXAONE 경량화 실험 보고서 자동 생성기

Usage:
    make report
    # 또는
    uv run python src/compression/report.py --experiment drop28 --model ./submit/model_drop28
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys

# Project Root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_baseline() -> dict:
    """Baseline 결과 로드"""
    baseline_path = PROJECT_ROOT / "outputs" / "baseline_result.json"
    if baseline_path.exists():
        with open(baseline_path) as f:
            return json.load(f)
    return {
        "perplexity": 2659.93,
        "tokens_per_sec": 27.30,
        "time_per_token_ms": 36.63,
        "num_params": 1_280_000_000,
    }


def load_experiment_result(model_path: str) -> dict:
    """실험 결과 로드 (JSON 파일 또는 수동 입력)"""
    result_path = Path(model_path) / "eval_result.json"
    if result_path.exists():
        with open(result_path) as f:
            return json.load(f)
    
    # 결과 파일이 없으면 빈 딕셔너리 반환
    return {}


def calculate_scores(result: dict, baseline: dict) -> dict:
    """PerfNorm, SpeedNorm, Score 계산"""
    if not result:
        return {"perf_norm": None, "speed_norm": None, "score": None}
    
    perf_norm = baseline["perplexity"] / result.get("perplexity", 1) if result.get("perplexity") else None
    
    base_time = baseline.get("time_per_token_ms", 36.63)
    model_time = result.get("time_per_token_ms", base_time)
    speed_norm = 1 - (model_time / base_time) if base_time else 0
    
    if perf_norm is not None:
        score = max(0.5 * perf_norm + 0.5 * speed_norm, 0)
    else:
        score = None
    
    return {
        "perf_norm": perf_norm,
        "speed_norm": speed_norm,
        "score": score
    }


def format_number(value, precision=4):
    """숫자 포맷팅 (None 처리)"""
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{precision}f}"
    return str(value)


def format_params(num_params):
    """파라미터 수 포맷팅 (1.28B 형식)"""
    if num_params is None:
        return "N/A"
    if num_params >= 1e9:
        return f"{num_params/1e9:.2f}B"
    elif num_params >= 1e6:
        return f"{num_params/1e6:.2f}M"
    return str(num_params)


def generate_report(
    experiment_name: str,
    model_path: str,
    description: str = "",
    conclusion: str = ""
) -> str:
    """보고서 마크다운 생성"""
    
    today = datetime.now().strftime("%Y-%m-%d")
    baseline = load_baseline()
    result = load_experiment_result(model_path)
    scores = calculate_scores(result, baseline)
    
    # 성공/실패 판단
    if scores["score"] is None:
        status = "⚠️ 평가 필요"
        status_emoji = "⚠️"
    elif scores["score"] >= 0.5:
        status = "✅ 성공 (Baseline 초과)"
        status_emoji = "✅"
    else:
        status = "❌ 실패 (Baseline 미달)"
        status_emoji = "❌"
    
    report = f"""# EXAONE 경량화 실험 보고서

> **작성일**: {today}  
> **실험명**: {experiment_name}  
> **모델 경로**: `{model_path}`

---

## 1. 실험 개요

### 1.1 목표
{description if description else f"{experiment_name} 전략을 적용하여 모델 경량화 및 성능 평가"}

### 1.2 주요 결과 요약
- **결과**: {status}
- **Score**: {format_number(scores['score'])}
- **PerfNorm**: {format_number(scores['perf_norm'])}
- **SpeedNorm**: {format_number(scores['speed_norm'])}

---

## 2. 실험 결과 비교

| 모델 | Params | Tokens/sec | Perplexity | PerfNorm | SpeedNorm | **Score** |
|------|--------|------------|------------|----------|-----------|-----------|
| **Baseline** | {format_params(baseline.get('num_params'))} | {format_number(baseline.get('tokens_per_sec'), 2)} | {format_number(baseline.get('perplexity'), 2)} | 1.0000 | 0.0000 | **0.5000** |
| **{experiment_name}** | {format_params(result.get('num_params'))} | {format_number(result.get('tokens_per_sec'), 2)} | {format_number(result.get('perplexity'), 2)} | {format_number(scores['perf_norm'])} | {format_number(scores['speed_norm'])} | **{format_number(scores['score'])}** |

---

## 3. 분석

### 3.1 속도 변화
- Baseline 대비 SpeedNorm: **{format_number(scores['speed_norm'])}**
- {"속도가 개선되었습니다." if scores['speed_norm'] and scores['speed_norm'] > 0 else "속도 개선이 미미하거나 없습니다."}

### 3.2 성능 변화  
- Baseline 대비 PerfNorm: **{format_number(scores['perf_norm'])}**
- {"성능이 유지되었습니다." if scores['perf_norm'] and scores['perf_norm'] >= 0.9 else "성능 손실이 발생했습니다." if scores['perf_norm'] else "성능 평가가 필요합니다."}

---

## 4. 결론 및 제안

{conclusion if conclusion else f'''
### {status_emoji} 결론
{"이 전략은 Baseline Score(0.5)를 초과하여 **제출 가능**합니다." if scores['score'] and scores['score'] >= 0.5 else "이 전략은 Baseline Score(0.5)에 미달하여 **제출 비권장**입니다." if scores['score'] else "평가 결과를 확인한 후 제출 여부를 결정하세요."}

### 📌 다음 단계
1. `make eval-model` 로 평가 실행 (아직 안 했다면)
2. MLflow에서 결과 확인
3. 제출 여부 결정
'''}
"""
    return report


def save_report(report: str, experiment_name: str):
    """보고서 저장"""
    today = datetime.now().strftime("%Y-%m-%d")
    output_dir = PROJECT_ROOT / "outputs" / f"{today}_{experiment_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = output_dir / f"{today}_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    
    print(f"✅ 보고서 저장: {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(description="EXAONE 경량화 실험 보고서 생성")
    parser.add_argument("--experiment", "-e", type=str, default="experiment",
                        help="실험 이름 (예: drop28, fp16)")
    parser.add_argument("--model", "-m", type=str, default="./submit/model",
                        help="경량화 모델 경로")
    parser.add_argument("--description", "-d", type=str, default="",
                        help="실험 설명")
    parser.add_argument("--conclusion", "-c", type=str, default="",
                        help="결론 (직접 입력)")
    parser.add_argument("--print-only", action="store_true",
                        help="파일 저장 없이 출력만")
    
    args = parser.parse_args()
    
    print(f"📝 보고서 생성 중...")
    print(f"  - 실험: {args.experiment}")
    print(f"  - 모델: {args.model}")
    
    report = generate_report(
        experiment_name=args.experiment,
        model_path=args.model,
        description=args.description,
        conclusion=args.conclusion
    )
    
    if args.print_only:
        print("\n" + "="*60)
        print(report)
    else:
        save_report(report, args.experiment)
        print("\n💡 Tip: 더 상세한 분석이 필요하면 AI에게 요청하세요!")


if __name__ == "__main__":
    main()
