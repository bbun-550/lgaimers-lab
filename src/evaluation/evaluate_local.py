"""
LG Aimers - 로컬 모델 평가 스크립트
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

목적: 양자화된 모델들을 기본 모델(EXAONE-4.0-1.2B) 대비 비교 평가하여
      가장 좋은 모델을 선별한 후 대회 서버에 제출

평가 산식:
  Score = max(0.5 × PerfNorm + 0.5 × SpeedNorm, 0)
  - PerfNorm  = 모델 벤치마크 정확도 / 기본 모델 벤치마크 정확도
  - SpeedNorm = 1 - (모델 토큰당 시간) / (기본 모델 토큰당 시간)

PerfNorm 측정:
  lm-evaluation-harness를 사용한 실제 벤치마크 (MMLU, GSM8K 등)

SpeedNorm 측정:
  동일 환경에서의 토큰 생성 속도 상대 비교

사전 설치:
  pip install lm-eval torch transformers accelerate safetensors

사용법:
  # 기본 모델 baseline 측정 (최초 1회)
  python evaluate_local.py --base-model ./base_model --mode baseline

  # 양자화 모델 평가 (기본 모델 대비 비교)
  python evaluate_local.py --base-model ./base_model --target-model ./model_DB/optimized_submit/model

  # 여러 모델 한번에 비교
  python evaluate_local.py --base-model ./base_model --target-model ./modelA ./modelB ./modelC

  # 벤치마크 태스크 지정 (기본: gsm8k,mmlu)
  python evaluate_local.py --base-model ./base_model --target-model ./model --tasks gsm8k,mmlu

  # 속도 측정 생략 (정확도만 비교)
  python evaluate_local.py --base-model ./base_model --target-model ./model --skip-speed

  # 이전 baseline 결과 재사용 (시간 절약)
  python evaluate_local.py --target-model ./model --baseline-json ./baseline_result.json
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional, List, Dict

import torch


# =========================================================
# 데이터 클래스
# =========================================================

@dataclass
class ModelResult:
    """단일 모델 평가 결과"""
    model_path: str
    # 벤치마크 정확도 (PerfNorm 산출용)
    benchmark_scores: Dict[str, float] = field(default_factory=dict)
    avg_accuracy: float = 0.0
    # 속도 (SpeedNorm 산출용)
    time_per_token_ms: float = 0.0
    tokens_per_sec: float = 0.0
    total_tokens: int = 0
    total_time_sec: float = 0.0
    # 모델 정보
    num_parameters: int = 0
    model_size_mb: float = 0.0


@dataclass
class ComparisonEntry:
    """모델 간 비교 결과 (한 줄)"""
    model_path: str
    avg_accuracy: float
    perf_norm: float
    time_per_token_ms: float
    speed_norm: float
    score: float
    benchmark_details: Dict[str, float] = field(default_factory=dict)


# =========================================================
# 1. 벤치마크 평가 (PerfNorm용) - lm-evaluation-harness 사용
# =========================================================

def run_benchmarks(model_path: str, tasks: List[str], 
                   batch_size: str = "auto", num_fewshot: int = None) -> Dict[str, float]:
    """
    lm-evaluation-harness를 사용하여 벤치마크 정확도 측정
    
    Returns:
        Dict[task_name, accuracy]  (0.0 ~ 1.0)
    """
    import lm_eval

    print(f"\n  📊 벤치마크 평가 시작: {', '.join(tasks)}")
    print(f"     모델: {model_path}")
    
    model_args = f"pretrained={model_path},trust_remote_code=True"
    
    # GPU VRAM 부족 시 dtype 지정
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if vram_gb < 12:
            model_args += ",dtype=float16"
    
    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=tasks,
        batch_size=batch_size,
        num_fewshot=num_fewshot,
    )
    
    # 태스크별 정확도 추출
    scores = {}
    for task_name in tasks:
        task_result = results["results"].get(task_name, {})
        
        # lm-eval은 태스크에 따라 다른 metric명을 사용
        # 우선순위: acc_norm > acc > exact_match
        acc = None
        for metric_key in ["acc_norm,none", "acc,none", "exact_match,none",
                           "acc_norm", "acc", "exact_match"]:
            if metric_key in task_result:
                acc = task_result[metric_key]
                break
        
        if acc is not None:
            scores[task_name] = acc
            print(f"     ✅ {task_name}: {acc:.4f} ({acc*100:.2f}%)")
        else:
            # 하위 태스크가 있는 경우 (예: mmlu는 여러 subject)
            # 그룹 평균 찾기
            for key, val in task_result.items():
                if "acc" in key and isinstance(val, (int, float)):
                    scores[task_name] = val
                    print(f"     ✅ {task_name}: {val:.4f} ({val*100:.2f}%)")
                    break
            else:
                print(f"     ⚠️ {task_name}: 결과를 찾을 수 없음 (건너뜁니다)")
                print(f"        사용 가능한 키: {list(task_result.keys())}")
    
    return scores


# =========================================================
# 2. 속도 평가 (SpeedNorm용) - HF generate() 상대 비교
# =========================================================

SPEED_PROMPTS = [
    "Explain the concept of machine learning in simple terms.",
    "What are the benefits of renewable energy?",
    "Write a short paragraph about artificial intelligence.",
    "Describe the process of photosynthesis.",
    "What is the capital of France and why is it famous?",
    "인공지능의 미래에 대해 설명해주세요.",
    "한국의 전통 음식 중 하나를 소개해주세요.",
    "프로그래밍을 배우는 좋은 방법은 무엇인가요?",
    "Solve: If a train travels 60km/h for 2 hours, how far?",
    "What is the difference between a stack and a queue?",
    "딥러닝과 머신러닝의 차이점을 설명해주세요.",
    "Write a Python function to reverse a string.",
]


def measure_speed(model_path: str, max_new_tokens: int = 128) -> Dict:
    """
    HuggingFace model.generate()로 토큰 생성 속도 측정
    (상대 비교 목적, 절대 수치는 대회와 다를 수 있음)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"\n  ⏱️  속도 측정 시작 ({len(SPEED_PROMPTS)}개 프롬프트, max_tokens={max_new_tokens})")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True,
        local_files_only=os.path.isdir(model_path),
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
        local_files_only=os.path.isdir(model_path),
    )
    model.eval()
    
    # 워밍업
    warmup = tokenizer("Hello", return_tensors="pt").to(device)
    with torch.no_grad():
        _ = model.generate(**warmup, max_new_tokens=5)
    if device == "cuda":
        torch.cuda.synchronize()
    
    total_tokens = 0
    total_time = 0.0
    
    for prompt in SPEED_PROMPTS:
        messages = [{"role": "user", "content": prompt}]
        try:
            input_ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            ).to(device)
        except Exception:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            input_ids = inputs["input_ids"]
        
        if device == "cuda":
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        with torch.no_grad():
            output = model.generate(
                input_ids, max_new_tokens=max_new_tokens,
                do_sample=False, pad_token_id=tokenizer.eos_token_id
            )
        if device == "cuda":
            torch.cuda.synchronize()
        
        elapsed = time.perf_counter() - start
        new_tokens = output.shape[1] - input_ids.shape[1]
        total_tokens += new_tokens
        total_time += elapsed
    
    # 메모리 정리
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    
    result = {
        "total_time_sec": total_time,
        "total_tokens": total_tokens,
        "tokens_per_sec": total_tokens / total_time if total_time > 0 else 0,
        "time_per_token_ms": (total_time / total_tokens * 1000) if total_tokens > 0 else 0,
    }
    
    print(f"     Tokens/sec: {result['tokens_per_sec']:.2f}")
    print(f"     Time/token: {result['time_per_token_ms']:.2f} ms")
    
    return result


# =========================================================
# 3. 전체 평가 + 점수 계산
# =========================================================

def evaluate_model(model_path: str, tasks: List[str],
                   skip_speed: bool = False, max_new_tokens: int = 128) -> ModelResult:
    """단일 모델 전체 평가"""
    
    print(f"\n{'━' * 60}")
    print(f"  📌 평가 모델: {model_path}")
    print(f"{'━' * 60}")
    
    # 모델 크기 확인
    model_dir = Path(model_path)
    model_size_mb = 0
    if model_dir.is_dir():
        for f in model_dir.glob("*.safetensors"):
            model_size_mb += f.stat().st_size / (1024 * 1024)
        for f in model_dir.glob("*.bin"):
            model_size_mb += f.stat().st_size / (1024 * 1024)
        print(f"  모델 가중치 크기: {model_size_mb:.1f} MB")
    
    # 벤치마크 평가
    scores = run_benchmarks(model_path, tasks)
    avg_acc = sum(scores.values()) / len(scores) if scores else 0.0
    print(f"  📊 평균 정확도: {avg_acc:.4f} ({avg_acc*100:.2f}%)")
    
    # 속도 측정
    speed = {"time_per_token_ms": 0, "tokens_per_sec": 0, "total_tokens": 0, "total_time_sec": 0}
    if not skip_speed:
        # 벤치마크에서 사용한 모델 메모리 해제 후 속도 측정
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        speed = measure_speed(model_path, max_new_tokens)
    
    return ModelResult(
        model_path=model_path,
        benchmark_scores=scores,
        avg_accuracy=avg_acc,
        time_per_token_ms=speed["time_per_token_ms"],
        tokens_per_sec=speed["tokens_per_sec"],
        total_tokens=speed["total_tokens"],
        total_time_sec=speed["total_time_sec"],
        model_size_mb=model_size_mb,
    )


def calculate_score(base: ModelResult, target: ModelResult, skip_speed: bool = False) -> ComparisonEntry:
    """대회 산식에 따른 점수 계산"""
    
    # PerfNorm = target 정확도 / base 정확도
    if base.avg_accuracy > 0:
        perf_norm = target.avg_accuracy / base.avg_accuracy
    else:
        perf_norm = 1.0
    
    # SpeedNorm = 1 - (target time/token) / (base time/token)
    if not skip_speed and base.time_per_token_ms > 0 and target.time_per_token_ms > 0:
        speed_norm = 1 - (target.time_per_token_ms / base.time_per_token_ms)
    else:
        speed_norm = 0.0  # 속도 미측정 시 0으로 처리
    
    # Score
    score = max(0.5 * perf_norm + 0.5 * speed_norm, 0)
    
    return ComparisonEntry(
        model_path=target.model_path,
        avg_accuracy=target.avg_accuracy,
        perf_norm=perf_norm,
        time_per_token_ms=target.time_per_token_ms,
        speed_norm=speed_norm,
        score=score,
        benchmark_details=target.benchmark_scores,
    )


# =========================================================
# 4. 결과 출력
# =========================================================

def print_comparison(base: ModelResult, entries: List[ComparisonEntry], skip_speed: bool):
    """각 모델을 기본 모델 대비 비율로 개별 출력"""
    
    print("\n\n" + "=" * 80)
    print("  🏆 LG Aimers 로컬 평가 결과 (lm-evaluation-harness 기반)")
    print("=" * 80)
    
    # 기본 모델 (기준)
    base_name = Path(base.model_path).name or "base"
    print(f"\n📋 기준 모델: {base_name}")
    print(f"{'─' * 80}")
    print(f"  경로:       {base.model_path}")
    print(f"  평균 정확도: {base.avg_accuracy:.4f} ({base.avg_accuracy*100:.2f}%)")
    for task, score in base.benchmark_scores.items():
        print(f"    - {task}: {score:.4f}")
    if not skip_speed:
        print(f"  Time/token: {base.time_per_token_ms:.2f} ms")
    print(f"  → 이 모델이 PerfNorm=1.0, SpeedNorm=0.0, Score=0.5 의 기준입니다.")
    
    # ── 각 모델을 개별적으로 기본 모델 대비 비교 ──
    entries_sorted = sorted(entries, key=lambda x: x.score, reverse=True)
    tasks = list(base.benchmark_scores.keys())
    
    for idx, e in enumerate(entries_sorted, 1):
        model_name = Path(e.model_path).name or e.model_path
        
        print(f"\n\n{'━' * 80}")
        print(f"  📌 [{idx}] {model_name} / 기준 모델 비교")
        print(f"{'━' * 80}")
        
        # 태스크별 비율
        print(f"\n  🎯 PerfNorm (벤치마크 정확도 비율)")
        print(f"  {'─' * 60}")
        print(f"  {'태스크':<15} {'기준 모델':<12} {'이 모델':<12} {'비율 (모델/기준)':<18}")
        print(f"  {'─' * 60}")
        
        for task in tasks:
            base_score = base.benchmark_scores.get(task, 0)
            target_score = e.benchmark_details.get(task, 0)
            ratio = target_score / base_score if base_score > 0 else 0
            arrow = "✅" if ratio >= 0.95 else ("⚠️" if ratio >= 0.85 else "❌")
            print(f"  {task:<15} {base_score:.4f}       {target_score:.4f}       {ratio:.4f} ({ratio*100:.1f}%)  {arrow}")
        
        # 평균
        print(f"  {'─' * 60}")
        print(f"  {'평균':<15} {base.avg_accuracy:.4f}       {e.avg_accuracy:.4f}       {e.perf_norm:.4f} ({e.perf_norm*100:.1f}%)")
        print(f"\n  → PerfNorm = {e.avg_accuracy:.4f} / {base.avg_accuracy:.4f} = {e.perf_norm:.4f}")
        
        # 속도 비율
        if not skip_speed:
            print(f"\n  ⏱️  SpeedNorm (토큰당 추론 시간 비율)")
            print(f"  {'─' * 60}")
            print(f"  기준 모델 Time/token:  {base.time_per_token_ms:.2f} ms")
            print(f"  이 모델 Time/token:    {e.time_per_token_ms:.2f} ms")
            time_ratio = e.time_per_token_ms / base.time_per_token_ms if base.time_per_token_ms > 0 else 1
            print(f"  시간 비율:             {time_ratio:.4f} ({time_ratio*100:.1f}%)")
            speed_arrow = "✅ 빨라짐" if e.speed_norm > 0 else ("⚡ 동일" if e.speed_norm == 0 else "❌ 느려짐")
            print(f"\n  → SpeedNorm = 1 - {e.time_per_token_ms:.2f} / {base.time_per_token_ms:.2f} = {e.speed_norm:+.4f}  {speed_arrow}")
        
        # 최종 Score
        print(f"\n  🏆 최종 Score")
        print(f"  {'─' * 60}")
        print(f"  Score = max(0.5 × PerfNorm + 0.5 × SpeedNorm, 0)")
        print(f"        = max(0.5 × {e.perf_norm:.4f} + 0.5 × {e.speed_norm:+.4f}, 0)")
        print(f"        = {e.score:.4f}")
        
        if e.score > 0.5:
            print(f"\n  ✅ 수료 기준 (> 0.5) 통과!  (기준 대비 +{e.score - 0.5:.4f})")
        else:
            print(f"\n  ❌ 수료 기준 (> 0.5) 미달  (부족분: {0.5 - e.score:.4f})")
    
    # ── 최종 요약 순위 ──
    print(f"\n\n{'=' * 80}")
    print(f"  📊 최종 순위 요약 (모든 모델 / 기준 모델 비교)")
    print(f"{'=' * 80}")
    print(f"  {'순위':<4} {'모델':<28} {'PerfNorm':<10} {'SpeedNorm':<11} {'Score':<8} {'판정'}")
    print(f"{'─' * 80}")
    
    for i, e in enumerate(entries_sorted, 1):
        name = Path(e.model_path).name or e.model_path
        if len(name) > 26:
            name = name[:23] + "..."
        verdict = "✅ 통과" if e.score > 0.5 else "❌ 미달"
        star = " ⭐ BEST" if i == 1 else ""
        print(f"  {i:<4} {name:<28} {e.perf_norm:.4f}    {e.speed_norm:+.4f}    {e.score:.4f}  {verdict}{star}")
    
    print(f"{'─' * 80}")
    print(f"  ref  {'기준(EXAONE-4.0-1.2B)':<28} 1.0000    +0.0000    0.5000  기준선")
    print(f"{'=' * 80}")
    
    if skip_speed:
        print(f"  ⚠️  SpeedNorm 미측정: 실제 Score는 속도 개선분만큼 더 높을 수 있음")
    print(f"  ⚠️  PerfNorm은 공개 벤치마크 기준이며 대회 비공개 벤치셋과 차이 가능")
    
    return entries_sorted


def save_result(base: ModelResult, entries: List[ComparisonEntry], output_path: str):
    """결과 JSON 저장"""
    data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "baseline": asdict(base),
        "models": [asdict(e) for e in entries],
        "ranking": [
            {"rank": i+1, "model": Path(e.model_path).name, "score": e.score}
            for i, e in enumerate(sorted(entries, key=lambda x: x.score, reverse=True))
        ],
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\n💾 결과 저장: {output_path}")


# =========================================================
# 메인
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description="LG Aimers 로컬 모델 평가 (lm-evaluation-harness 기반)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 1. baseline 측정 + 저장
  python evaluate_local.py --base-model ./base_model --mode baseline

  # 2. 양자화 모델 1개 평가
  python evaluate_local.py --base-model ./base_model --target-model ./model_DB/optimized_submit/model

  # 3. 여러 모델 한번에 비교
  python evaluate_local.py --base-model ./base_model --target-model ./modelA ./modelB ./modelC

  # 4. 저장된 baseline 재사용 (시간 절약)
  python evaluate_local.py --baseline-json ./baseline_result.json --target-model ./modelA

  # 5. 정확도만 비교 (속도 생략)
  python evaluate_local.py --base-model ./base_model --target-model ./model --skip-speed
        """
    )
    
    parser.add_argument("--base-model", type=str, default=None,
                        help="기본 모델 경로 (EXAONE-4.0-1.2B)")
    parser.add_argument("--target-model", type=str, nargs="+", default=None,
                        help="평가할 양자화 모델 경로 (여러 개 가능)")
    parser.add_argument("--mode", choices=["baseline", "compare"], default="compare",
                        help="baseline: 기본 모델만 측정 / compare: 비교 평가")
    parser.add_argument("--tasks", type=str, default="gsm8k,mmlu",
                        help="벤치마크 태스크 (쉼표 구분, 기본: gsm8k,mmlu)")
    parser.add_argument("--skip-speed", action="store_true",
                        help="속도 측정 생략 (정확도만 비교)")
    parser.add_argument("--max-tokens", type=int, default=128,
                        help="속도 측정 시 최대 생성 토큰 수")
    parser.add_argument("--baseline-json", type=str, default=None,
                        help="이전에 저장한 baseline 결과 JSON 경로 (재측정 생략)")
    parser.add_argument("--output", type=str, default=None,
                        help="결과 저장 경로 (기본: 자동 생성)")
    
    args = parser.parse_args()
    tasks = [t.strip() for t in args.tasks.split(",")]
    
    print("\n" + "=" * 80)
    print("  LG Aimers 로컬 모델 평가 (lm-evaluation-harness 기반)")
    print("=" * 80)
    print(f"  벤치마크: {', '.join(tasks)}")
    print(f"  속도 측정: {'생략' if args.skip_speed else '실행'}")
    
    # ─── Baseline 처리 ─────────────────────────────
    base_result = None
    
    if args.baseline_json:
        # 저장된 baseline 로드
        print(f"\n  📂 Baseline 로드: {args.baseline_json}")
        with open(args.baseline_json, "r") as f:
            data = json.load(f)
        base_data = data if "benchmark_scores" in data else data.get("baseline", data)
        base_result = ModelResult(**{k: v for k, v in base_data.items() if k in ModelResult.__dataclass_fields__})
        print(f"     평균 정확도: {base_result.avg_accuracy:.4f}")
    
    elif args.base_model:
        # 기본 모델 평가
        base_result = evaluate_model(args.base_model, tasks, args.skip_speed, args.max_tokens)
        
        # Baseline 결과 저장
        baseline_path = "baseline_result.json"
        save_result(base_result, [], baseline_path)
        print(f"  💾 Baseline 저장됨 → 다음부터 --baseline-json {baseline_path} 로 재사용 가능")
    
    if args.mode == "baseline":
        if base_result:
            print(f"\n✅ Baseline 측정 완료!")
            print(f"   평균 정확도: {base_result.avg_accuracy:.4f}")
            for t, s in base_result.benchmark_scores.items():
                print(f"   - {t}: {s:.4f}")
        else:
            print("❌ --base-model 을 지정해주세요")
        return
    
    # ─── Target 모델 평가 ─────────────────────────
    if not args.target_model:
        print("❌ --target-model 을 지정해주세요")
        return
    
    if base_result is None:
        print("❌ --base-model 또는 --baseline-json 을 지정해주세요")
        return
    
    entries = []
    for model_path in args.target_model:
        target_result = evaluate_model(model_path, tasks, args.skip_speed, args.max_tokens)
        entry = calculate_score(base_result, target_result, args.skip_speed)
        entries.append(entry)
        
        # 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # ─── 결과 출력 ─────────────────────────────────
    sorted_entries = print_comparison(base_result, entries, args.skip_speed)
    
    # ─── 결과 저장 ─────────────────────────────────
    if args.output is None:
        output_path = f"eval_comparison_{time.strftime('%Y%m%d_%H%M%S')}.json"
    else:
        output_path = args.output
    
    save_result(base_result, sorted_entries, output_path)


if __name__ == "__main__":
    main()
