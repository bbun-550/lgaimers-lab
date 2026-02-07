"""
EXAONE 모델 성능/속도 평가

평가 지표:
1. PerfNorm = Perf_model / Perf_base_model (성능 유지 비율)
2. SpeedNorm = 1 - (Time/Tokens)_model / (Time/Tokens)_base (속도 개선 비율)
3. Score = max(0.5 * PerfNorm + 0.5 * SpeedNorm, 0)

MLflow로 모든 실험 기록 (DagsHub 연동)
"""

import os
import time
import json
import torch
import mlflow
from dotenv import load_dotenv

# .env 파일에서 환경변수 로드 (MLflow 인증 정보)
load_dotenv()

# DagsHub MLflow 설정
if os.getenv("MLFLOW_TRACKING_URI"):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    print(f"📡 MLflow tracking: {os.getenv('MLFLOW_TRACKING_URI')}")
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class EvalResult:
    """평가 결과"""
    model_name: str
    num_samples: int
    total_time_sec: float
    total_tokens: int
    tokens_per_sec: float
    time_per_token_ms: float
    # 성능 지표 (perplexity 등)
    perplexity: Optional[float] = None
    # 경량화 비교용
    perf_norm: Optional[float] = None
    speed_norm: Optional[float] = None
    score: Optional[float] = None


def get_device():
    """사용 가능한 디바이스 반환"""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(model_path: str, device: str = None):
    """모델 로드"""
    if device is None:
        device = get_device()
    
    print(f"Loading model from: {model_path}")
    print(f"Device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    if device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        ).to(device)
    else:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
        except ValueError:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            ).to(device)
    
    model.eval()
    return model, tokenizer, device


def get_test_prompts() -> list[str]:
    """테스트 프롬프트 목록"""
    return [
        "Explain the concept of machine learning in simple terms.",
        "What are the benefits of renewable energy?",
        "Write a short poem about the ocean.",
        "Describe the process of photosynthesis.",
        "What is the capital of France and why is it famous?",
        "인공지능의 미래에 대해 설명해주세요.",
        "한국의 전통 음식 중 하나를 소개해주세요.",
        "프로그래밍을 배우는 좋은 방법은 무엇인가요?",
    ]


def evaluate_speed(model, tokenizer, device: str, 
                   prompts: list[str], max_new_tokens: int = 64) -> dict:
    """속도 평가: 토큰 생성 시간 측정"""
    total_tokens = 0
    total_time = 0.0
    
    # 워밍업 (첫 실행은 느릴 수 있음)
    warmup_input = tokenizer("Hello", return_tensors="pt").to(device)
    with torch.no_grad():
        _ = model.generate(**warmup_input, max_new_tokens=5)
    
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(device)
        
        # 시간 측정
        if device == "cuda":
            torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        if device == "cuda":
            torch.cuda.synchronize()
        
        elapsed = time.perf_counter() - start_time
        
        # 생성된 토큰 수 (입력 제외)
        new_tokens = output.shape[1] - input_ids.shape[1]
        total_tokens += new_tokens
        total_time += elapsed
    
    return {
        "total_time_sec": total_time,
        "total_tokens": total_tokens,
        "tokens_per_sec": total_tokens / total_time if total_time > 0 else 0,
        "time_per_token_ms": (total_time / total_tokens * 1000) if total_tokens > 0 else 0
    }


def evaluate_perplexity(model, tokenizer, device: str, 
                        texts: list[str] = None) -> float:
    """성능 평가: Perplexity 계산 (낮을수록 좋음)"""
    if texts is None:
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "인공지능은 컴퓨터 과학의 한 분야입니다.",
        ]
    
    total_loss = 0.0
    total_tokens = 0
    
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss.item()
        
        # 토큰 수 가중 평균
        num_tokens = inputs["input_ids"].shape[1]
        total_loss += loss * num_tokens
        total_tokens += num_tokens
    
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return perplexity


def calculate_score(result: EvalResult, baseline: EvalResult = None) -> EvalResult:
    """최종 점수 계산"""
    if baseline is None:
        # Baseline이면 기준값 설정
        result.perf_norm = 1.0
        result.speed_norm = 0.0  # 속도 개선 없음
        result.score = 0.5  # 0.5 * 1.0 + 0.5 * 0.0
    else:
        # PerfNorm: 성능 비율 (perplexity는 낮을수록 좋으므로 역수)
        # 실제로는 정확도 기반이지만, perplexity로 대체
        result.perf_norm = baseline.perplexity / result.perplexity if result.perplexity else 1.0
        
        # SpeedNorm: 속도 개선 비율
        base_time_per_token = baseline.time_per_token_ms
        model_time_per_token = result.time_per_token_ms
        result.speed_norm = 1 - (model_time_per_token / base_time_per_token) if base_time_per_token else 0.0
        
        # Score
        result.score = max(0.5 * result.perf_norm + 0.5 * result.speed_norm, 0)
    
    return result


def evaluate_model(model_path: str, 
                   experiment_name: str = "exaone_compression",
                   run_name: str = None,
                   baseline_result: EvalResult = None,
                   max_new_tokens: int = 64) -> EvalResult:
    """모델 종합 평가"""
    
    # 모델 로드
    model, tokenizer, device = load_model(model_path)
    
    # 모델 정보
    num_params = sum(p.numel() for p in model.parameters())
    num_layers = getattr(model.config, "num_hidden_layers", None)
    num_heads = getattr(model.config, "num_attention_heads", None)
    hidden_dim = getattr(model.config, "hidden_size", None)
    dtype = next(model.parameters()).dtype
    bytes_per_param = 2 if dtype in (torch.float16, torch.bfloat16) else 4
    model_size_mb = (num_params * bytes_per_param) / (1024 * 1024)
    
    # 테스ト 프롬프트
    prompts = get_test_prompts()
    
    print(f"\n{'='*60}")
    print(f"🔍 Evaluating: {model_path}")
    print(f"{'='*60}")
    print(f"  Parameters: {num_params:,} ({num_params/1e9:.2f}B)")
    print(f"  Device: {device}")
    print(f"  Test samples: {len(prompts)}")
    
    # 속도 평가
    print("\n⏱️  Speed evaluation...")
    speed_result = evaluate_speed(model, tokenizer, device, prompts, max_new_tokens)
    print(f"  Tokens/sec: {speed_result['tokens_per_sec']:.2f}")
    print(f"  Time/token: {speed_result['time_per_token_ms']:.2f} ms")
    
    # 성능 평가 (Perplexity)
    print("\n📊 Performance evaluation (Perplexity)...")
    perplexity = evaluate_perplexity(model, tokenizer, device)
    print(f"  Perplexity: {perplexity:.2f}")
    
    # 결과 생성
    result = EvalResult(
        model_name=model_path,
        num_samples=len(prompts),
        total_time_sec=speed_result["total_time_sec"],
        total_tokens=speed_result["total_tokens"],
        tokens_per_sec=speed_result["tokens_per_sec"],
        time_per_token_ms=speed_result["time_per_token_ms"],
        perplexity=perplexity
    )
    
    # 점수 계산
    result = calculate_score(result, baseline_result)
    
    print(f"\n🎯 Score:")
    print(f"  PerfNorm: {result.perf_norm:.4f}")
    print(f"  SpeedNorm: {result.speed_norm:.4f}")
    print(f"  Final Score: {result.score:.4f}")
    
    # MLflow 기록
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=run_name or Path(model_path).name):
        # 파라미터
        mlflow.log_param("model_path", model_path)
        mlflow.log_param("device", device)
        mlflow.log_param("num_params", num_params)
        mlflow.log_param("number_of_layers", num_layers)
        mlflow.log_param("number_of_heads", num_heads)
        mlflow.log_param("hidden_dim", hidden_dim)
        mlflow.log_param("total_parameters", num_params)
        mlflow.log_param("model_size_mb", model_size_mb)
        mlflow.log_param("inference_latency_ms", result.time_per_token_ms)
        mlflow.log_param("num_samples", len(prompts))
        mlflow.log_param("max_new_tokens", max_new_tokens)
        
        # 메트릭
        mlflow.log_metric("tokens_per_sec", result.tokens_per_sec)
        mlflow.log_metric("time_per_token_ms", result.time_per_token_ms)
        mlflow.log_metric("perplexity", result.perplexity)
        mlflow.log_metric("perf_norm", result.perf_norm)
        mlflow.log_metric("speed_norm", result.speed_norm)
        mlflow.log_metric("score", result.score)
        
        # 태그
        if baseline_result is None:
            mlflow.set_tag("model_type", "baseline")
            mlflow.set_tag("experiment_stage", "baseline")
        else:
            mlflow.set_tag("model_type", "compressed")
            mlflow.set_tag("experiment_stage", "compression")
        mlflow.set_tag("compression_type", "layer_drop" if "drop_layers" in Path(model_path).name else "unknown")
        mlflow.set_tag("variant_name", run_name or Path(model_path).name)
    
    print(f"\n✅ Results logged to MLflow (experiment: {experiment_name})")
    
    # 평가 결과를 모델 폴더에 JSON으로 저장 (보고서 자동화용)
    if Path(model_path).is_dir():
        eval_result_path = Path(model_path) / "eval_result.json"
        with open(eval_result_path, "w") as f:
            json.dump(asdict(result), f, indent=2)
        print(f"💾 Eval result saved to: {eval_result_path}")
    
    # 메모리 정리
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    
    return result


def save_baseline(result: EvalResult, path: str = "outputs/baseline_result.json"):
    """Baseline 결과 저장"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(asdict(result), f, indent=2)
    print(f"💾 Baseline saved to: {path}")


def load_baseline(path: str = "outputs/baseline_result.json") -> EvalResult:
    """Baseline 결과 로드"""
    with open(path, "r") as f:
        data = json.load(f)
    return EvalResult(**data)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate EXAONE model")
    parser.add_argument("--model", type=str, default="LGAI-EXAONE/EXAONE-4.0-1.2B",
                        help="Model path or HuggingFace name")
    parser.add_argument("--baseline", action="store_true",
                        help="Evaluate as baseline (no comparison)")
    parser.add_argument("--baseline-path", type=str, default="outputs/baseline_result.json",
                        help="Path to baseline result JSON")
    parser.add_argument("--run-name", type=str, default=None,
                        help="MLflow run name")
    parser.add_argument("--max-tokens", type=int, default=64,
                        help="Max new tokens to generate")
    
    args = parser.parse_args()
    
    # Baseline 로드 or None
    baseline_result = None
    if not args.baseline and Path(args.baseline_path).exists():
        print(f"📂 Loading baseline from: {args.baseline_path}")
        baseline_result = load_baseline(args.baseline_path)
    
    # 평가 실행
    result = evaluate_model(
        model_path=args.model,
        run_name=args.run_name or ("baseline" if args.baseline else None),
        baseline_result=baseline_result,
        max_new_tokens=args.max_tokens
    )
    
    # Baseline이면 저장
    if args.baseline:
        save_baseline(result, args.baseline_path)
