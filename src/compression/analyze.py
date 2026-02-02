"""
EXAONE 모델 구조 분석

분석 항목:
1. 모델 아키텍처 구조 (레이어, 헤드, 차원)
2. 파라미터 수 (총, 레이어별, 컴포넌트별)
3. 메모리 사용량 예측
4. 경량화 대상 분석
"""

import torch
from transformers import AutoModelForCausalLM, AutoConfig
from collections import defaultdict
import json


def get_device():
    """사용 가능한 디바이스 반환"""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def count_parameters(model) -> dict:
    """모델 파라미터 수 계산"""
    total_params = 0
    trainable_params = 0
    
    layer_params = defaultdict(int)
    component_params = defaultdict(int)
    
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        
        if param.requires_grad:
            trainable_params += num_params
        
        # 레이어별 분류
        parts = name.split(".")
        if "layers" in name:
            # transformer.layers.0.xxx -> layer_0
            layer_idx = None
            for i, part in enumerate(parts):
                if part == "layers" and i + 1 < len(parts):
                    layer_idx = parts[i + 1]
                    break
            if layer_idx:
                layer_params[f"layer_{layer_idx}"] += num_params
        
        # 컴포넌트별 분류
        if "embed" in name.lower():
            component_params["embedding"] += num_params
        elif "attn" in name.lower() or "attention" in name.lower():
            component_params["attention"] += num_params
        elif "mlp" in name.lower() or "ffn" in name.lower() or "fc" in name.lower():
            component_params["mlp/ffn"] += num_params
        elif "norm" in name.lower() or "ln" in name.lower():
            component_params["layer_norm"] += num_params
        elif "lm_head" in name.lower():
            component_params["lm_head"] += num_params
        else:
            component_params["other"] += num_params
    
    return {
        "total": total_params,
        "trainable": trainable_params,
        "by_layer": dict(sorted(layer_params.items(), key=lambda x: int(x[0].split("_")[1]))),
        "by_component": dict(component_params)
    }


def analyze_architecture(config) -> dict:
    """모델 아키텍처 분석"""
    # EXAONE 모델의 설정 키 매핑 (모델마다 다를 수 있음)
    arch_info = {
        "model_type": getattr(config, "model_type", "unknown"),
        "hidden_size": getattr(config, "hidden_size", None),
        "intermediate_size": getattr(config, "intermediate_size", None),
        "num_hidden_layers": getattr(config, "num_hidden_layers", None),
        "num_attention_heads": getattr(config, "num_attention_heads", None),
        "num_key_value_heads": getattr(config, "num_key_value_heads", None),  # GQA
        "vocab_size": getattr(config, "vocab_size", None),
        "max_position_embeddings": getattr(config, "max_position_embeddings", None),
        "rope_theta": getattr(config, "rope_theta", None),
    }
    
    # Head dimension 계산
    if arch_info["hidden_size"] and arch_info["num_attention_heads"]:
        arch_info["head_dim"] = arch_info["hidden_size"] // arch_info["num_attention_heads"]
    
    return arch_info


def estimate_memory(param_count: int, dtype: str = "bfloat16") -> dict:
    """메모리 사용량 추정"""
    bytes_per_param = {
        "float32": 4,
        "float16": 2,
        "bfloat16": 2,
        "int8": 1,
        "int4": 0.5
    }
    
    bpp = bytes_per_param.get(dtype, 2)
    model_size_bytes = param_count * bpp
    
    return {
        "dtype": dtype,
        "model_size_mb": model_size_bytes / (1024 ** 2),
        "model_size_gb": model_size_bytes / (1024 ** 3),
        "estimated_inference_gb": model_size_bytes / (1024 ** 3) * 1.2,  # +20% for activations
    }


def analyze_layers(model) -> list:
    """각 레이어 상세 분석"""
    layers_info = []
    
    # 모델 구조에서 transformer layers 찾기
    transformer = None
    for name, module in model.named_modules():
        if hasattr(module, 'layers') and len(list(module.layers)) > 0:
            transformer = module
            break
    
    if transformer is None:
        return layers_info
    
    for i, layer in enumerate(transformer.layers):
        layer_info = {
            "index": i,
            "params": sum(p.numel() for p in layer.parameters()),
            "components": {}
        }
        
        for name, submodule in layer.named_children():
            submodule_params = sum(p.numel() for p in submodule.parameters())
            layer_info["components"][name] = submodule_params
        
        layers_info.append(layer_info)
    
    return layers_info


def print_analysis(analysis: dict):
    """분석 결과 출력"""
    print("\n" + "=" * 70)
    print("🔍 EXAONE Model Analysis Report")
    print("=" * 70)
    
    # 아키텍처 정보
    print("\n📐 Architecture:")
    arch = analysis["architecture"]
    print(f"  • Model Type: {arch['model_type']}")
    print(f"  • Hidden Size: {arch['hidden_size']:,}")
    print(f"  • Intermediate Size (FFN): {arch['intermediate_size']:,}")
    print(f"  • Num Layers: {arch['num_hidden_layers']}")
    print(f"  • Num Attention Heads: {arch['num_attention_heads']}")
    print(f"  • Num KV Heads (GQA): {arch.get('num_key_value_heads', 'N/A')}")
    print(f"  • Head Dimension: {arch.get('head_dim', 'N/A')}")
    print(f"  • Vocab Size: {arch['vocab_size']:,}")
    
    # 파라미터 정보
    print("\n📊 Parameters:")
    params = analysis["parameters"]
    print(f"  • Total: {params['total']:,} ({params['total']/1e9:.2f}B)")
    print(f"  • Trainable: {params['trainable']:,}")
    
    print("\n  📦 By Component:")
    for comp, count in params["by_component"].items():
        pct = count / params["total"] * 100
        print(f"    - {comp}: {count:,} ({pct:.1f}%)")
    
    # 레이어별 파라미터 (처음 3개, 마지막 3개만)
    print("\n  📚 By Layer (sample):")
    layers = list(params["by_layer"].items())
    for layer, count in layers[:3]:
        print(f"    - {layer}: {count:,}")
    if len(layers) > 6:
        print(f"    ... ({len(layers) - 6} layers omitted)")
    for layer, count in layers[-3:]:
        print(f"    - {layer}: {count:,}")
    
    # 메모리 예측
    print("\n💾 Memory Estimation:")
    mem = analysis["memory"]
    print(f"  • Dtype: {mem['dtype']}")
    print(f"  • Model Size: {mem['model_size_gb']:.2f} GB")
    print(f"  • Est. Inference: {mem['estimated_inference_gb']:.2f} GB")
    
    # 경량화 추천
    print("\n🎯 Compression Recommendations:")
    arch = analysis["architecture"]
    
    # 레이어 드롭 추천
    num_layers = arch["num_hidden_layers"]
    print(f"  1. Layer Dropping:")
    print(f"     - Current: {num_layers} layers")
    print(f"     - Suggested: Drop upper 2-4 layers → {num_layers-2}~{num_layers-4} layers")
    print(f"     - Expected reduction: ~{100 * 2/num_layers:.1f}% ~ {100 * 4/num_layers:.1f}%")
    
    # Head pruning 추천
    num_heads = arch["num_attention_heads"]
    print(f"  2. Head Pruning:")
    print(f"     - Current: {num_heads} heads")
    print(f"     - Suggested: Prune 25-50% → {int(num_heads*0.75)}~{num_heads//2} heads")
    
    # FFN 축소 추천
    intermediate = arch["intermediate_size"]
    hidden = arch["hidden_size"]
    print(f"  3. FFN Reduction:")
    print(f"     - Current ratio: {intermediate/hidden:.1f}x hidden")
    print(f"     - Suggested: Reduce to 2.5x-3x → {int(hidden*2.5):,}~{hidden*3:,}")
    
    print("\n" + "=" * 70)


def analyze_model(model_name: str = "LGAI-EXAONE/EXAONE-4.0-1.2B", 
                  load_model: bool = True,
                  save_path: str = None) -> dict:
    """
    모델 분석 메인 함수
    
    Args:
        model_name: HuggingFace 모델 이름
        load_model: True면 전체 모델 로드, False면 config만 로드
        save_path: 분석 결과 저장 경로 (JSON)
    """
    print(f"\n🔄 Analyzing: {model_name}")
    
    # Config 로드
    print("  Loading config...")
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    
    # 아키텍처 분석
    architecture = analyze_architecture(config)
    
    if load_model:
        print(f"  Loading model...")
        device = get_device()
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        
        # 파라미터 분석
        parameters = count_parameters(model)
        
        # 레이어 분석
        layers = analyze_layers(model)
        
        # 메모리 추정
        memory = estimate_memory(parameters["total"], "bfloat16")
        
        # 메모리 정리
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    else:
        # Config만으로 추정
        # 대략적인 파라미터 수 추정
        hidden = architecture["hidden_size"]
        inter = architecture["intermediate_size"]
        layers_count = architecture["num_hidden_layers"]
        vocab = architecture["vocab_size"]
        
        # 추정: embedding + layers * (attn + ffn + norms) + lm_head
        estimated_params = (
            vocab * hidden +  # embedding
            layers_count * (
                4 * hidden * hidden +  # attention (Q, K, V, O)
                3 * hidden * inter +   # FFN (up, gate, down)
                4 * hidden             # layer norms
            ) +
            vocab * hidden  # lm_head (tied 아닐 경우)
        )
        
        parameters = {
            "total": estimated_params,
            "trainable": estimated_params,
            "by_layer": {},
            "by_component": {"estimated": estimated_params}
        }
        layers = []
        memory = estimate_memory(estimated_params, "bfloat16")
    
    analysis = {
        "model_name": model_name,
        "architecture": architecture,
        "parameters": parameters,
        "layers": layers,
        "memory": memory
    }
    
    # 결과 출력
    print_analysis(analysis)
    
    # 결과 저장
    if save_path:
        with open(save_path, "w") as f:
            json.dump(analysis, f, indent=2)
        print(f"\n💾 Analysis saved to: {save_path}")
    
    return analysis


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze EXAONE model structure")
    parser.add_argument("--model", type=str, default="LGAI-EXAONE/EXAONE-4.0-1.2B",
                        help="Model name or path")
    parser.add_argument("--config-only", action="store_true",
                        help="Only load config (faster, less accurate)")
    parser.add_argument("--save", type=str, default=None,
                        help="Path to save analysis JSON")
    
    args = parser.parse_args()
    
    analyze_model(
        model_name=args.model,
        load_model=not args.config_only,
        save_path=args.save
    )
