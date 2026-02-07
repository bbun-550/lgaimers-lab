"""
Hydra 기반 모델 경량화 학습/실험 스크립트

Usage:
    # 단일 실험
    uv run python src/models/train.py experiments=02-04_drop_layers
    
    # Multirun (여러 실험 순차 실행)
    uv run python src/models/train.py -m experiments=exp1,exp2,exp3
    
    # CLI 오버라이드
    uv run python src/models/train.py compression.keep_layers=24 experiment_name=test
"""

import os
import sys
import json
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import torch

# 프로젝트 루트를 Python 경로에 추가
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.variants.drop_layers import (
    load_base_model,
    test_generation,
    drop_layers,
    save_compressed_model,
    get_device
)
from src.models.variants.prune_heads import prune_attention_heads, prune_kv_heads
from src.models.variants.quantize import quantize_model
from src.models.variants.reduce_ffn import reduce_ffn


def setup_mlflow(cfg: DictConfig):
    """MLflow 설정"""
    try:
        import mlflow
        from dotenv import load_dotenv
        
        load_dotenv()
        
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            print(f"📡 MLflow tracking: {tracking_uri}")
        
        mlflow.set_experiment("exaone_compression")
        return mlflow
    except ImportError:
        print("⚠️ MLflow not available, skipping logging")
        return None


def log_to_mlflow(mlflow, cfg: DictConfig, results: dict):
    """MLflow에 결과 기록"""
    if mlflow is None:
        return
    
    with mlflow.start_run(run_name=cfg.experiment_name):
        # 파라미터 기록
        compression_method = cfg.compression.get("method", "none")
        keep_layers = cfg.compression.get("keep_layers", "all")
        drop_from = cfg.compression.get("drop_from", "top")
        new_layers = results.get("new_layers")
        new_params = results.get("new_params")
        hidden_dim = cfg.model.get("hidden_dim")
        num_heads = results.get("new_heads", cfg.model.get("num_heads"))
        model_size_mb = None
        if isinstance(new_params, int):
            model_size_mb = (new_params * 2) / (1024 * 1024)

        mlflow.log_params({
            "experiment_name": cfg.experiment_name,
            "compression_method": compression_method,
            "keep_layers": keep_layers,
            "drop_from": drop_from,
            "number_of_layers": new_layers,
            "number_of_heads": num_heads,
            "hidden_dim": hidden_dim,
            "total_parameters": new_params,
            "model_size_mb": model_size_mb,
        })
        
        # 메트릭 기록
        for key, value in results.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value)
        
        # 태그
        mlflow.set_tags({
            "compression_type": compression_method,
            "variant_name": cfg.experiment_name,
            "experiment_stage": "compression"
        })
        
        print(f"✅ Results logged to MLflow: {cfg.experiment_name}")


def run_compression(cfg: DictConfig) -> dict:
    """경량화 실행"""
    print("=" * 60)
    print(f"🧪 Experiment: {cfg.experiment_name}")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    
    results = {}
    
    # compression method에 따라 분기
    method = cfg.compression.get("method")
    
    if method == "drop_layers":
        results = run_layer_dropping(cfg)
    elif method == "prune_heads":
        results = run_head_pruning(cfg)
    elif method == "drop_layers_prune_heads":
        results = run_layer_drop_then_prune_heads(cfg)
    elif method == "drop_layers_prune_heads_kv":
        results = run_layer_drop_prune_heads_kv(cfg)
    elif method == "quantization" or method == "quantize":
        results = run_quantization(cfg)
    elif method == "reduce_ffn":
        results = run_ffn_reduction(cfg)
    elif method == "reduce_hidden":
        print("⚠️ Hidden dim reduction not yet implemented via Hydra")
        results = {"status": "not_implemented"}
    elif method is None or method == "none":
        print("ℹ️ No compression method specified, running baseline")
        results = {"status": "baseline"}
    else:
        print(f"❌ Unknown compression method: {method}")
        results = {"status": "error"}
    
    return results


def run_layer_dropping(cfg: DictConfig) -> dict:
    """Layer Dropping 실행"""
    # 설정값 가져오기
    model_name = cfg.model.get("name", "LGAI-EXAONE/EXAONE-4.0-1.2B")
    if model_name == "exaone":
        model_name = "LGAI-EXAONE/EXAONE-4.0-1.2B"
    
    keep_layers = cfg.compression.get("keep_layers", 26)
    drop_from = cfg.compression.get("drop_from", "top")
    save_path = cfg.get("save_path", "./submit/model")
    
    print(f"\n🔧 Layer Dropping Configuration:")
    print(f"  Model: {model_name}")
    print(f"  Keep layers: {keep_layers}")
    print(f"  Drop from: {drop_from}")
    print(f"  Save path: {save_path}")
    
    # 모델 로드
    model, tokenizer, config = load_base_model(model_name)
    original_layers = config.num_hidden_layers
    original_params = sum(p.numel() for p in model.parameters())
    
    # 레이어 드롭
    model, new_config = drop_layers(
        model, config,
        num_layers_to_keep=keep_layers,
        drop_from=drop_from
    )
    
    new_params = sum(p.numel() for p in model.parameters())
    param_reduction = (1 - new_params / original_params) * 100
    
    # 테스트
    device = get_device()
    test_generation(model, tokenizer, device)
    
    # 저장
    save_compressed_model(model, tokenizer, new_config, save_path)
    
    results = {
        "original_layers": original_layers,
        "new_layers": keep_layers,
        "original_params": original_params,
        "new_params": new_params,
        "param_reduction_percent": param_reduction,
        "status": "success"
    }
    
    print(f"\n📊 Results:")
    print(f"  Layers: {original_layers} → {keep_layers}")
    print(f"  Params: {original_params:,} → {new_params:,}")
    print(f"  Reduction: {param_reduction:.1f}%")
    
    return results


def run_head_pruning(cfg: DictConfig) -> dict:
    """Head Pruning 실행"""
    # 설정값
    model_name = cfg.model.get("name", "LGAI-EXAONE/EXAONE-4.0-1.2B")
    if model_name == "exaone":
        model_name = "LGAI-EXAONE/EXAONE-4.0-1.2B"
        
    keep_heads = cfg.compression.get("keep_heads", 24)
    save_path = cfg.get("save_path", "./submit/model")
    # prune_kv 기본값 True
    prune_kv = cfg.compression.get("prune_kv", True)
    
    print(f"\n✂️ Head Pruning Configuration:")
    print(f"  Model: {model_name}")
    print(f"  Keep Q heads: {keep_heads}")
    print(f"  Prune KV heads: {prune_kv}")
    print(f"  Save path: {save_path}")
    
    # 모델 로드
    model, tokenizer, config = load_base_model(model_name)
    original_heads = config.num_attention_heads
    original_params = sum(p.numel() for p in model.parameters())
    
    # Pruning 실행
    model, new_config = prune_attention_heads(
        model, config,
        num_heads_to_keep=keep_heads,
        prune_kv=prune_kv
    )
    
    new_params = sum(p.numel() for p in model.parameters())
    param_reduction = (1 - new_params / original_params) * 100
    
    # 테스트
    device = get_device()
    test_generation(model, tokenizer, device)
    
    # 저장
    save_compressed_model(model, tokenizer, new_config, save_path)
    
    results = {
        "original_heads": original_heads,
        "new_heads": keep_heads,
        "original_params": original_params,
        "new_params": new_params,
        "param_reduction_percent": param_reduction,
        "status": "success"
    }
    
    print(f"\n📊 Results:")
    print(f"  Heads: {original_heads} → {keep_heads}")
    print(f"  Params: {original_params:,} → {new_params:,}")
    print(f"  Reduction: {param_reduction:.1f}%")
    
    return results


def run_layer_drop_then_prune_heads(cfg: DictConfig) -> dict:
    """Layer Dropping 후 Head Pruning 조합 실행 (최종 실험용)"""
    model_name = cfg.model.get("name", "LGAI-EXAONE/EXAONE-4.0-1.2B")
    if model_name == "exaone":
        model_name = "LGAI-EXAONE/EXAONE-4.0-1.2B"

    keep_layers = cfg.compression.get("keep_layers", 28)
    drop_from = cfg.compression.get("drop_from", "top")
    keep_heads = cfg.compression.get("keep_heads", 28)
    prune_kv = cfg.compression.get("prune_kv", True)
    save_path = cfg.get("save_path", "./submit/model")

    print(f"\n🔧 Combined Configuration:")
    print(f"  Model: {model_name}")
    print(f"  Keep layers: {keep_layers} (drop from {drop_from})")
    print(f"  Keep Q heads: {keep_heads} (prune KV: {prune_kv})")
    print(f"  Save path: {save_path}")

    # 모델 로드
    model, tokenizer, config = load_base_model(model_name)
    original_layers = config.num_hidden_layers
    original_heads = config.num_attention_heads
    original_params = sum(p.numel() for p in model.parameters())

    # Layer dropping
    model, config = drop_layers(
        model, config,
        num_layers_to_keep=keep_layers,
        drop_from=drop_from
    )

    # Head pruning
    model, config = prune_attention_heads(
        model, config,
        num_heads_to_keep=keep_heads,
        prune_kv=prune_kv
    )

    # 테스트
    device = get_device()
    test_generation(model, tokenizer, device)

    # 저장
    save_compressed_model(model, tokenizer, config, save_path)

    new_params = sum(p.numel() for p in model.parameters())
    param_reduction = (1 - new_params / original_params) * 100

    results = {
        "original_layers": original_layers,
        "new_layers": keep_layers,
        "original_heads": original_heads,
        "new_heads": keep_heads,
        "original_params": original_params,
        "new_params": new_params,
        "param_reduction_percent": param_reduction,
        "status": "success"
    }

    print(f"\n📊 Results:")
    print(f"  Layers: {original_layers} → {keep_layers}")
    print(f"  Heads: {original_heads} → {keep_heads}")
    print(f"  Params: {original_params:,} → {new_params:,}")
    print(f"  Reduction: {param_reduction:.1f}%")

    return results


def run_layer_drop_prune_heads_kv(cfg: DictConfig) -> dict:
    """Layer Dropping + Head Pruning + KV Pruning 조합 실행"""
    model_name = cfg.model.get("name", "LGAI-EXAONE/EXAONE-4.0-1.2B")
    if model_name == "exaone":
        model_name = "LGAI-EXAONE/EXAONE-4.0-1.2B"

    keep_layers = cfg.compression.get("keep_layers", 28)
    drop_from = cfg.compression.get("drop_from", "top")
    keep_heads = cfg.compression.get("keep_heads", 28)
    keep_kv_heads = cfg.compression.get("keep_kv_heads", 4)
    prune_kv = cfg.compression.get("prune_kv", False)  # Q/KV 동시 pruning 여부
    save_path = cfg.get("save_path", "./submit/model")

    print(f"\n🔧 Combined Configuration (with KV Pruning):")
    print(f"  Model: {model_name}")
    print(f"  Keep layers: {keep_layers} (drop from {drop_from})")
    print(f"  Keep Q heads: {keep_heads}")
    print(f"  Keep KV heads: {keep_kv_heads}")
    print(f"  Save path: {save_path}")

    # 모델 로드
    model, tokenizer, config = load_base_model(model_name)
    original_layers = config.num_hidden_layers
    original_heads = config.num_attention_heads
    original_kv_heads = config.num_key_value_heads
    original_params = sum(p.numel() for p in model.parameters())

    # 1. Layer dropping
    model, config = drop_layers(
        model, config,
        num_layers_to_keep=keep_layers,
        drop_from=drop_from
    )

    # 2. Head pruning (Q heads만, prune_kv=False로 KV 유지)
    model, config = prune_attention_heads(
        model, config,
        num_heads_to_keep=keep_heads,
        prune_kv=prune_kv  # False면 KV는 원래대로 유지
    )

    # 3. KV-only pruning
    model, config = prune_kv_heads(
        model, config,
        num_kv_heads_to_keep=keep_kv_heads
    )

    # 테스트
    device = get_device()
    test_generation(model, tokenizer, device)

    # 저장
    save_compressed_model(model, tokenizer, config, save_path)

    new_params = sum(p.numel() for p in model.parameters())
    param_reduction = (1 - new_params / original_params) * 100

    results = {
        "original_layers": original_layers,
        "new_layers": keep_layers,
        "original_heads": original_heads,
        "new_heads": keep_heads,
        "original_kv_heads": original_kv_heads,
        "new_kv_heads": keep_kv_heads,
        "original_params": original_params,
        "new_params": new_params,
        "param_reduction_percent": param_reduction,
        "status": "success"
    }

    print(f"\n📊 Results:")
    print(f"  Layers: {original_layers} → {keep_layers}")
    print(f"  Q Heads: {original_heads} → {keep_heads}")
    print(f"  KV Heads: {original_kv_heads} → {keep_kv_heads}")
    print(f"  Params: {original_params:,} → {new_params:,}")
    print(f"  Reduction: {param_reduction:.1f}%")

    return results


def run_quantization(cfg: DictConfig) -> dict:
    """Quantization 실행"""
    # 설정값
    model_name = cfg.model.get("name", "LGAI-EXAONE/EXAONE-4.0-1.2B")
    if model_name == "exaone":
        model_name = "LGAI-EXAONE/EXAONE-4.0-1.2B"
    
    dtype = cfg.compression.get("dtype", "float16")
    save_path = cfg.get("save_path", "./submit/model")
    
    print(f"\n📉 Quantization Configuration:")
    print(f"  Model: {model_name}")
    print(f"  Target Dtype: {dtype}")
    print(f"  Save path: {save_path}")
    
    # Quantization 실행
    model, tokenizer = quantize_model(
        model_name=model_name,
        dtype=dtype,
        save_path=save_path,
        test=True
    )
    
    # 결과 정보
    total_params = sum(p.numel() for p in model.parameters())
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    
    results = {
        "dtype": dtype,
        "total_parameters": total_params,
        "model_size_mb": model_size_mb,
        "status": "success"
    }
    
    return results


def run_ffn_reduction(cfg: DictConfig) -> dict:
    """FFN 축소 실행"""
    from src.models.variants.reduce_ffn import (
        load_base_model as load_model_ffn,
        reduce_ffn as do_reduce_ffn,
        save_compressed_model as save_model_ffn,
        test_generation as test_gen_ffn,
        get_device as get_dev_ffn
    )
    
    model_name = cfg.model.get("name", "LGAI-EXAONE/EXAONE-4.0-1.2B")
    if model_name == "exaone":
        model_name = "LGAI-EXAONE/EXAONE-4.0-1.2B"
    
    target_intermediate = cfg.compression.get("target_intermediate_size", 3072)
    save_path = cfg.get("save_path", "./submit/model")
    
    print(f"\n📉 FFN Reduction Configuration:")
    print(f"  Model: {model_name}")
    print(f"  Target intermediate_size: {target_intermediate}")
    print(f"  Save path: {save_path}")
    
    # 모델 로드
    model, tokenizer, config = load_model_ffn(model_name)
    original_intermediate = config.intermediate_size
    original_params = sum(p.numel() for p in model.parameters())
    
    # FFN 축소
    model, new_config = do_reduce_ffn(
        model, config,
        target_intermediate_size=target_intermediate
    )
    
    new_params = sum(p.numel() for p in model.parameters())
    param_reduction = (1 - new_params / original_params) * 100
    
    # 테스트
    device = get_dev_ffn()
    test_gen_ffn(model, tokenizer, device)
    
    # 저장
    save_model_ffn(model, tokenizer, new_config, save_path)
    
    results = {
        "original_intermediate_size": original_intermediate,
        "new_intermediate_size": target_intermediate,
        "original_params": original_params,
        "new_params": new_params,
        "param_reduction_percent": param_reduction,
        "status": "success"
    }
    
    print(f"\n📊 Results:")
    print(f"  Intermediate: {original_intermediate} → {target_intermediate}")
    print(f"  Params: {original_params:,} → {new_params:,}")
    print(f"  Reduction: {param_reduction:.1f}%")
    
    return results


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """메인 함수"""
    print("\n" + "=" * 60)
    print("🚀 EXAONE Model Compression with Hydra")
    print("=" * 60)
    
    # MLflow 설정
    mlflow = setup_mlflow(cfg)
    
    # 경량화 실행
    results = run_compression(cfg)
    
    # 결과 저장
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    results_file = output_dir / "results.json"
    
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to: {results_file}")
    
    # MLflow 기록
    if results.get("status") == "success":
        log_to_mlflow(mlflow, cfg, results)
    
    print("\n" + "=" * 60)
    print("✅ Experiment Complete!")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    main()
