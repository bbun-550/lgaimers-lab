"""
Attention Head Pruning 경량화

가설: 일부 attention head는 redundant하므로 제거해도 성능 유지 가능
방법: 각 레이어에서 중요도가 낮은 attention head 제거

EXAONE-4.0-1.2B 구조:
- num_attention_heads: 32
- num_key_value_heads: 8 (GQA - Grouped Query Attention)
- head_dim: 64

주의: GQA 구조에서는 Q heads와 KV heads 비율을 유지해야 함
"""

import torch
import copy
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch.nn as nn


def get_device():
    """사용 가능한 디바이스 반환"""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_base_model(model_name: str = "LGAI-EXAONE/EXAONE-4.0-1.2B"):
    """베이스 모델 로드"""
    print(f"Loading base model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    return model, tokenizer, config


def get_attention_info(config):
    """모델 attention 구조 정보 추출"""
    info = {
        "num_attention_heads": getattr(config, "num_attention_heads", 32),
        "num_key_value_heads": getattr(config, "num_key_value_heads", 8),
        "head_dim": getattr(config, "head_dim", 64),
        "hidden_size": getattr(config, "hidden_size", 2048),
        "num_layers": getattr(config, "num_hidden_layers", 30),
    }
    
    # GQA ratio 계산
    info["gqa_ratio"] = info["num_attention_heads"] // info["num_key_value_heads"]
    
    print(f"\n📊 Attention Configuration:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    return info


def get_transformer_layers(model):
    """모델에서 transformer layers 모듈 찾기"""
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'layers'):
        return model.transformer, 'layers'
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model, 'layers'
    elif hasattr(model, 'model') and hasattr(model.model, 'decoder') and hasattr(model.model.decoder, 'layers'):
        return model.model.decoder, 'layers'
    else:
        for name, module in model.named_modules():
            if hasattr(module, 'layers') and len(list(module.layers)) > 0:
                return module, 'layers'
        raise ValueError("Could not find transformer layers in model")


def prune_linear_layer(layer: nn.Linear, keep_indices: list, dim: int = 0):
    """Linear 레이어에서 특정 인덱스만 유지"""
    weight = layer.weight.data
    bias = layer.bias.data if layer.bias is not None else None
    
    if dim == 0:  # output features
        new_weight = weight[keep_indices, :].clone()
        new_bias = bias[keep_indices].clone() if bias is not None else None
        new_out_features = len(keep_indices)
        new_in_features = weight.shape[1]
    else:  # input features (dim == 1)
        new_weight = weight[:, keep_indices].clone()
        new_bias = bias.clone() if bias is not None else None
        new_out_features = weight.shape[0]
        new_in_features = len(keep_indices)
    
    new_layer = nn.Linear(new_in_features, new_out_features, bias=bias is not None)
    new_layer.weight.data = new_weight
    if new_bias is not None:
        new_layer.bias.data = new_bias
    
    return new_layer


def prune_attention_heads(model, config, num_heads_to_keep: int, prune_kv: bool = True):
    """
    Attention Head Pruning
    
    Args:
        model: 원본 모델
        config: 모델 설정
        num_heads_to_keep: 유지할 Q head 수
        prune_kv: KV heads도 비율에 맞게 pruning할지 여부
    
    Returns:
        pruned_model, new_config
    """
    attn_info = get_attention_info(config)
    
    original_q_heads = attn_info["num_attention_heads"]
    original_kv_heads = attn_info["num_key_value_heads"]
    head_dim = attn_info["head_dim"]
    gqa_ratio = attn_info["gqa_ratio"]
    
    # 유지할 head 수 계산
    new_q_heads = num_heads_to_keep
    if prune_kv:
        # GQA 비율 유지하면서 KV heads도 감소
        if new_q_heads % gqa_ratio != 0:
            raise ValueError(
                f"num_heads_to_keep ({new_q_heads}) must be divisible by gqa_ratio ({gqa_ratio})"
            )
        new_kv_heads = max(1, new_q_heads // gqa_ratio)
    else:
        new_kv_heads = original_kv_heads
    
    print(f"\n🔧 Head Pruning Configuration:")
    print(f"  Q Heads: {original_q_heads} → {new_q_heads}")
    print(f"  KV Heads: {original_kv_heads} → {new_kv_heads}")
    print(f"  Head dim: {head_dim} (unchanged)")
    
    # 유지할 head indices (앞에서부터)
    q_head_indices = list(range(new_q_heads))
    kv_head_indices = list(range(new_kv_heads))
    
    # Q, K, V projection 크기 계산
    q_features = [h * head_dim + i for h in q_head_indices for i in range(head_dim)]
    kv_features = [h * head_dim + i for h in kv_head_indices for i in range(head_dim)]
    
    new_q_dim = new_q_heads * head_dim
    new_kv_dim = new_kv_heads * head_dim
    
    print(f"  Q projection: {original_q_heads * head_dim} → {new_q_dim}")
    print(f"  KV projection: {original_kv_heads * head_dim} → {new_kv_dim}")
    
    # Transformer layers 가져오기
    transformer, layers_attr = get_transformer_layers(model)
    layers = getattr(transformer, layers_attr)
    
    # 각 레이어의 attention 수정
    for layer_idx, layer in enumerate(layers):
        # Attention 모듈 찾기
        attn = None
        for name in ['self_attn', 'attention', 'attn']:
            if hasattr(layer, name):
                attn = getattr(layer, name)
                break
        
        if attn is None:
            print(f"  ⚠️ Layer {layer_idx}: Could not find attention module")
            continue
        
        # Q, K, V projection 찾기
        q_proj = getattr(attn, 'q_proj', None)
        k_proj = getattr(attn, 'k_proj', None)
        v_proj = getattr(attn, 'v_proj', None)
        o_proj = getattr(attn, 'o_proj', None)
        
        if q_proj is None:
            print(f"  ⚠️ Layer {layer_idx}: Could not find q_proj")
            continue
        
        # Q projection pruning
        attn.q_proj = prune_linear_layer(q_proj, q_features, dim=0)
        
        # K, V projection pruning (if applicable)
        if prune_kv and k_proj is not None:
            attn.k_proj = prune_linear_layer(k_proj, kv_features, dim=0)
        if prune_kv and v_proj is not None:
            attn.v_proj = prune_linear_layer(v_proj, kv_features, dim=0)
        
        # Output projection pruning (input dim matches Q heads)
        if o_proj is not None:
            attn.o_proj = prune_linear_layer(o_proj, q_features, dim=1)
        
        # Attention 모듈의 head 수 업데이트
        if hasattr(attn, 'num_heads'):
            attn.num_heads = new_q_heads
        if hasattr(attn, 'num_key_value_heads'):
            attn.num_key_value_heads = new_kv_heads
    
    # Config 업데이트
    new_config = copy.deepcopy(config)
    new_config.num_attention_heads = new_q_heads
    new_config.num_key_value_heads = new_kv_heads
    
    # 파라미터 수 계산
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"\n📊 After Pruning:")
    print(f"  Total parameters: {total_params:,}")
    
    return model, new_config


def prune_kv_heads(model, config, num_kv_heads_to_keep: int):
    """
    KV Head만 Pruning (Q heads 유지)
    
    Args:
        model: 원본 모델
        config: 모델 설정
        num_kv_heads_to_keep: 유지할 KV head 수
    
    Returns:
        pruned_model, new_config
    """
    attn_info = get_attention_info(config)
    
    original_q_heads = attn_info["num_attention_heads"]
    original_kv_heads = attn_info["num_key_value_heads"]
    head_dim = attn_info["head_dim"]
    
    new_kv_heads = num_kv_heads_to_keep
    new_gqa_ratio = original_q_heads // new_kv_heads
    
    print(f"\n🔧 KV-Only Pruning Configuration:")
    print(f"  Q Heads: {original_q_heads} (unchanged)")
    print(f"  KV Heads: {original_kv_heads} → {new_kv_heads}")
    print(f"  New GQA Ratio: {new_gqa_ratio}")
    print(f"  Head dim: {head_dim} (unchanged)")
    
    # KV head indices
    kv_head_indices = list(range(new_kv_heads))
    kv_features = [h * head_dim + i for h in kv_head_indices for i in range(head_dim)]
    
    new_kv_dim = new_kv_heads * head_dim
    print(f"  KV projection: {original_kv_heads * head_dim} → {new_kv_dim}")
    
    # Transformer layers 가져오기
    transformer, layers_attr = get_transformer_layers(model)
    layers = getattr(transformer, layers_attr)
    
    for layer_idx, layer in enumerate(layers):
        attn = None
        for name in ['self_attn', 'attention', 'attn']:
            if hasattr(layer, name):
                attn = getattr(layer, name)
                break
        
        if attn is None:
            continue
        
        k_proj = getattr(attn, 'k_proj', None)
        v_proj = getattr(attn, 'v_proj', None)
        
        # K, V projection만 pruning
        if k_proj is not None:
            attn.k_proj = prune_linear_layer(k_proj, kv_features, dim=0)
        if v_proj is not None:
            attn.v_proj = prune_linear_layer(v_proj, kv_features, dim=0)
        
        # Attention 모듈의 KV head 수 업데이트
        if hasattr(attn, 'num_key_value_heads'):
            attn.num_key_value_heads = new_kv_heads
    
    # Config 업데이트
    new_config = copy.deepcopy(config)
    new_config.num_key_value_heads = new_kv_heads
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n📊 After KV Pruning:")
    print(f"  Total parameters: {total_params:,}")
    
    return model, new_config


def save_compressed_model(model, tokenizer, config, save_path: str):
    """압축된 모델 저장"""
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving compressed model to: {save_path}")
    
    # 모델/Config 저장 (모델의 config를 최신으로 맞춘 뒤 저장)
    model.config = config
    config.save_pretrained(save_path)
    
    # 모델 가중치 저장
    model.save_pretrained(save_path, safe_serialization=True)
    
    # 토크나이저 저장
    tokenizer.save_pretrained(save_path)
    
    print("✅ Model saved successfully!")
    
    # 저장된 파일 목록
    files = list(save_path.glob("*"))
    print(f"\n📁 Saved files:")
    for f in files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  - {f.name} ({size_mb:.1f} MB)")


def test_generation(model, tokenizer, device: str, prompt: str = "Hello, how are you?"):
    """압축된 모델 테스트"""
    print(f"\n🧪 Testing generation...")
    
    model = model.to(device)
    model.eval()
    
    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"  Prompt: {prompt}")
    print(f"  Response: {response[:200]}...")
    
    return response


def create_head_pruned_model(
    model_name: str = "LGAI-EXAONE/EXAONE-4.0-1.2B",
    num_heads_to_keep: int = 24,
    prune_kv: bool = True,
    save_path: str = None,
    test: bool = True
):
    """
    Head Pruning 모델 생성 메인 함수
    
    Args:
        model_name: 베이스 모델
        num_heads_to_keep: 유지할 Q attention head 수 (기본 32 -> 24)
        prune_kv: KV heads도 비율에 맞게 pruning
        save_path: 저장 경로
        test: 생성 테스트 여부
    """
    print("=" * 60)
    print("✂️ Attention Head Pruning Compression")
    print("=" * 60)
    
    # 모델 로드
    model, tokenizer, config = load_base_model(model_name)
    
    original_heads = config.num_attention_heads
    device = get_device()
    
    # Head Pruning
    model, new_config = prune_attention_heads(
        model, config,
        num_heads_to_keep=num_heads_to_keep,
        prune_kv=prune_kv
    )
    
    # 테스트
    if test:
        test_generation(model, tokenizer, device)
    
    # 저장
    if save_path:
        save_compressed_model(model, tokenizer, new_config, save_path)
    
    print("\n" + "=" * 60)
    print(f"✅ Head Pruning Complete!")
    print(f"   {original_heads} heads → {num_heads_to_keep} heads")
    print("=" * 60)
    
    return model, tokenizer, new_config


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Attention Head Pruning Compression")
    parser.add_argument("--model", type=str, default="LGAI-EXAONE/EXAONE-4.0-1.2B",
                        help="Base model name")
    parser.add_argument("--keep-heads", type=int, default=24,
                        help="Number of Q heads to keep (default: 24, original: 32)")
    parser.add_argument("--no-prune-kv", action="store_true",
                        help="Don't prune KV heads (keep original)")
    parser.add_argument("--save-path", type=str, default=None,
                        help="Path to save compressed model")
    parser.add_argument("--no-test", action="store_true",
                        help="Skip generation test")
    
    args = parser.parse_args()
    
    create_head_pruned_model(
        model_name=args.model,
        num_heads_to_keep=args.keep_heads,
        prune_kv=not args.no_prune_kv,
        save_path=args.save_path,
        test=not args.no_test
    )
