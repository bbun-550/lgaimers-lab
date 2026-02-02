"""
Layer Dropping 경량화

가설: 상위 레이어들은 task-specific하므로 일부 제거해도 기본 성능 유지 가능
방법: Transformer의 상위 N개 레이어를 제거

EXAONE-4.0-1.2B: 30 layers → 26~28 layers로 축소
"""

import torch
import copy
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


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


def get_transformer_layers(model):
    """모델에서 transformer layers 모듈 찾기"""
    # EXAONE 모델 구조 탐색
    # 일반적인 구조: model.transformer.layers 또는 model.model.layers
    
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'layers'):
        return model.transformer, 'layers'
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model, 'layers'
    elif hasattr(model, 'model') and hasattr(model.model, 'decoder') and hasattr(model.model.decoder, 'layers'):
        return model.model.decoder, 'layers'
    else:
        # 구조 탐색
        for name, module in model.named_modules():
            if hasattr(module, 'layers') and len(list(module.layers)) > 0:
                return module, 'layers'
        raise ValueError("Could not find transformer layers in model")


def drop_layers(model, config, layers_to_drop: list[int] = None, 
                num_layers_to_keep: int = None, drop_from: str = "top"):
    """
    레이어 드롭핑
    
    Args:
        model: 원본 모델
        config: 모델 설정
        layers_to_drop: 제거할 레이어 인덱스 리스트
        num_layers_to_keep: 유지할 레이어 수 (layers_to_drop과 배타적)
        drop_from: 'top' (상위 레이어 제거) 또는 'bottom' (하위 레이어 제거)
    
    Returns:
        compressed_model, new_config
    """
    # 현재 레이어 수 확인
    transformer, layers_attr = get_transformer_layers(model)
    original_layers = getattr(transformer, layers_attr)
    num_original_layers = len(original_layers)
    
    print(f"\n🔧 Layer Dropping Configuration:")
    print(f"  Original layers: {num_original_layers}")
    
    # 제거할 레이어 결정
    if layers_to_drop is None and num_layers_to_keep is not None:
        num_to_drop = num_original_layers - num_layers_to_keep
        if drop_from == "top":
            # 상위 레이어 제거 (마지막 N개)
            layers_to_drop = list(range(num_original_layers - num_to_drop, num_original_layers))
        else:
            # 하위 레이어 제거 (처음 N개)
            layers_to_drop = list(range(num_to_drop))
    
    if layers_to_drop is None:
        layers_to_drop = []
    
    layers_to_keep = [i for i in range(num_original_layers) if i not in layers_to_drop]
    
    print(f"  Layers to drop: {layers_to_drop}")
    print(f"  Layers to keep: {layers_to_keep}")
    print(f"  New layer count: {len(layers_to_keep)}")
    
    # 새 레이어 리스트 생성
    new_layers = torch.nn.ModuleList([
        copy.deepcopy(original_layers[i]) for i in layers_to_keep
    ])
    
    # 레이어 교체
    setattr(transformer, layers_attr, new_layers)
    
    # Config 업데이트
    new_config = copy.deepcopy(config)
    new_config.num_hidden_layers = len(layers_to_keep)
    
    # 파라미터 수 계산
    original_params = sum(p.numel() for p in original_layers.parameters())
    new_params = sum(p.numel() for p in new_layers.parameters())
    reduction = (1 - new_params / original_params) * 100
    
    print(f"\n📊 Parameter Reduction:")
    print(f"  Original: {original_params:,}")
    print(f"  New: {new_params:,}")
    print(f"  Reduction: {reduction:.1f}%")
    
    return model, new_config


def save_compressed_model(model, tokenizer, config, save_path: str):
    """압축된 모델 저장"""
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving compressed model to: {save_path}")
    
    # Config 저장 (레이어 수 업데이트됨)
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


def create_layer_dropped_model(
    model_name: str = "LGAI-EXAONE/EXAONE-4.0-1.2B",
    num_layers_to_keep: int = 26,
    drop_from: str = "top",
    save_path: str = None,
    test: bool = True
):
    """
    Layer Dropping 모델 생성 메인 함수
    
    Args:
        model_name: 베이스 모델
        num_layers_to_keep: 유지할 레이어 수 (기본 30 -> 26)
        drop_from: 'top' 또는 'bottom'
        save_path: 저장 경로 (None이면 저장 안함)
        test: 생성 테스트 여부
    """
    print("=" * 60)
    print("🔪 Layer Dropping Compression")
    print("=" * 60)
    
    # 모델 로드
    model, tokenizer, config = load_base_model(model_name)
    
    original_layers = config.num_hidden_layers
    device = get_device()
    
    # 레이어 드롭
    model, new_config = drop_layers(
        model, config,
        num_layers_to_keep=num_layers_to_keep,
        drop_from=drop_from
    )
    
    # 테스트
    if test:
        test_generation(model, tokenizer, device)
    
    # 저장
    if save_path:
        save_compressed_model(model, tokenizer, new_config, save_path)
    
    print("\n" + "=" * 60)
    print(f"✅ Layer Dropping Complete!")
    print(f"   {original_layers} layers → {num_layers_to_keep} layers")
    print("=" * 60)
    
    return model, tokenizer, new_config


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Layer Dropping Compression")
    parser.add_argument("--model", type=str, default="LGAI-EXAONE/EXAONE-4.0-1.2B",
                        help="Base model name")
    parser.add_argument("--keep-layers", type=int, default=26,
                        help="Number of layers to keep (default: 26, original: 30)")
    parser.add_argument("--drop-from", type=str, default="top", choices=["top", "bottom"],
                        help="Drop layers from 'top' or 'bottom'")
    parser.add_argument("--save-path", type=str, default=None,
                        help="Path to save compressed model")
    parser.add_argument("--no-test", action="store_true",
                        help="Skip generation test")
    
    args = parser.parse_args()
    
    create_layer_dropped_model(
        model_name=args.model,
        num_layers_to_keep=args.keep_layers,
        drop_from=args.drop_from,
        save_path=args.save_path,
        test=not args.no_test
    )
