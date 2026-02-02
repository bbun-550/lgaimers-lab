#!/bin/bash
# submit.zip 생성 스크립트

cd "$(dirname "$0")"

# model 폴더 확인
if [ ! -d "submit/model" ]; then
    echo "❌ Error: submit/model 폴더가 없습니다!"
    exit 1
fi

# model 폴더가 비어있는지 확인 (README.md만 있는 경우)
MODEL_FILES=$(find submit/model -type f ! -name "README.md" ! -name ".gitkeep" | wc -l)
if [ "$MODEL_FILES" -eq 0 ]; then
    echo "⚠️  Warning: submit/model 폴더에 모델 파일이 없습니다!"
    echo "   경량화된 모델을 먼저 저장하세요:"
    echo "   model.save_pretrained('./submit/model')"
    echo "   tokenizer.save_pretrained('./submit/model')"
    exit 1
fi

# 기존 zip 삭제
rm -f submit.zip

# submit/model 폴더 내용을 model/ 경로로 압축
# 결과: submit.zip/model/config.json, model.safetensors, ...
cd submit
zip -r ../submit.zip model -x "*.DS_Store" -x "__pycache__/*" -x "*.pyc" -x "README.md"
cd ..

echo ""
echo "✅ submit.zip 생성 완료!"
echo ""
echo "📦 내용물:"
unzip -l submit.zip
