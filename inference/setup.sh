#!/bin/bash
# Setup script for YuE SGLang inference
# Run this on RunPod/Colab before running inference

set -e

echo "=== YuE SGLang Setup Script ==="

# Install git-lfs if not present
if ! command -v git-lfs &> /dev/null; then
    echo "Installing git-lfs..."
    apt-get update && apt-get install -y git-lfs
fi
git lfs install

# Navigate to inference directory
cd "$(dirname "$0")"

# Clone xcodec_mini_infer if not present or empty
if [ ! -d "xcodec_mini_infer/models" ]; then
    echo "Cloning xcodec_mini_infer from HuggingFace..."
    rm -rf xcodec_mini_infer
    git clone https://huggingface.co/m-a-p/xcodec_mini_infer
fi

# Fix hardcoded path in soundstream_hubert_new.py
SOUNDSTREAM_FILE="xcodec_mini_infer/models/soundstream_hubert_new.py"
if [ -f "$SOUNDSTREAM_FILE" ]; then
    echo "Fixing paths in soundstream_hubert_new.py..."
    
    # Add import os if not present
    if ! grep -q "^import os" "$SOUNDSTREAM_FILE"; then
        sed -i '1s/^/import os\n/' "$SOUNDSTREAM_FILE"
    fi
    
    # Fix the hardcoded path
    sed -i 's|"./xcodec_mini_infer/semantic_ckpts/hf_1_325000"|os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "semantic_ckpts", "hf_1_325000")|g' "$SOUNDSTREAM_FILE"
    
    echo "Path fix applied!"
fi

echo "=== Setup Complete ==="
echo "You can now run: python infer_sglang.py --genre_txt <genre.txt> --lyrics_txt <lyrics.txt>"
