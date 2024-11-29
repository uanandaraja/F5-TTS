#!/bin/bash

uv venv --python 3.12

source .venv/bin/activate

uv pip install torch==2.4.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124

git submodule update --init --recursive

sed -i '7iimport sys\nsys.path.append(os.path.dirname(os.path.abspath(__file__)))' src/third_party/BigVGAN/bigvgan.py

uv pip install -e . --no-cache-dir
