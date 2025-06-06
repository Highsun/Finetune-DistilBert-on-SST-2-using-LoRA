#!/bin/bash

pip install --upgrade \
    transformers \
    datasets \
    peft \
    accelerate \
    evaluate \
    fsspec \
    huggingface_hub

rm -rf ~/.cache/huggingface/datasets
rm -rf ~/.cache/huggingface/hub

echo "Setup complete."