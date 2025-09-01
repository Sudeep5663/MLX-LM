MLX-LM Fine-Tuning

-- Introduction
    MLX-LM is Apple’s Machine Learning eXperimental framework for running and fine-tuning LLMs (Large Language Models) directly on Apple Silicon (M1/M2/M3).
    This explains how to fine-tune models (e.g., LLaMA, Mistral) using LoRA (Low-Rank Adaptation) with MLX-LM.

-- Prerequisites
    macOS with Apple Silicon (M1/M2/M3).
    Python ≥ 3.10

-- Install MLX-LM
    python -m venv .venv && source .venv/bin/activate
    pip install -U mlx-lm
-- Convert HF model → MLX (4-bit)
    mlx_lm.convert --hf-path <HF_REPO> -q # -q makes a 4-bit MLX model so you train LoRA on a quantized base.
                                          # saves to ./mlx_model by default (4-bit)
    
-- Fine-tune
    python -m mlx_lm.lora \
  --train \
  --model ./mlx_model \
  --data ./data/ \ # A data of both training and validation set
  --iters 1000 \
  --batch-size 1 \
  --save-every 100 \
  --out ./fine_tuned_model

-- Running the Fine-Tuned Model
    mlx_lm.chat --model ./fine_tuned_model

-- References
    https://github.com/ml-explore/mlx-examples?utm_source=chatgpt.com
    https://ml-explore.github.io/mlx/build/html/index.html#
    https://medium.com/@levchevajoana/fine-tuning-llms-with-lora-and-mlx-lm-c0b143642deb
