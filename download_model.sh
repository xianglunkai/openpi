pip install huggingface_hub
export HF_ENDPOINT=https://hf-mirror.com

huggingface-cli download leo009/paligemma_tokenizer.model --local-dir ./models --local-dir-use-symlinks False --resume-download 
