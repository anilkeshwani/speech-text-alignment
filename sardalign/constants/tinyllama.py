TINYLLAMA_HF_REPO: str = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
TINYLLAMA_HF_REVISION: str = "036fa4651240b9a1487f709833b9e4b96b4c1574"
SENTENCEPIECE_TOKENIZER_FILENAME: str = "tokenizer.model"
TINYLLAMA_HF_TOKENIZER_FILES: tuple[str, ...] = (
    "special_tokens_map.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "tokenizer.model",
)
TINYLLAMA_HF_MODEL_FILES: tuple[str, ...] = (
    "config.json",
    "generation_config.json",
    "model.safetensors",
)
TINYLLAMA_HF_AUXILIARY_FILES: tuple[str, ...] = (
    ".gitattributes",
    "README.md",
    "pytorch_model.bin",
)
