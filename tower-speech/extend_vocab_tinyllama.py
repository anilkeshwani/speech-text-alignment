#!/usr/bin/env python

import logging
import os
import sys
from pathlib import Path

from sardalign.config import LOG_DATEFMT, LOG_FORMAT, LOG_LEVEL
from sardalign.constants import HUBERT_TOKEN_FSTRING, MODALITY_TOKEN_SPEECH, MODALITY_TOKEN_TEXT
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.llama.tokenization_llama import LlamaTokenizer


logging.basicConfig(
    format=LOG_FORMAT,
    datefmt=LOG_DATEFMT,
    level=os.environ.get("LOGLEVEL", LOG_LEVEL).upper(),
    stream=sys.stdout,
)

LOGGER = logging.getLogger(__file__)

HF_DIR = Path("/mnt/scratch-artemis/anilkeshwani/huggingface")
MODEL = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"

tokenizer: LlamaTokenizer = LlamaTokenizer.from_pretrained(MODEL, use_fast=False)
model: LlamaForCausalLM = LlamaForCausalLM.from_pretrained(MODEL)

len_tokenizer_orig = len(tokenizer)

additional_hubert_tokens: list[str] = [HUBERT_TOKEN_FSTRING.format(i) for i in range(5000)]
additional_modality_tokens: list[str] = [MODALITY_TOKEN_SPEECH, MODALITY_TOKEN_TEXT]
additional_special_tokens: list[str] = additional_hubert_tokens + additional_modality_tokens

tokenizer.add_tokens(additional_special_tokens)

LOGGER.info(f"Extended tokenizer length from {len_tokenizer_orig} to {len(tokenizer)} tokens")

extended_tokenizer_path = (
    HF_DIR
    / "tower"
    / "TinyLlama"
    / "extended-5000-clusters-modality-tokens"
    / "TinyLlama-1.1B-intermediate-step-1431k-3T"
)

tokenizer.save_pretrained(extended_tokenizer_path)

llama_tokenizer_from_disk = LlamaTokenizer.from_pretrained(extended_tokenizer_path)

LOGGER.info(f"Extended tokenizer from disk length {len(llama_tokenizer_from_disk)} tokens")
