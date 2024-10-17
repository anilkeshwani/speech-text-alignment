#!/usr/bin/env python

import logging
import os
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
from sentencepiece import sentencepiece_model_pb2
from transformers import AutoModelForCausalLM, PreTrainedModel
from transformers.configuration_utils import PretrainedConfig

from sardalign.config import LOG_DATEFMT, LOG_FORMAT, LOG_LEVEL
from sardalign.constants import MODALITY_TOKEN_SPEECH, MODALITY_TOKEN_TEXT, SEED
from sardalign.constants.megatron import MAKE_VOCAB_SIZE_DIVISIBLE_BY
from sardalign.constants.tinyllama import SENTENCEPIECE_TOKENIZER_FILENAME
from sardalign.utils import dsu2pua, multivariate_normal_from_weights, seed_everything


# NOTE Script structured to be trivially extensible for use of other base models in future

logging.basicConfig(
    format=LOG_FORMAT,
    datefmt=LOG_DATEFMT,
    level=os.environ.get("LOGLEVEL", LOG_LEVEL).upper(),
    stream=sys.stdout,
)

LOGGER = logging.getLogger(__file__)


def get_hf_config_embedding_size(config: PretrainedConfig) -> int:
    """
    Extract embedding size from a Hugging Face config.

    Args:
        config (PretrainedConfig): Hugging Face config.

    Returns:
        int: Embedding size from the config. Uses `embedding_size` attribute if present (e.g. ALBERT) else `hidden_size`
    """
    if not isinstance(config, PretrainedConfig):
        raise TypeError(f"Expected PretrainedConfig, got {type(config)}")
    if hasattr(config, "embedding_size"):
        return config.embedding_size
    return config.hidden_size


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "pretrained_dir",
        type=Path,
        help="Directory containing the Hugging Face model and tokenizer. "
        r"Should be the LFS commit directory of the HF repo e.g. ${HAFH}/"
        "models/base-hf/TinyLlama-1.1B-intermediate-step-1431k-3T/snapshots/036fa4651240b9a1487f709833b9e4b96b4c1574/",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Output directory for extended model and tokenizer"
    )
    parser.add_argument("--n-dsus", type=int, required=True, help="Number of HuBERT tokens (DSUs) to add")
    parser.add_argument(
        "--no-modality-tokens",
        action="store_false",
        dest="use_modality_tokens",
        help="Do no prepend special modality tokens to spans of text/speech tokens",
    )
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    return parser.parse_args()


def extend_llama_dsus(*, n_dsus, output_dir, pretrained_dir, seed, use_modality_tokens) -> None:
    if not pretrained_dir.exists():
        raise FileNotFoundError(f"Pretrained model directory {pretrained_dir} not found. Download artefacts first")
    if output_dir.exists():
        raise FileExistsError(f"Output directory {output_dir} already exists")
    seed_everything(seed)
    m = sentencepiece_model_pb2.ModelProto()
    with open(pretrained_dir / SENTENCEPIECE_TOKENIZER_FILENAME, "rb") as f:
        m.ParseFromString(f.read())
    vocab_size_curr = len(m.pieces)
    dsu_tokens = [dsu2pua(idx_dsu) for idx_dsu in range(n_dsus)]
    for token in dsu_tokens:
        new_token = sentencepiece_model_pb2.ModelProto().SentencePiece()
        new_token.piece = token
        new_token.score = 0
        # new_token.type = 0 # NOTE For future use if needed
        m.pieces.append(new_token)
    LOGGER.info(f"Added {len(dsu_tokens)} DSU tokens ({dsu_tokens[0]}...{dsu_tokens[-1]}) to tokenizer")
    if use_modality_tokens:
        modality_tokens = [MODALITY_TOKEN_TEXT, MODALITY_TOKEN_SPEECH]
        for token in modality_tokens:
            new_token = sentencepiece_model_pb2.ModelProto().SentencePiece()
            new_token.piece = token
            new_token.score = 0
            # new_token.type = 0 # NOTE For future use if needed
            m.pieces.append(new_token)
        LOGGER.info(f"Added modality switch tokens to tokenizer: {MODALITY_TOKEN_TEXT, MODALITY_TOKEN_SPEECH}")
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    with open(output_dir / SENTENCEPIECE_TOKENIZER_FILENAME, "xb") as f:
        f.write(m.SerializeToString())
    vocab_size_extd = len(m.pieces)
    n_new_tkns = vocab_size_extd - vocab_size_curr
    LOGGER.info(f"Extended tokenizer vocab size from {vocab_size_curr} to {vocab_size_extd} ({n_new_tkns} new tokens)")
    LOGGER.info(f"Saved SentencePiece tokenizer to {output_dir / SENTENCEPIECE_TOKENIZER_FILENAME!s}")
    # Resize model embedding layer
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(pretrained_dir)
    LOGGER.info(f"Loaded weights from {pretrained_dir!s}")
    if vocab_size_curr != model.config.vocab_size:
        raise AssertionError(f"Vocab size mismatch model vs tokenizer: {vocab_size_curr} != {model.config.vocab_size}")
    n_pad_embd_vctrs = MAKE_VOCAB_SIZE_DIVISIBLE_BY - (vocab_size_extd % MAKE_VOCAB_SIZE_DIVISIBLE_BY)
    model.resize_token_embeddings(vocab_size_extd, pad_to_multiple_of=MAKE_VOCAB_SIZE_DIVISIBLE_BY)
    n_new_emb_vctrs = model.config.vocab_size - vocab_size_curr
    embed_dim = get_hf_config_embedding_size(model.config)
    # Initialize new token embeddings as normal vectors with mean equal to current embeddings' mean
    with torch.no_grad():
        # input embedding layer -> init input padding vectors as NaN; should never be accessed/propagated through model
        new_pads_embed_init = torch.full((n_pad_embd_vctrs, embed_dim), torch.nan)
        input_embeddings = model.get_input_embeddings()
        mvnorm_input = multivariate_normal_from_weights(input_embeddings.weight)
        new_tkns_embed_init = mvnorm_input.sample(torch.Size((n_new_tkns,)))
        input_embeddings.weight.data[vocab_size_curr:] = torch.cat((new_tkns_embed_init, new_pads_embed_init))
        # output embedding layer (no weight tying) -> init all new vectors as MV Gaussian
        output_embeddings = model.get_output_embeddings()
        mvnorm_output = multivariate_normal_from_weights(output_embeddings.weight)
        new_tkns_embed_init = mvnorm_output.sample(torch.Size((n_new_emb_vctrs,)))  # resample new multivar. Gaussians
        output_embeddings.weight.data[vocab_size_curr:] = new_tkns_embed_init
    LOGGER.info("Initialized new embedding vectors via multivariate normal of trained embeddings")
    LOGGER.info(f"Extended embedding layer from {vocab_size_curr} to {model.config.vocab_size}")
    LOGGER.info(f"Added {n_new_emb_vctrs} new embedding vectors: {n_new_tkns} trained and {n_new_emb_vctrs} padding")
    # Save model weights
    model.save_pretrained(output_dir)
    LOGGER.info(f"Saved tokenizer and model weights to {output_dir}")


if __name__ == "__main__":
    extend_llama_dsus(**vars(parse_args()))
