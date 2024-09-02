#!/usr/bin/env python

import logging
import os
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sardalign.config import LOG_DATEFMT, LOG_FORMAT, LOG_LEVEL
from sardalign.constants import MODALITY_TOKEN_SPEECH, MODALITY_TOKEN_TEXT, SEED
from sardalign.constants.tinyllama import TINYLLAMA_HF_REPO, TINYLLAMA_HF_REVISION
from sardalign.utils import dsu2pua, multivariate_normal_from_weights, seed_everything


# NOTE Script structured to be trivially extensible for use of other base models in future

logging.basicConfig(
    format=LOG_FORMAT,
    datefmt=LOG_DATEFMT,
    level=os.environ.get("LOGLEVEL", LOG_LEVEL).upper(),
    stream=sys.stdout,
)

LOGGER = logging.getLogger(__file__)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Output directory for the extended model and tokenizer"
    )
    parser.add_argument("--n-dsus", type=int, required=True, help="Number of HuBERT tokens (DSUs) to add")
    parser.add_argument(
        "--pretrained-model-name-or-path", type=str, default=None, help="Pretrained Hugging Face model repository name"
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,  # TODO when generalising, set default value to "main"
        help="Revision of the pretrained Hugging Face model (commit ID or branch name)",
    )
    parser.add_argument(
        "--no-weights",
        action="store_false",
        dest="download_weights",
        help="Do not download the model weights i.e. download only the tokenizer",
    )
    parser.add_argument(
        "--no-tokenizer",
        action="store_false",
        dest="download_tokenizer",
        help="Do not download the model's tokenizer i.e. download only the model weights",
    )
    parser.add_argument(
        "--use-fast-tokenizer", action="store_true", help="Use a fast Rust-based Hugging Face tokenizer (if available)"
    )
    parser.add_argument(
        "--no-modality-tokens",
        action="store_false",
        dest="use_modality_tokens",
        help="Do no prepend special modality tokens to spans of text/speech tokens",
    )
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    args = parser.parse_args()
    if args.pretrained_model_name_or_path is None:
        args.pretrained_model_name_or_path = TINYLLAMA_HF_REPO  # TODO remove when using Gemma
        LOGGER.info(f"No Hugging Face repository specified. Using default model: {args.pretrained_model_name_or_path}")
    if args.revision is None:
        args.revision = TINYLLAMA_HF_REVISION  # TODO remove when using Gemma
        LOGGER.info(
            "No Hugging Face repo revision (commit ID or branch name) specified. "
            f"Using default revision: {args.revision}"
        )
    return args


def main(args: Namespace) -> None:
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory {args.output_dir} already exists")
    seed_everything(args.seed)
    # Download tokenizer and model weights if necessary
    if args.download_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            revision=args.revision,
            # NOTE Use of slow tokenizer enforces serialisation of underlying SentencePiece model on save_pretrained()
            use_fast=args.use_fast_tokenizer,
        )
        LOGGER.info(f"Downloaded tokenizer for {args.pretrained_model_name_or_path}")
    if args.download_weights:
        model = AutoModelForCausalLM.from_pretrained(args.pretrained_model_name_or_path, revision=args.revision)
        LOGGER.info(f"Downloaded weights for {args.pretrained_model_name_or_path}")
    # Add new tokens
    dsu_tokens = [dsu2pua(idx_dsu) for idx_dsu in range(args.n_dsus)]
    tokenizer.add_tokens(dsu_tokens, special_tokens=False)
    LOGGER.info(f"Added {len(dsu_tokens)} DSU tokens ({dsu_tokens[0]}...{dsu_tokens[-1]}) to tokenizer")
    if args.use_modality_tokens:
        tokenizer.add_tokens([MODALITY_TOKEN_TEXT, MODALITY_TOKEN_SPEECH], special_tokens=False)
        LOGGER.info(f"Added modality switch tokens to tokenizer: {MODALITY_TOKEN_TEXT, MODALITY_TOKEN_SPEECH}")
    # Resize model embedding layer
    vocab_size_curr = model.config.vocab_size
    vocab_size_extd = len(tokenizer)
    model.resize_token_embeddings(len(tokenizer))
    n_new_tkns = vocab_size_extd - vocab_size_curr
    LOGGER.info(f"Extended vocab size from {vocab_size_curr} to {vocab_size_extd} with {n_new_tkns} new tokens")
    # Initialize new embeddings as normal vectors with mean equal to current embeddings' mean
    with torch.no_grad():
        input_embeddings = model.get_input_embeddings()
        mvnorm_input = multivariate_normal_from_weights(input_embeddings.weight)
        input_embeddings.weight.data[vocab_size_curr:] = mvnorm_input.sample(torch.Size((n_new_tkns,)))
        output_embeddings = model.get_output_embeddings()
        mvnorm_output = multivariate_normal_from_weights(output_embeddings.weight)
        output_embeddings.weight.data[vocab_size_curr:] = mvnorm_output.sample(torch.Size((n_new_tkns,)))
    LOGGER.info("Initialized new embedding vectors via multivariate normal of trained embeddings")
    # Save tokenizer and model weights
    tokenizer.save_pretrained(args.output_dir)
    model.save_pretrained(args.output_dir)
    LOGGER.info(f"Saved tokenizer and model weights to {args.output_dir}")


if __name__ == "__main__":
    main(parse_args())
