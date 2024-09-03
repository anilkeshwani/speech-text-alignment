#!/usr/bin/env python

import logging
import os
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
from sentencepiece import sentencepiece_model_pb2
from transformers import AutoModelForCausalLM

from sardalign.config import LOG_DATEFMT, LOG_FORMAT, LOG_LEVEL
from sardalign.constants import MODALITY_TOKEN_SPEECH, MODALITY_TOKEN_TEXT, SEED
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


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("pretrained_dir", type=Path, help="Directory containing the Hugging Face model and tokenizer")
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


def main(args: Namespace) -> None:
    if not args.pretrained_dir.exists():
        raise FileNotFoundError(f"Pretrained model directory {args.pretrained_dir} not found. Download artefacts first")
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory {args.output_dir} already exists")
    seed_everything(args.seed)
    m = sentencepiece_model_pb2.ModelProto()
    with open(args.pretrained_dir / SENTENCEPIECE_TOKENIZER_FILENAME, "rb") as f:
        m.ParseFromString(f.read())
    dsu_tokens = [dsu2pua(idx_dsu) for idx_dsu in range(args.n_dsus)]
    for token in dsu_tokens:
        new_token = sentencepiece_model_pb2.ModelProto().SentencePiece()
        new_token.piece = token
        new_token.score = 0
        # new_token.type = 0 # NOTE For future use if needed
        m.pieces.append(new_token)
    LOGGER.info(f"Added {len(dsu_tokens)} DSU tokens ({dsu_tokens[0]}...{dsu_tokens[-1]}) to tokenizer")
    if args.use_modality_tokens:
        modality_tokens = [MODALITY_TOKEN_TEXT, MODALITY_TOKEN_SPEECH]
        for token in modality_tokens:
            new_token = sentencepiece_model_pb2.ModelProto().SentencePiece()
            new_token.piece = token
            new_token.score = 0
            # new_token.type = 0 # NOTE For future use if needed
            m.pieces.append(new_token)
        LOGGER.info(f"Added modality switch tokens to tokenizer: {MODALITY_TOKEN_TEXT, MODALITY_TOKEN_SPEECH}")
    if not args.output_dir.exists():
        args.output_dir.mkdir(parents=True)
    with open(args.output_dir / SENTENCEPIECE_TOKENIZER_FILENAME, "xb") as f:
        f.write(m.SerializeToString())
    LOGGER.info(f"Saved SentencePiece tokenizer to {args.output_dir / SENTENCEPIECE_TOKENIZER_FILENAME!s}")
    # Resize model embedding layer
    model = AutoModelForCausalLM.from_pretrained(args.pretrained_dir)
    LOGGER.info(f"Looaded weights from {args.pretrained_dir!s}")
    vocab_size_curr = model.config.vocab_size
    vocab_size_extd = len(m.pieces)
    model.resize_token_embeddings(vocab_size_extd)
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
    # Save model weights
    model.save_pretrained(args.output_dir)
    LOGGER.info(f"Saved tokenizer and model weights to {args.output_dir}")


if __name__ == "__main__":
    main(parse_args())
