#!/usr/bin/env python

import logging
import os
import sys
from argparse import ArgumentParser, Namespace

from transformers import AutoModelForCausalLM, AutoTokenizer

from sardalign.config import LOG_DATEFMT, LOG_FORMAT, LOG_LEVEL
from sardalign.constants.tinyllama import TINYLLAMA_HF_REPO, TINYLLAMA_HF_REVISION


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
        "--pretrained_model_name_or_path", type=str, default=None, help="Pretrained Hugging Face model repository name"
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
    args = parser.parse_args()
    if args.pretrained_model_name_or_path is None:
        args.pretrained_model_name_or_path = TINYLLAMA_HF_REPO  # TODO remove when using Gemma
        LOGGER.info(f"No Hugging Face repository specified. Using default model: {args.pretrained_model_name_or_path}")
    if args.revision is None:
        args.revision = TINYLLAMA_HF_REVISION  # TODO remove when using Gemma
        LOGGER.info(
            "No Hugging Face repo revision (commit ID or branch name) specified. "
            f"Using default model: {args.revision}"
        )
    return args


def main(args: Namespace) -> None:
    if args.download_tokenizer:
        AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, revision=args.revision)
        LOGGER.info(f"Downloaded tokenizer for {args.pretrained_model_name_or_path}")
    if args.download_weights:
        AutoModelForCausalLM.from_pretrained(args.pretrained_model_name_or_path, revision=args.revision)
        LOGGER.info(f"Downloaded weights for {args.pretrained_model_name_or_path}")


if __name__ == "__main__":
    main(parse_args())
