#!/usr/bin/env python

import logging
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

from sentencepiece import sentencepiece_model_pb2

from sardalign.config import LOG_DATEFMT, LOG_FORMAT, LOG_LEVEL


logging.basicConfig(
    format=LOG_FORMAT,
    datefmt=LOG_DATEFMT,
    level=os.environ.get("LOGLEVEL", LOG_LEVEL).upper(),
    stream=sys.stdout,
)

LOGGER = logging.getLogger(__file__)


def main(pretrained_model: Path) -> None:
    m = sentencepiece_model_pb2.ModelProto()
    m.ParseFromString(open(pretrained_model, "rb").read())

    LOGGER.info(f"Loaded SentencePiece tokenizer from {pretrained_model}")
    LOGGER.info(f"SentencePiece vocab size: {len(m.pieces)}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("pretrained_model", type=Path)
    args = parser.parse_args()
    main(args.pretrained_model)
