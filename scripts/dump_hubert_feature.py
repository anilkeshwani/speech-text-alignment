#!/usr/bin/env python

import logging
import os
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from pprint import pformat

from sardalign.config import LOG_DATEFMT, LOG_FORMAT, LOG_LEVEL
from sardalign.utils.features import dump_feature, get_mls_path_iterator, HubertFeatureReader


logging.basicConfig(
    format=LOG_FORMAT,
    datefmt=LOG_DATEFMT,
    level=os.environ.get("LOGLEVEL", LOG_LEVEL).upper(),
    stream=sys.stdout,
)

LOGGER = logging.getLogger(__name__)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--jsonl", type=Path, required=True, help="Path to JSON lines manifest file")
    parser.add_argument("--audio-dir", type=Path, required=True, help="Directory containing MLS audios")
    parser.add_argument("--ckpt-path", type=str, required=True, help="Path to pre-trained HuBERT checkpoint")
    parser.add_argument("--layer", type=int, required=True, help="BERT layer whose embeddings to use as features")
    parser.add_argument("--nshard", type=int, default=1, help="Number of shards for parallel processing")
    parser.add_argument("--rank", type=int, default=0, help="GPU rank")
    parser.add_argument("--feat-dir", type=Path, required=True, help="Output directory for HuBERT features")
    parser.add_argument("--max-chunk", type=int, default=1_600_000, help="Maximum audio chunk length in samples")
    parser.add_argument("--suffix", type=str, default=".flac", help="File extension for audio files")
    args = parser.parse_args()
    return args


def main(
    jsonl: Path,
    audio_dir: Path,
    # NOTE ckpt_path can't be a `Path` as `fairseq.checkpoint_utils.get_maybe_sharded_checkpoint_filename` (line 383)
    #      calls filename.replace(".pt", suffix + ".pt") with ckpt_path passed as the filename
    ckpt_path: str,
    layer: int,
    nshard: int,
    rank: int,
    feat_dir: Path,
    max_chunk: int,
    suffix: str,
):
    split = jsonl.stem
    reader = HubertFeatureReader(ckpt_path, layer, max_chunk)
    generator, num = get_mls_path_iterator(jsonl, audio_dir=audio_dir, nshard=nshard, rank=rank, suffix=suffix)
    dump_feature(reader, generator, num, split, nshard, rank, feat_dir)


if __name__ == "__main__":
    args = parse_args()
    LOGGER.info(pformat(vars(args), sort_dicts=False))
    main(**vars(args))
