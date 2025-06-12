#!/usr/bin/env python

"""Joins the time alignment columns onto the MLS dataset encoded as SpeechTokenizer RVQ tokens."""

import gc
import logging
import os
import sys
from argparse import ArgumentParser, Namespace
from math import ceil
from pathlib import Path

from datasets import load_dataset
from pandas import DataFrame

from sardalign.config import LOG_DATEFMT, LOG_FORMAT, LOG_LEVEL
from sardalign.constants.mls import MLS_SPLIT_SIZES


logging.basicConfig(
    format=LOG_FORMAT,
    datefmt=LOG_DATEFMT,
    level=os.environ.get("LOGLEVEL", LOG_LEVEL).upper(),
    stream=sys.stdout,
)

LOGGER = logging.getLogger(__file__)

# NOTE Aligned and MLS SpeechTokenizer datasets are hard-coded due to dataset-specific logic e.g. column removal by name
ALIGNED_REPO_ID: str = "anilkeshwani/mls-hubert_large_ll60k-layer_22"
MLS_SPEECHTOKENIZER_REPO_ID: str = "anilkeshwani/mls-speechtokenizer"


# Joined dataset repository ID (default format string to be used with RVQ layer)
_JOINED_REPO_ID_DEFAULT: str = "mls-speechtokenizer-rvq_{}"


# Arguments
def parse_args() -> Namespace:
    parser = ArgumentParser(description="Join MLS time alignment with SpeechTokenizer RVQ tokens.")
    # Required
    parser.add_argument("--split", type=str, required=True, choices=["train", "validation", "test"])
    parser.add_argument("--output_dir", type=Path, required=True, help="Output directory for Parquet files")
    parser.add_argument("--layer", type=int, required=True, help="Residual Vector Quantizer to use", choices=range(8))
    # Optional
    parser.add_argument("--block_size", type=int, default=100_000, help="Size of each block to process")
    parser.add_argument("--joined_repo_id", type=str, help="Repository ID for the joined dataset")
    args = parser.parse_args()
    if args.joined_repo_id is None:
        args.joined_repo_id = _JOINED_REPO_ID_DEFAULT.format(args.layer)
    return args


def main(args: Namespace) -> None:
    speech_tokens_colname = "speech_tokens"
    aligned_drop_cols = [speech_tokens_colname]
    mls_speechtokenizer_drop_cols = [f"RVQ_{i}" for i in range(8) if i != args.layer]
    mls_speechtokenizer_rvq_colname = f"RVQ_{args.layer}"
    mls_aligned: DataFrame = (
        load_dataset(ALIGNED_REPO_ID, split=args.split).remove_columns(aligned_drop_cols).to_pandas()  # type: ignore
    )
    mls_speechtokenizer: DataFrame = (
        load_dataset(MLS_SPEECHTOKENIZER_REPO_ID, split=args.split)
        .remove_columns(mls_speechtokenizer_drop_cols)
        .to_pandas()  # type: ignore
    )

    # Checks
    MLS_SPLIT_SIZE = MLS_SPLIT_SIZES[args.split]
    if len(mls_aligned) != MLS_SPLIT_SIZE:
        raise ValueError(
            f"MLS aligned dataset size {len(mls_aligned)} does not match expected split size {MLS_SPLIT_SIZE}."
        )
    if len(mls_speechtokenizer) != MLS_SPLIT_SIZE:
        raise ValueError(
            f"MLS SpeechTokenizer dataset size {len(mls_speechtokenizer)} does not match expected split "
            f"size {MLS_SPLIT_SIZE}."
        )
    if not (mls_aligned["ID"] == mls_speechtokenizer["ID"]).all():
        raise ValueError("IDs do not match along the series. This may indicate a problem with the dataset.")
    else:
        LOGGER.info("IDs match")

    # Rename RVQ_0 to speech_tokens (standard key)
    mls_speechtokenizer.rename(columns={mls_speechtokenizer_rvq_colname: speech_tokens_colname}, inplace=True)
    LOGGER.info(f"Renamed column {mls_speechtokenizer_rvq_colname} -> {speech_tokens_colname}")
    mls_joined: DataFrame = mls_aligned.merge(mls_speechtokenizer, on="ID", how="inner")

    # Free up memory
    del mls_aligned, mls_speechtokenizer
    gc.collect()

    LOGGER.info(f"{len(mls_joined) = :,}")
    LOGGER.info(f"{mls_joined.columns = }")

    if not args.output_dir.exists():
        LOGGER.info(f"Creating output directory: {args.output_dir}")
        args.output_dir.mkdir(parents=True, exist_ok=True)

    # Serialise as Parquet
    n_blocks = ceil(len(mls_joined) / args.block_size)
    LOGGER.info(f"Splitting dataset into {n_blocks} blocks of size {args.block_size:,} each...")
    for i in range(n_blocks):
        block = mls_joined.iloc[i * args.block_size : min((i + 1) * args.block_size, len(mls_joined))]
        block_label = str(i + 1).zfill(len(str(n_blocks)))
        LOGGER.info(f"Processing block {block_label} of {n_blocks} (size: {len(block):,})")
        # Serialise as Parquet using pandas.DataFrame.to_parquet
        parquet_filename = f"{args.split}-{block_label}-of-{n_blocks}-{args.joined_repo_id}.parquet"
        block.to_parquet(args.output_dir / parquet_filename)


if __name__ == "__main__":
    args = parse_args()
    main(args)
