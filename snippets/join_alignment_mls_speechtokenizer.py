#!/usr/bin/env python

"""Joins the time alignment columns onto the MLS dataset encoded as SpeechTokenizer RVQ tokens."""

import gc
import logging
import os
import sys
from argparse import ArgumentParser, Namespace
from math import ceil
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset, NamedSplit
from huggingface_hub import HfApi
from pandas import DataFrame

from sardalign.config import LOG_DATEFMT, LOG_FORMAT, LOG_LEVEL


logging.basicConfig(
    format=LOG_FORMAT,
    datefmt=LOG_DATEFMT,
    level=os.environ.get("LOGLEVEL", LOG_LEVEL).upper(),
    stream=sys.stdout,
)

LOGGER = logging.getLogger(__file__)

HF_DATASETS_BASE_URL: str = "https://huggingface.co/datasets"
REPO_TYPE: str = "dataset"

# Arguments
CHUNK_SIZE = 100_000
SPLIT = "train"
REPO_NAME: str = "mls-speechtokenizer-RVQ-0-aligned"
PRIVATE: bool = False
COMMIT_MESSAGE: str = "Upload dataset"
LOCAL_PARQUET_OUTPUT_DIR: Path = Path("/mnt/scratch-artemis/anilkeshwani/mls-st-aligned")

hfapi = HfApi()
wai = hfapi.whoami()
username = wai["name"]
LOGGER.info(f"Logged in as user {username} ({wai['fullname']}. Email: {wai['email']})")
REPO_ID = "/".join((username, REPO_NAME))

mls_aligned = (
    load_dataset("anilkeshwani/mls-hubert_large_ll60k-layer_22", split=SPLIT)
    .remove_columns(["speech_tokens"])
    .to_pandas()
)
mls_st = (
    load_dataset("anilkeshwani/mls-speechtokenizer", split=SPLIT)
    .remove_columns([f"RVQ_{i}" for i in range(1, 8)])
    .to_pandas()
)

mls_st.rename(columns={"RVQ_0": "speech_tokens"}, inplace=True)
LOGGER.info("Renamed column RVQ_0 -> speech_tokens")

LOGGER.info(f"{type(mls_aligned['ID']) = }")
LOGGER.info(f"{type(mls_st['ID']) = }")
LOGGER.info(f"{len(mls_aligned['ID']) = }")
LOGGER.info(f"{len(mls_st['ID']) = }")

LOGGER.info(f"Check IDs match along series: {(mls_aligned['ID'] == mls_aligned['ID']).all()}")

# Inner join on a shared key "ID"
joined: DataFrame = mls_aligned.merge(mls_st, on="ID", how="inner")

del mls_aligned, mls_st
gc.collect()

LOGGER.info(f"{len(joined) = :,}")
LOGGER.info(f"{joined.columns = }")

# Chunk and serialise as Parquet #############################################

n_chunks = ceil(len(joined) / CHUNK_SIZE)
LOGGER.info(f"Splitting dataset into {n_chunks} chunks of size {CHUNK_SIZE:,} each...")

for i in range(n_chunks):
    chunk: DataFrame = joined.iloc[i * CHUNK_SIZE : min((i + 1) * CHUNK_SIZE, len(joined))]
    LOGGER.info(f"Processing chunk {i + 1} of {n_chunks} (size: {len(chunk):,})")
    chunk_label = str(i + 1).zfill(len(str(n_chunks)))
    # Serialise as Parquet using pandas.DataFrame.to_parquet
    chunk.to_parquet(LOCAL_PARQUET_OUTPUT_DIR / f"data-{chunk_label}-of-{n_chunks}-{REPO_NAME}-{SPLIT}.parquet")
