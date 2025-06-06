#!/usr/bin/env python

"""Joins the time alignment columns onto the MLS dataset encoded as SpeechTokenizer RVQ tokens."""

import gc
from math import ceil
from pathlib import Path

from datasets import load_dataset, Split


# Constants
MLS_SIZES = {"train": 10_808_037, "dev": 3_807, "test": 3_769}

# Arguments
CHUNK_SIZE = 100_000
SPLIT = Split.VALIDATION  # "train", "dev", or "test" # TODO BUG HF rejects "dev" or "validation" resp.
ALIGNED_REPO_ID: str = "anilkeshwani/MLS_english_train_strat_sample_aligned_hubert"
MLS_STOK_REPO_ID: str = "anilkeshwani/mls-speechtokenizer"
JOINED_REPO_ID: str = "mls-speechtokenizer-RVQ-0-aligned"
LOCAL_PARQUET_OUTPUT_DIR: Path = Path("/mnt/scratch-artemis/anilkeshwani/mls-st-aligned/dev").resolve()

aligned = load_dataset(ALIGNED_REPO_ID, split="dev").remove_columns(["speech_tokens"]).to_pandas()
stok = load_dataset(MLS_STOK_REPO_ID, split="validation").remove_columns([f"RVQ_{i}" for i in range(1, 8)]).to_pandas()

# Check datasets contain matching samples
ids_match = (aligned["ID"] == aligned["ID"]).all()
if not ids_match:
    raise ValueError("IDs do not match along the series. This may indicate a problem with the dataset.")
else:
    print("IDs match")

# Rename RVQ_0 to speech_tokens (standard key)
stok.rename(columns={"RVQ_0": "speech_tokens"}, inplace=True)
print("Renamed column RVQ_0 -> speech_tokens")
joined = aligned.merge(stok, on="ID", how="inner")  # Inner join on a shared key "ID"

# Free up memory
del aligned, stok
gc.collect()

print(f"{len(joined) = :,}")
print(f"{joined.columns = }")

if not LOCAL_PARQUET_OUTPUT_DIR.exists():
    print(f"Creating output directory: {LOCAL_PARQUET_OUTPUT_DIR}")
    LOCAL_PARQUET_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Serialise as Parquet
n_chunks = ceil(len(joined) / CHUNK_SIZE)
print(f"Splitting dataset into {n_chunks} chunks of size {CHUNK_SIZE:,} each...")
for i in range(n_chunks):
    chunk = joined.iloc[i * CHUNK_SIZE : min((i + 1) * CHUNK_SIZE, len(joined))]
    print(f"Processing chunk {i + 1} of {n_chunks} (size: {len(chunk):,})")
    chunk_label = str(i + 1).zfill(len(str(n_chunks)))
    # Serialise as Parquet using pandas.DataFrame.to_parquet
    chunk.to_parquet(LOCAL_PARQUET_OUTPUT_DIR / f"data-{chunk_label}-of-{n_chunks}-{JOINED_REPO_ID}-{SPLIT}.parquet")
