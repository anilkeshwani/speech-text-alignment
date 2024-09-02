import json
import logging
import os
import sys
from argparse import ArgumentTypeError
from pathlib import Path

import torch
from tqdm import tqdm

from sardalign.config import LOG_DATEFMT, LOG_FORMAT, LOG_LEVEL
from sardalign.constants import PUA_PL_END, PUA_PL_START


logging.basicConfig(
    format=LOG_FORMAT,
    datefmt=LOG_DATEFMT,
    level=os.environ.get("LOGLEVEL", LOG_LEVEL).upper(),
    stream=sys.stdout,
)

LOGGER = logging.getLogger(__name__)


################################################################################
# General
################################################################################


def dsu2pua(idx_dsu: int) -> str:
    dsu_ord = PUA_PL_START + idx_dsu
    if dsu_ord > PUA_PL_END:
        raise RuntimeError(f"DSU ordinal out of PUA range: {idx_dsu}. PUA range: {PUA_PL_START - PUA_PL_END:,}")
    return chr(dsu_ord)


def get_type_mapping(data):
    """
    Recursively generates a type mapping for the given data structure. Intended for deserialised JSON.

    Args:
        data: The data structure to generate a type mapping for. Can be a dict, list, or any other type.

    Returns:
        A type mapping for the given data structure, represented as a nested dict or list of type names.

    Notes:
        Function intended for use with data resulting from JSON (lines) deserialisation, which yields
        lists not tuples for sequence data types and dictionaries not other mapping types for mappings.
    """
    if isinstance(data, dict):
        return {key: get_type_mapping(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [get_type_mapping(item) for item in data]
    else:
        return type(data).__name__


################################################################################
# Data helpers
################################################################################


def write_jsonl(
    jsonl: Path, samples: list[dict], mode: str = "x", encoding: str = "utf-8", ensure_ascii: bool = False
) -> None:
    with open(jsonl, mode=mode, encoding=encoding) as f:
        f.write("\n".join(json.dumps(sd, ensure_ascii=ensure_ascii) for sd in samples) + "\n")


def read_jsonl(jsonl: Path, mode: str = "r", encoding: str = "utf-8") -> list[dict]:
    with open(jsonl, mode=mode, encoding=encoding) as f:
        return [json.loads(line) for line in tqdm(f, desc=f"Reading data from {jsonl!s}")]


def count_lines(file: Path | str, max_bytes_to_check: int = len(b"\n")) -> int:
    with open(file, mode="rb") as f:
        _current_stream_position = f.tell()
        if f.seek(0, os.SEEK_END) < max_bytes_to_check:
            raise ValueError(
                f"Cannot count lines. Input file too small when checking {max_bytes_to_check} bytes: {file!s}"
            )
        f.seek(-max_bytes_to_check, os.SEEK_END)
        data = f.read(max_bytes_to_check)
        if not data.endswith(b"\n"):
            raise ValueError(f"Input file is not terminated with a trailing newline: {file!s}")
        f.seek(_current_stream_position)
        return sum(1 for _ in f)


def ljspeech_id_to_path(lj_id: str, audio_dir: Path, suffix: str = ".wav") -> Path:
    return (audio_dir / lj_id).with_suffix(suffix)


def mls_id_to_path(mls_id: str, audio_dir: Path, suffix: str = ".flac") -> Path:
    """_summary_

    Args:
        mls_id (str): ID as found in transcripts.txt file e.g. 10214_10108_000000
        audio_dir (Path): "audio" directory e.g. /mnt/scratch-artemis/anilkeshwani/data/MLS/mls_english/dev/audio
        suffix (str, optional): File extension. Defaults to ".flac".

    Returns:
        Path: Resolved path pointing to audio file
    """
    speaker_id, book_id, file_specifier = mls_id.removesuffix(suffix).split("_")
    return (audio_dir / speaker_id / book_id / mls_id).with_suffix(suffix)


def shard_jsonl(
    jsonl: Path,
    *,
    shard_size: int | None = None,
    n_shards: int | None = None,
    shard_dir: Path | None = None,
) -> None:
    if (shard_size is None) == (n_shards is None):  # XOR
        raise ValueError("Specify exactly one of `shard_size` or `n_shards`")
    dataset = read_jsonl(jsonl)
    if n_shards is not None:
        shard_size = -(-len(dataset) // n_shards)
    if shard_dir is None:
        shard_dir = jsonl.parent / f"{jsonl.stem}_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shards = [dataset[i : i + shard_size] for i in range(0, len(dataset), shard_size)]
    for i, shard in enumerate(shards):
        shard_jsonl = shard_dir / jsonl.with_stem(f"{jsonl.stem}_shard_{i:0{len(str(n_shards))}}").name
        write_jsonl(shard_jsonl, shard)
    LOGGER.info(f"Sharded {jsonl} into {len(shards)} shards in {shard_dir}")


################################################################################
# Hardware helpers
################################################################################


def echo_environment_info(torch, torchaudio, device: torch.device) -> None:
    print("Using torch version:", torch.__version__)
    print("Using torchaudio version:", torchaudio.__version__)
    print("Using device: ", device)


def get_device(device: str | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    mps_available = torch.backends.mps.is_available()
    mps_built = torch.backends.mps.is_built()
    local_device = "mps" if (mps_available and mps_built) else "cpu"
    return torch.device("cuda" if torch.cuda.is_available() else local_device)


################################################################################
# Argument parsing helpers
################################################################################


def get_integer_sample_size(sample_size: int | float, N: int) -> int:
    if not isinstance(sample_size, (int, float)):
        raise TypeError(f"sample_size should be one of int or float but got {type(sample_size)}")
    if isinstance(sample_size, float):
        sample_size = int(sample_size * N)
    return sample_size


def parse_arg_int_or_float(value: str) -> int | float:
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            raise ArgumentTypeError(f"{value} is neither an integer nor a float.")
