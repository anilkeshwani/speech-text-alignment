import json
import logging
import os
import random
from argparse import ArgumentTypeError
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.distributions.multivariate_normal import MultivariateNormal
from tqdm import tqdm

from sardalign.constants import PUA_PL_END, PUA_PL_START


LOGGER = logging.getLogger(__name__)


################################################################################
# General
################################################################################


def dsu2pua(idx_dsu: int) -> str:
    dsu_ord = PUA_PL_START + idx_dsu
    if dsu_ord > PUA_PL_END:
        raise RuntimeError(f"DSU ordinal out of PUA range: {idx_dsu}. PUA range: {PUA_PL_START - PUA_PL_END:,}")
    return chr(dsu_ord)


def pua2dsu(pua: str) -> int:
    if len(pua) != 1:
        raise ValueError(f"PUA should be a single character, got: {pua}")
    dsu_ord = ord(pua)
    if dsu_ord < PUA_PL_START or dsu_ord > PUA_PL_END:
        raise ValueError(f"PUA ordinal out of PUA range: {dsu_ord}. PUA range: {PUA_PL_START:,} - {PUA_PL_END:,}")
    return dsu_ord - PUA_PL_START


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


def multivariate_normal_from_weights(embeddings: Tensor, sigma_scaling: float = 1e-5) -> MultivariateNormal:
    mu = torch.mean(embeddings, dim=0)
    n = embeddings.size(0)
    sigma = ((embeddings - mu).T @ (embeddings - mu)) / n
    return MultivariateNormal(mu, covariance_matrix=sigma * sigma_scaling)


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


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
