import json
import logging
import os
import sys
from argparse import ArgumentTypeError
from pathlib import Path

import torch
from sardalign.config import LOG_DATEFMT, LOG_FORMAT, LOG_LEVEL
from tqdm import tqdm


logging.basicConfig(
    format=LOG_FORMAT,
    datefmt=LOG_DATEFMT,
    level=os.environ.get("LOGLEVEL", LOG_LEVEL).upper(),
    stream=sys.stdout,
)

LOGGER = logging.getLogger(__name__)

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


def count_lines(file: Path | str, max_bytes_to_check: int = 32):
    with open(file, mode="rb") as f:
        f.seek(-max_bytes_to_check, os.SEEK_END)
        data = f.read(max_bytes_to_check)
        if not data.endswith(b"\n"):
            raise ValueError(f"Input file is not terminated with a trailing newline: {file!s}")
        f.seek(0)
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
