import logging
from pathlib import Path

import torch
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset

from sardalign.constants.mls import MLS_SPLIT_SIZES


LOGGER = logging.getLogger(__name__)


def mls_id_to_path(mls_id: str, audio_dir: Path, suffix: str = ".flac") -> Path:
    """Resolve the path to an MLS audio file given its ID.

    Args:
        mls_id (str): ID as found in transcripts.txt file e.g. 10214_10108_000000
        audio_dir (Path): "audio" directory e.g. /mnt/scratch-artemis/anilkeshwani/data/MLS/mls_english/dev/audio
        suffix (str, optional): File extension. Defaults to ".flac".

    Returns:
        Path: Resolved path pointing to audio file
    """
    speaker_id, book_id, file_specifier = mls_id.removesuffix(suffix).split("_")
    return (audio_dir / speaker_id / book_id / mls_id).with_suffix(suffix)
