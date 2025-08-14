import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
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


class MLSDataset(Dataset):
    def __init__(self, segments: Path, audio_dir: Path, strict: bool = True):
        self.audio_dir = audio_dir
        with open(segments, "r") as f:
            self.mls_ids = [line.strip().split(None, 1)[0] for line in f]

        if len(self.mls_ids) not in MLS_SPLIT_SIZES.values():
            message = f"Dataset size {len(self.mls_ids)} does not match any of the MLS split sizes."
            if strict:
                raise ValueError(message + " Consider using torch.utils.data.Subset.")
            else:
                LOGGER.warning(message + " Proceeding anyway.")

    def __len__(self) -> int:
        return len(self.mls_ids)

    def __getitem__(self, idx: int) -> tuple[str, tuple[Tensor, int]]:
        mls_id = self.mls_ids[idx]
        wav, sr = torchaudio.load(mls_id_to_path(mls_id, self.audio_dir))
        return mls_id, (wav, sr)


def collate_fn_mls(
    batch: list[tuple[str, tuple[Tensor, int]]],
    sample_rate: int,
    channels: str = "first",
    max_duration: float | None = None,
) -> dict[str, Any]:
    """Collate function for MLS dataset.

    Args:
        batch (list[tuple[str, tuple[Tensor, int]]]): List of tuples containing MLS ID and (wav, sample_rate).
        sample_rate (int): Target sample rate for audio.
        channels (str, optional): How to handle multiple channels. Options: "all", "first", "mean". Defaults to "first".
        max_duration (float | None, optional): Maximum duration in seconds for audio. If None, no truncation is applied.

    Returns:
        dict[str, Any]: Dictionary containing "mls_ids", "wavs" (padded tensor), and "lengths" (list of lengths).
    """

    channels_fn_map: dict[str, Callable[[Tensor], Tensor]] = {
        "all": lambda x: x,
        "first": lambda x: x[:1],
        "mean": lambda x: x.mean(dim=0, keepdim=True),
    }
    channels_fn = channels_fn_map.get(channels)
    if channels_fn is None:
        raise ValueError(f"Invalid channels choice: {channels}. Choose from {channels_fn_map.keys()}")

    wavs: list[Tensor] = []
    lengths: list[int] = []
    mls_ids: list[str] = []
    for mls_id, (wav, sr) in batch:
        if wav.dim() != 2:
            raise ValueError(f"MLS audio {mls_id} is expected to be 2D (channels, samples), but got {wav.dim()}D.")
        if wav.size(0) > 1 and channels != "all":
            LOGGER.warning(f"MLS audio {mls_id} is not monophonic. Shape: {wav.shape}")  # NOTE warn since MLS is mono
            wav = channels_fn(wav)
        if sr != sample_rate:
            wav = torchaudio.functional.resample(wav, sr, sample_rate)
        if max_duration is not None and wav.size(1) > sample_rate * max_duration:
            LOGGER.warning(f"MLS audio {mls_id} exceeds max duration {max_duration} seconds. Truncating.")
            wav = wav[:, : int(sample_rate * max_duration)]
        if wav.size(0) == 0:
            raise RuntimeError(f"MLS audio {mls_id} is empty after processing.")
        wavs.append(wav)
        lengths.append(wav.size(1))
        mls_ids.append(mls_id)

    max_length = max(lengths)
    wavs_padded: list[Tensor] = [F.pad(wav, (0, max_length - len_wav)) for wav, len_wav in zip(wavs, lengths)]
    wavs_tensor = torch.stack(wavs_padded, dim=0)  # [B, C, T]
    return {"mls_ids": mls_ids, "wavs": wavs_tensor, "lengths": lengths}
