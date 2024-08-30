import logging
import os
import sys
from itertools import groupby
from pathlib import Path

import sox
import torch
import torchaudio
import torchaudio.functional as F
from torch import Tensor

from sardalign.config import LOG_DATEFMT, LOG_FORMAT, LOG_LEVEL
from sardalign.constants import EMISSION_INTERVAL, SAMPLING_FREQ
from sardalign.utils.align import merge_repeats, Segment, time_to_frame


logging.basicConfig(
    format=LOG_FORMAT,
    datefmt=LOG_DATEFMT,
    level=os.environ.get("LOGLEVEL", LOG_LEVEL).upper(),
    stream=sys.stdout,
)

LOGGER = logging.getLogger(__name__)


class AlignmentException(Exception):
    pass


def generate_emissions(
    model: torchaudio.models.Wav2Vec2Model,
    audio_file: str | Path,
    device: torch.device,
    check_sampling_rate: int | None = SAMPLING_FREQ,
) -> tuple[Tensor, float, Tensor]:
    waveform, sr = torchaudio.load(audio_file)  # waveform: channels X T
    waveform = waveform.to(device)
    total_duration = sox.file_info.duration(audio_file)  # output from bash soxi -D "$audio_file"; e.g. 15.180000
    if total_duration is None:
        raise RuntimeError(f"Could not determine duration of audio file: {audio_file!s}")
    if check_sampling_rate and (sr != SAMPLING_FREQ):
        raise RuntimeError(f"Expected sampling rate {check_sampling_rate:,}, found: {sr:,} for {audio_file!s}")
    context: float = EMISSION_INTERVAL * 0.1
    emissions_arr = []
    with torch.inference_mode():
        t_seconds = 0
        while t_seconds < total_duration:  # NOTE usually lasts 1 iteration - EMISSION_INTERVAL is 30s out-of-the-box
            segment_start_time, segment_end_time = (t_seconds, t_seconds + EMISSION_INTERVAL)
            input_start_time = max(segment_start_time - context, 0)
            input_end_time = min(segment_end_time + context, total_duration)
            waveform_split = waveform[:, int(SAMPLING_FREQ * input_start_time) : int(SAMPLING_FREQ * (input_end_time))]
            model_outs, _ = model(waveform_split)
            emissions_ = model_outs[0]
            emission_start_frame = time_to_frame(segment_start_time)
            emission_end_frame = time_to_frame(segment_end_time)
            offset = time_to_frame(input_start_time)
            emissions_ = emissions_[emission_start_frame - offset : emission_end_frame - offset, :]
            emissions_arr.append(emissions_)
            t_seconds += EMISSION_INTERVAL
    emissions = torch.cat(emissions_arr, dim=0).squeeze()
    emissions = torch.log_softmax(emissions, dim=-1)
    stride_ms = float(waveform.size(1) * 1000 / emissions.size(0) / SAMPLING_FREQ)  # milliseconds
    return emissions, stride_ms, waveform


def get_alignments(
    audio_file: str | Path,
    uroman_tokens: list[str],
    model: torchaudio.models.Wav2Vec2Model,
    dictionary: dict[str, int],
    use_star: bool,
    device: torch.device,
) -> tuple[list[Segment], float, Tensor]:
    # generate emissions: log prob distributions of uroman tokens per Wav2Vec2 output frame (usually 320x downsampling)
    emissions, stride_ms, waveform = generate_emissions(model, audio_file, device)
    T, N = emissions.size()
    if use_star:
        emissions = torch.cat([emissions, torch.zeros(T, 1).to(device)], dim=1)  # add star entry to dist with zero prob
    # force alignment
    if uroman_tokens:
        token_indices = [dictionary[c] for c in " ".join(uroman_tokens).split(" ") if c in dictionary]  # removes spaces
    else:
        raise AlignmentException("Empty transcript for audio-text sample")
    blank = dictionary["<blank>"]
    targets = torch.tensor(token_indices, dtype=torch.int32).to(device)
    input_lengths = torch.tensor(emissions.shape[0]).unsqueeze(-1)
    target_lengths = torch.tensor(targets.shape[0]).unsqueeze(-1)
    # sum consecutive repeats -> validate targets + repeats <= logprobs condition to run CTC forced alignment
    str_grpd_by_tkn = ((k, sum(1 for _ in g)) for k, g in groupby(token_indices))
    n_repeats = sum(cnt - 1 for k, cnt in str_grpd_by_tkn if cnt > 1)
    if target_lengths + n_repeats > input_lengths:
        raise AlignmentException(
            "Targets length is too long for CTC. "
            f"Found targets length: {target_lengths.item()}, log_probs length: {input_lengths.item()}; "
            "repeats not considered"
        )
    path, _ = F.forced_align(emissions.unsqueeze(0), targets.unsqueeze(0), input_lengths, target_lengths, blank=blank)
    path = path.squeeze().to("cpu").tolist()
    segments = merge_repeats(path, {v: k for k, v in dictionary.items()})
    return segments, stride_ms, waveform
