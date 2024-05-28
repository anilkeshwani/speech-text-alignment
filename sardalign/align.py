import logging
import os
import sys
from pathlib import Path

import sox
import torch
import torchaudio
import torchaudio.functional as F
from sardalign.config import LOG_DATEFMT, LOG_FORMAT, LOG_LEVEL
from sardalign.constants import EMISSION_INTERVAL, SAMPLING_FREQ
from sardalign.utils.align import merge_repeats, time_to_frame
from torch import Tensor


logging.basicConfig(
    format=LOG_FORMAT,
    datefmt=LOG_DATEFMT,
    level=os.environ.get("LOGLEVEL", LOG_LEVEL).upper(),
    stream=sys.stdout,
)

LOGGER = logging.getLogger(__name__)


def generate_emissions(
    model: torchaudio.models.Wav2Vec2Model, audio_file: str | Path, device: torch.device
) -> tuple[Tensor, float]:
    waveform, _ = torchaudio.load(audio_file)  # waveform: channels X T
    waveform = waveform.to(device)
    total_duration = sox.file_info.duration(audio_file)  # output from bash soxi -D "$audio_file"; e.g. 15.180000
    assert total_duration is not None
    audio_sf = sox.file_info.sample_rate(audio_file)
    assert audio_sf == SAMPLING_FREQ
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
    return emissions, stride_ms


def get_alignments(
    audio_file: str | Path,
    uroman_tokens: list[str],
    model: torchaudio.models.Wav2Vec2Model,
    dictionary: dict[str, int],
    use_star: bool,
    device: torch.device,
):
    # generate emissions: log prob distributions of uroman tokens per Wav2Vec2 output frame (usually 320x downsampling)
    emissions, stride_ms = generate_emissions(model, audio_file, device)
    T, N = emissions.size()
    if use_star:
        emissions = torch.cat([emissions, torch.zeros(T, 1).to(device)], dim=1)  # add star entry to dist with zero prob
    # force alignment
    if uroman_tokens:
        token_indices = [dictionary[c] for c in " ".join(uroman_tokens).split(" ") if c in dictionary]  # TODO ??
    else:
        LOGGER.warning(f"Empty transcript!!!!! for audio file {audio_file}")
        token_indices = []

    blank = dictionary["<blank>"]

    targets = torch.tensor(token_indices, dtype=torch.int32).to(device)

    input_lengths = torch.tensor(emissions.shape[0]).unsqueeze(-1)
    target_lengths = torch.tensor(targets.shape[0]).unsqueeze(-1)
    path, _ = F.forced_align(emissions.unsqueeze(0), targets.unsqueeze(0), input_lengths, target_lengths, blank=blank)
    path = path.squeeze().to("cpu").tolist()

    segments = merge_repeats(path, {v: k for k, v in dictionary.items()})
    return segments, stride_ms
