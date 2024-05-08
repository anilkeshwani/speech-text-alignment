import json
from pathlib import Path

import torch


def write_jsonl(
    jsonl: Path, samples: list[dict], mode: str = "x", encoding: str = "utf-8", ensure_ascii: bool = False
) -> None:
    with open(jsonl, mode=mode, encoding=encoding) as f:
        f.write("\n".join(json.dumps(sd, ensure_ascii=ensure_ascii) for sd in samples) + "\n")


def read_jsonl(jsonl: Path, mode: str = "r", encoding: str = "utf-8") -> list[dict]:
    with open(jsonl, mode=mode, encoding=encoding) as f:
        return [json.loads(line) for line in f]


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
