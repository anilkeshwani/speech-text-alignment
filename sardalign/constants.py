from pathlib import Path


PROJECT_ROOT: Path = Path(__file__).parents[1]
SEED: int = 42831

# Common
SAMPLING_FREQ: int = 16_000

# Alignment
EMISSION_INTERVAL: int = 30
STAR_TOKEN: str = "<star>"
CTC_ALIGNMENT_MLING_UROMAN_MODEL_PATH: str = "/tmp/ctc_alignment_mling_uroman_model.pt"
CTC_ALIGNMENT_MLING_UROMAN_DICT_PATH: str = "/tmp/ctc_alignment_mling_uroman_model.dict"
