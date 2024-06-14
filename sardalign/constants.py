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

# HuBERT
HUBERT_DOWNSAMPLING_RATIO: int = 320

# Dataset Processing
TEXT_KEY: str = "transcript"  # key for text transcripts (pre-tokenization) in JSON lines manifests
SPEECH_TOKENS_KEY: str = "speech_tokens"  # key for speech tokens in JSON lines manifests
ALIGNMENT_KEY: str = "alignment"  # key for text token-audio alignments in JSON lines manifests
HUBERT_TOKEN_FSTRING: str = "<extra_id_{}>"
MODALITY_TOKEN_SPEECH: str = "<extra_id_MM_SPEECH>"
MODALITY_TOKEN_TEXT: str = "<extra_id_MM_TEXT>"
MEGATRON_TEXT_KEY: str = "text"

TOKEN_DELIMITER: str | None = None
