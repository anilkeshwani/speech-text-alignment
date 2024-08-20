from pathlib import Path


# Common
PROJECT_ROOT: Path = Path(__file__).parents[2]
SEED: int = 42831

# Audio
SAMPLING_FREQ: int = 16_000

# Alignment
EMISSION_INTERVAL: int = 30
STAR_TOKEN: str = "<star>"
CTC_ALIGNMENT_MLING_UROMAN_MODEL_PATH: str = "/tmp/ctc_alignment_mling_uroman_model.pt"
CTC_ALIGNMENT_MLING_UROMAN_DICT_PATH: str = "/tmp/ctc_alignment_mling_uroman_model.dict"

# HuBERT
HUBERT_DOWNSAMPLING_RATIO: int = 320

# Tokenization
TOKEN_DELIMITER_DEFAULT: str | None = None  # None induces str.split to split on any whitespace

# Dataset Processing > JSON lines manifest keys
TEXT_KEY_DEFAULT: str = "transcript"  # key for text transcripts (pre-tokenization)
TOKENIZED_KEY: str = "tokenized"  # key for tokenized text (array of strings)
NORMALIZED_KEY: str = "normalized"  # key for normalized text (array of strings)
UROMAN_KEY: str = "uroman"  # key for uroman tokens (array of strings)
SPEECH_TOKENS_KEY: str = "speech_tokens"  # key for speech tokens ("DSUs")
ALIGNMENT_START_TIME_KEY: str = "aligned_token_start_time"  # key for text token-audio alignments
ALIGNMENT_END_TIME_KEY: str = "aligned_token_end_time"  # key for text token-audio alignments

ALIGNMENT_KEY: str = "alignment"  # TODO remove - deprecated for HF datasets compatibility

# Dataset Processing > HuBERT (DSU) and multimodalality textual representation
HUBERT_TOKEN_FSTRING: str = "<extra_id_{}>"
MODALITY_TOKEN_SPEECH: str = "<extra_id_MM_SPEECH>"
MODALITY_TOKEN_TEXT: str = "<extra_id_MM_TEXT>"

# Megatron
MEGATRON_TEXT_KEY: str = "text"
