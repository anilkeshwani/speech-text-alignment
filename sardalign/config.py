LOG_LEVEL: str = "INFO"
LOG_FORMAT: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
LOG_DATEFMT: str = "%Y-%m-%d %H:%M:%S"

# Dataset Processing
EXP_SPAN_LENGTH: float = 39.43 / 4  # numerator: mean sequence length (tokens) in the MLS 25% stratified sample
