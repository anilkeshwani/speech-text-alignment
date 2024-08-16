GS_HF_DATASET_REPO: str = "speechcolab/gigaspeech"

# We keep 4 punctuations in the normalized text (see the text_tn entry in GigaSpeech.json)
GS_PUNCTUATION_TAGS: list[str] = ["<COMMA>", "<PERIOD>", "<QUESTIONMARK>", "<EXCLAMATIONPOINT>"]

# The Dev/Test evaluation sets are annotated by human annotators. They are instructed to label the
# entire audio file without "gaps". So for non-speech segments, garbage utterance tags are used
# instead. We recommend our users to discard these utterances in your training.
# A complete list of these tags are:
GS_GARBAGE_TAGS: list[str] = ["<SIL>", "<MUSIC>", "<NOISE>", "<OTHER>"]
