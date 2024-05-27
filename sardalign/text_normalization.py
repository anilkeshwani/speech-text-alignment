import re
import unicodedata

from unidecode import unidecode

from sardalign.norm_config import norm_config


def text_normalize(
    text: str,
    iso_code: str,
    lower_case: bool = True,
    remove_numbers: bool = True,
    remove_brackets: bool = False,
) -> str:
    """Normalize text by lowercasing, removing punctuation and words containing only digits, and collapsing spaces.

    Args:
        text (str): String to be normalized
        iso_code (str): Language ISO code
        lower_case (bool, optional): Lowercase. Defaults to True.
        remove_numbers (bool, optional): Remove numbers. Defaults to True.
        remove_brackets (bool, optional): Remove brackets. Defaults to False.

    Returns:
        str: Normalized string
    """
    config = norm_config.get(iso_code, norm_config["*"])
    for field in ["lower_case", "punc_set", "del_set", "mapping", "digit_set", "unicode_norm"]:
        if field not in config:
            config[field] = norm_config["*"][field]
    text = unicodedata.normalize(config["unicode_norm"], text)
    if config["lower_case"] and lower_case:
        text = text.lower()
    # always text inside brackets with numbers in them. usually corresponds to "(sam 23:17)"
    text = re.sub(r"\([^\)]*\d[^\)]*\)", " ", text)
    if remove_brackets:
        text = re.sub(r"\([^\)]*\)", " ", text)
    # apply mappings
    for old, new in config["mapping"].items():
        text = re.sub(old, new, text)
    # replace punctutations with space
    punct_pattern = r"[" + config["punc_set"]
    punct_pattern += "]"
    normalized_text = re.sub(punct_pattern, " ", text)
    # remove characters in delete list
    delete_patten = r"[" + config["del_set"] + "]"
    normalized_text = re.sub(delete_patten, "", normalized_text)
    # Remove words containing only digits
    # We check for 3 cases:
    #   a) text starts with a number
    #   b) a number is present somewhere in the middle of the text
    #   c) the text ends with a number
    # For each case we use lookaround regex pattern to see if the digit pattern is preceded and
    # followed by whitespaces, only then we replace the numbers with space.
    # The lookaround enables overlapping pattern matches to be replaced
    if remove_numbers:
        digits_pattern = "[" + config["digit_set"]
        digits_pattern += "]+"
        complete_digit_pattern = (
            r"^" + digits_pattern + "(?=\s)|(?<=\s)" + digits_pattern + "(?=\s)|(?<=\s)" + digits_pattern + "$"
        )
        normalized_text = re.sub(complete_digit_pattern, " ", normalized_text)
    if config["rm_diacritics"]:
        normalized_text = unidecode(normalized_text)
    # remove extra spaces
    normalized_text = re.sub(r"\s+", " ", normalized_text).strip()
    return normalized_text
