import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from interleave import get_span_idxs_binomial


class TestGetSpanIdxsBinomial:
    def test_deterministic_output_with_fixed_seed(self):
        result = get_span_idxs_binomial(n=39, p=0.1, seq_len=10, seed=42831)
        assert result[0] == 0, "First element should be 0"
        assert result[-1] == 10, "Last element should equal seq_len"
        for i in range(1, len(result)):
            assert result[i] > result[i - 1], f"Elements must be strictly increasing, but index {i - 1} -> {i} is not"

    def test_seq_len_one_returns_zero_and_one(self):
        result = get_span_idxs_binomial(n=39, p=0.1, seq_len=1, seed=42831)
        assert result == [0, 1]

    def test_seq_len_zero(self):
        result = get_span_idxs_binomial(n=39, p=0.1, seq_len=0, seed=42831)
        assert result == [0, 0] or result == [0]
