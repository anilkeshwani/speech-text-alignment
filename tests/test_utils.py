from argparse import ArgumentTypeError
from pathlib import Path

import pytest

from sardalign.constants import PUA_PL_END, PUA_PL_START
from sardalign.utils import count_lines, dsu2pua, get_integer_sample_size, parse_arg_int_or_float, pua2dsu, read_jsonl, write_jsonl


# ──────────────────────────────────────────────────────────────────────────────
# dsu2pua / pua2dsu round-trip
# ──────────────────────────────────────────────────────────────────────────────


class TestDsu2PuaAndPua2Dsu:
    def test_round_trip_index_zero(self):
        pua_char = dsu2pua(0)
        assert pua2dsu(pua_char) == 0

    def test_round_trip_mid_range(self):
        mid = (PUA_PL_END - PUA_PL_START) // 2
        pua_char = dsu2pua(mid)
        assert pua2dsu(pua_char) == mid

    def test_round_trip_max_index(self):
        max_idx = PUA_PL_END - PUA_PL_START
        pua_char = dsu2pua(max_idx)
        assert pua2dsu(pua_char) == max_idx

    def test_dsu2pua_out_of_range_raises_runtime_error(self):
        out_of_range_idx = PUA_PL_END - PUA_PL_START + 1
        with pytest.raises(RuntimeError):
            dsu2pua(out_of_range_idx)

    def test_pua2dsu_below_range_raises_value_error(self):
        below_range_char = chr(PUA_PL_START - 1)
        with pytest.raises(ValueError):
            pua2dsu(below_range_char)

    def test_pua2dsu_above_range_raises_value_error(self):
        above_range_char = chr(PUA_PL_END + 1)
        with pytest.raises(ValueError):
            pua2dsu(above_range_char)

    def test_pua2dsu_multi_char_raises_value_error(self):
        with pytest.raises(ValueError):
            pua2dsu("ab")


# ──────────────────────────────────────────────────────────────────────────────
# count_lines
# ──────────────────────────────────────────────────────────────────────────────


class TestCountLines:
    def test_normal_file_three_lines(self, tmp_path: Path):
        f = tmp_path / "three_lines.txt"
        f.write_text("line1\nline2\nline3\n")
        assert count_lines(f) == 3

    def test_file_without_trailing_newline_raises_value_error(self, tmp_path: Path):
        f = tmp_path / "no_trailing_newline.txt"
        f.write_text("line1\nline2")
        with pytest.raises(ValueError, match="trailing newline"):
            count_lines(f)


# ──────────────────────────────────────────────────────────────────────────────
# get_integer_sample_size
# ──────────────────────────────────────────────────────────────────────────────


class TestGetIntegerSampleSize:
    def test_int_input_returns_same(self):
        assert get_integer_sample_size(10, N=100) == 10

    def test_float_input_computes_fraction(self):
        assert get_integer_sample_size(0.5, N=100) == 50

    def test_invalid_type_raises_type_error(self):
        with pytest.raises(TypeError):
            get_integer_sample_size("10", N=100)


# ──────────────────────────────────────────────────────────────────────────────
# parse_arg_int_or_float
# ──────────────────────────────────────────────────────────────────────────────


class TestParseArgIntOrFloat:
    def test_integer_string(self):
        result = parse_arg_int_or_float("42")
        assert result == 42
        assert isinstance(result, int)

    def test_float_string(self):
        result = parse_arg_int_or_float("3.14")
        assert result == pytest.approx(3.14)
        assert isinstance(result, float)

    def test_invalid_string_raises_argument_type_error(self):
        with pytest.raises(ArgumentTypeError):
            parse_arg_int_or_float("abc")


# ──────────────────────────────────────────────────────────────────────────────
# write_jsonl / read_jsonl round-trip
# ──────────────────────────────────────────────────────────────────────────────


class TestWriteAndReadJsonl:
    def test_round_trip(self, tmp_path: Path):
        records = [
            {"id": 1, "text": "hello world"},
            {"id": 2, "text": "foo bar"},
            {"id": 3, "nested": {"key": "value"}, "numbers": [1, 2, 3]},
        ]
        fpath = tmp_path / "data.jsonl"
        write_jsonl(fpath, records)
        result = read_jsonl(fpath)
        assert result == records
