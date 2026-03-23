import pytest

from sardalign.utils.align import Segment, get_span_times, merge_repeats, normalize_uroman, time_to_frame, times_to_hubert_idxs


# ── normalize_uroman ─────────────────────────────────────────────────────────────────────────────────────────────────


class TestNormalizeUroman:
    def test_lowercases_text(self):
        assert normalize_uroman("Hello World") == "hello world"

    def test_strips_non_alpha_except_apostrophe_and_space(self):
        assert normalize_uroman("Hello World!") == "hello world"

    def test_preserves_apostrophe(self):
        assert normalize_uroman("it's") == "it's"

    def test_strips_digits(self):
        assert normalize_uroman("123abc") == "abc"

    def test_collapses_multiple_spaces(self):
        assert normalize_uroman("hello    world") == "hello world"

    def test_strips_leading_and_trailing_whitespace(self):
        assert normalize_uroman("  hello  ") == "hello"

    def test_empty_string(self):
        assert normalize_uroman("") == ""

    def test_only_special_characters(self):
        assert normalize_uroman("!@#$%^&*()") == ""

    def test_mixed_digits_and_alpha(self):
        assert normalize_uroman("a1b2c3") == "a b c"

    def test_multiple_apostrophes(self):
        assert normalize_uroman("don't won't") == "don't won't"


# ── time_to_frame ────────────────────────────────────────────────────────────────────────────────────────────────────


class TestTimeToFrame:
    def test_zero_time(self):
        assert time_to_frame(0) == 0

    def test_one_second(self):
        # stride_msec=20 => frames_per_sec = 1000/20 = 50
        assert time_to_frame(1.0) == 50

    def test_half_second(self):
        assert time_to_frame(0.5) == 25

    def test_small_fraction(self):
        # 0.02 seconds = 1 frame at 50 fps
        assert time_to_frame(0.02) == 1

    def test_large_time(self):
        assert time_to_frame(10.0) == 500

    def test_result_is_int(self):
        result = time_to_frame(1.0)
        assert isinstance(result, int)


# ── merge_repeats ────────────────────────────────────────────────────────────────────────────────────────────────────


class TestMergeRepeats:
    def test_simple_path(self):
        path = [0, 0, 1, 1, 1, 2]
        idx_to_token_map = {0: "a", 1: "b", 2: "c"}
        segments = merge_repeats(path, idx_to_token_map)

        assert len(segments) == 3

        assert segments[0].label == "a"
        assert segments[0].start == 0
        assert segments[0].end == 1

        assert segments[1].label == "b"
        assert segments[1].start == 2
        assert segments[1].end == 4

        assert segments[2].label == "c"
        assert segments[2].start == 5
        assert segments[2].end == 5

    def test_single_element(self):
        path = [0]
        idx_to_token_map = {0: "x"}
        segments = merge_repeats(path, idx_to_token_map)

        assert len(segments) == 1
        assert segments[0].label == "x"
        assert segments[0].start == 0
        assert segments[0].end == 0

    def test_no_repeats(self):
        path = [0, 1, 2]
        idx_to_token_map = {0: "a", 1: "b", 2: "c"}
        segments = merge_repeats(path, idx_to_token_map)

        assert len(segments) == 3
        for i, seg in enumerate(segments):
            assert seg.start == i
            assert seg.end == i

    def test_all_same(self):
        path = [0, 0, 0, 0]
        idx_to_token_map = {0: "a"}
        segments = merge_repeats(path, idx_to_token_map)

        assert len(segments) == 1
        assert segments[0].label == "a"
        assert segments[0].start == 0
        assert segments[0].end == 3

    def test_empty_path(self):
        segments = merge_repeats([], {})
        assert segments == []

    def test_returns_segment_objects(self):
        path = [0]
        idx_to_token_map = {0: "z"}
        segments = merge_repeats(path, idx_to_token_map)
        assert isinstance(segments[0], Segment)


# ── get_span_times ───────────────────────────────────────────────────────────────────────────────────────────────────


class TestGetSpanTimes:
    def test_basic_span(self):
        span = [Segment(label="a", start=0, end=5), Segment(label="b", start=5, end=10)]
        stride_ms = 20.0
        start_sec, end_sec = get_span_times(span, stride_ms)
        # start = 0 * 20 / 1000 = 0.0
        # end = 10 * 20 / 1000 = 0.2
        assert start_sec == pytest.approx(0.0)
        assert end_sec == pytest.approx(0.2)

    def test_single_segment_span(self):
        span = [Segment(label="x", start=50, end=100)]
        stride_ms = 20.0
        start_sec, end_sec = get_span_times(span, stride_ms)
        # start = 50 * 20 / 1000 = 1.0
        # end = 100 * 20 / 1000 = 2.0
        assert start_sec == pytest.approx(1.0)
        assert end_sec == pytest.approx(2.0)

    def test_different_stride(self):
        span = [Segment(label="a", start=0, end=100)]
        stride_ms = 10.0
        start_sec, end_sec = get_span_times(span, stride_ms)
        # start = 0 * 10 / 1000 = 0.0
        # end = 100 * 10 / 1000 = 1.0
        assert start_sec == pytest.approx(0.0)
        assert end_sec == pytest.approx(1.0)

    def test_nonzero_start(self):
        span = [Segment(label="a", start=25, end=50), Segment(label="b", start=50, end=75)]
        stride_ms = 20.0
        start_sec, end_sec = get_span_times(span, stride_ms)
        # start = 25 * 20 / 1000 = 0.5
        # end = 75 * 20 / 1000 = 1.5
        assert start_sec == pytest.approx(0.5)
        assert end_sec == pytest.approx(1.5)

    def test_returns_floats(self):
        span = [Segment(label="a", start=0, end=1)]
        start_sec, end_sec = get_span_times(span, 20.0)
        assert isinstance(start_sec, float)
        assert isinstance(end_sec, float)


# ── times_to_hubert_idxs ────────────────────────────────────────────────────────────────────────────────────────────


class TestTimesToHubertIdxs:
    def test_one_second_window(self):
        # (0.0, 1.0) with sr=16000, ds=320 => start = 0, end = ceil(16000/320) = 50
        start_idx, end_idx = times_to_hubert_idxs((0.0, 1.0), sampling_rate=16000, downsampling_ratio=320)
        assert start_idx == 0
        assert end_idx == 50

    def test_zero_duration(self):
        start_idx, end_idx = times_to_hubert_idxs((0.0, 0.0), sampling_rate=16000, downsampling_ratio=320)
        assert start_idx == 0
        assert end_idx == 0

    def test_offset_window(self):
        # (1.0, 2.0) => start = int(1.0 * 16000 / 320) = 50, end = ceil(2.0 * 16000 / 320) = 100
        start_idx, end_idx = times_to_hubert_idxs((1.0, 2.0), sampling_rate=16000, downsampling_ratio=320)
        assert start_idx == 50
        assert end_idx == 100

    def test_fractional_times(self):
        # (0.5, 1.5) => start = int(0.5 * 16000 / 320) = int(25.0) = 25
        #               end = ceil(1.5 * 16000 / 320) = ceil(75.0) = 75
        start_idx, end_idx = times_to_hubert_idxs((0.5, 1.5), sampling_rate=16000, downsampling_ratio=320)
        assert start_idx == 25
        assert end_idx == 75

    def test_end_uses_ceil(self):
        # Verify end_idx uses ceil: (0.0, 0.01) => end = ceil(0.01 * 16000 / 320) = ceil(0.5) = 1
        start_idx, end_idx = times_to_hubert_idxs((0.0, 0.01), sampling_rate=16000, downsampling_ratio=320)
        assert start_idx == 0
        assert end_idx == 1

    def test_start_uses_int_truncation(self):
        # Verify start_idx uses int (floor): (0.01, 0.02) => start = int(0.01 * 16000 / 320) = int(0.5) = 0
        start_idx, end_idx = times_to_hubert_idxs((0.01, 0.02), sampling_rate=16000, downsampling_ratio=320)
        assert start_idx == 0
        assert end_idx == 1

    def test_returns_ints(self):
        start_idx, end_idx = times_to_hubert_idxs((0.0, 1.0), sampling_rate=16000, downsampling_ratio=320)
        assert isinstance(start_idx, int)
        assert isinstance(end_idx, int)

    def test_different_sampling_rate(self):
        # sr=8000, ds=160 => frames_per_sec = 8000/160 = 50 (same ratio, same result)
        start_idx, end_idx = times_to_hubert_idxs((0.0, 1.0), sampling_rate=8000, downsampling_ratio=160)
        assert start_idx == 0
        assert end_idx == 50
