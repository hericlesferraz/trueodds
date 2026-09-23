import pytest

from trueodds.data.schema import Example
from trueodds.encode import build_ids

CLS, SEP = 100, 101


def test_short_state_is_kept_whole():
    ids = build_ids([1, 2, 3], [4, 5], [6], CLS, SEP, max_len=32)
    assert ids == [CLS, 1, 2, 3, SEP, 4, 5, SEP, 6, SEP]


def test_long_state_is_cut_at_its_end_and_question_and_option_stay_whole():
    state = list(range(1000, 1100))
    question, option = [4, 5, 6], [7, 8]
    ids = build_ids(state, question, option, CLS, SEP, max_len=20)
    assert len(ids) == 20
    # 20 - 3 - 2 - 4 special tokens = 11 state tokens, the first 11
    assert ids == [CLS, *state[:11], SEP, *question, SEP, *option, SEP]


def test_empty_state_keeps_its_segment():
    assert build_ids([], [4], [6], CLS, SEP) == [CLS, SEP, 4, SEP, 6, SEP]


def test_question_and_option_that_do_not_fit_raise():
    with pytest.raises(ValueError, match="more than 10"):
        build_ids([1], [4] * 5, [6] * 2, CLS, SEP, max_len=10)


def test_encoder_truncates_each_option_to_its_own_budget(encoder):
    long_option = " ".join(["w1"] * 10)
    ex = Example(
        id="x/test/0",
        source="x",
        split="test",
        state=" ".join(f"w{i}" for i in range(40)),
        question="w50 w51 w52",
        options=["yes", long_option],
        label_idx=1,
        template_id="native",
    )
    [enc] = encoder.encode([ex])
    tok = encoder.tokenizer
    q = tok("w50 w51 w52", add_special_tokens=False)["input_ids"]
    for seq, option in zip(enc.sequences, ex.options, strict=True):
        o = tok(option, add_special_tokens=False)["input_ids"]
        assert len(seq) == encoder.max_len  # the state fills what is left
        assert seq[-len(o) - 1 : -1] == o  # the option is whole
        assert seq[-len(o) - len(q) - 2 : -len(o) - 2] == q  # the question is whole
        assert seq[1] == tok.convert_tokens_to_ids("w0")  # the state is cut at its end
    assert enc.k == 2 and enc.label_idx == 1 and enc.length == encoder.max_len
