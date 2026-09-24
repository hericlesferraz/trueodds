"""Shared state encoding (D13, spec 004): the known answers, on a tiny random model on the CPU.

The tiny config has one full-attention and one sliding-window layer (window 64), and the states
here are longer than the window, so both masks are exercised.
"""

import random

import numpy as np
import pytest
import torch

import trueodds
from tests.conftest import WORDS, make_model
from trueodds.data.schema import Example
from trueodds.predict import ModelPredictor, predict_probs, predict_scores
from trueodds.shared import STATE, SharedEncoder, attention_masks, build_packed, state_budget

CLS, SEP = 2, 3


@pytest.fixture
def shared(tokenizer):
    return SharedEncoder(tokenizer, max_len=200)


@pytest.fixture
def model():
    return make_model(architecture="shared")


def example(i, state, question, options, label=0):
    return Example(
        id=f"t/test/{i}",
        source="t",
        split="test",
        state=state,
        question=question,
        options=options,
        label_idx=label,
        template_id="native",
    )


def about_one_state(n=5, seed=0):
    """`n` questions with K from 2 to 10 about one state of 120 words."""
    rng = random.Random(seed)
    state = " ".join(rng.choices(WORDS[:60], k=120))
    out = []
    for i in range(n):
        k = rng.choice([2, 3, 4, 10])
        options = [" ".join(rng.choices(WORDS, k=rng.randint(1, 3))) + f" w{j}" for j in range(k)]
        out.append(example(i, state, " ".join(rng.choices(WORDS, k=4)), options, label=k - 1))
    return out


def test_masks_equal_modernberts_own_on_an_unpacked_sequence():
    """With every token in the state, the masks are ModernBERT's: same hidden states."""
    enc = make_model().encoder
    torch.manual_seed(0)
    ids = torch.randint(4, 60, (3, 150))
    valid = torch.ones_like(ids)
    valid[1, 90:] = 0
    valid[2, 20:] = 0
    ids[valid == 0] = 0
    pos = torch.arange(150).expand(3, -1)
    state = torch.full_like(ids, STATE)
    masks = attention_masks(pos, state, state, valid, enc.config.sliding_window)
    with torch.no_grad():
        ours = enc(input_ids=ids, attention_mask=masks, position_ids=pos).last_hidden_state
        theirs = enc(input_ids=ids, attention_mask=valid).last_hidden_state
    real = valid.bool()
    torch.testing.assert_close(ours[real], theirs[real], atol=1e-5, rtol=1e-5)


def test_mask_structure():
    ids, pos, qid, oid, pools = build_packed(
        [10] * 70, [([20, 21], [[30], [31, 32]]), ([22], [[33], [34], [35]])], CLS, SEP
    )
    t = lambda xs: torch.tensor([xs])  # noqa: E731
    masks = attention_masks(t(pos), t(qid), t(oid), torch.ones(1, len(ids)), window=64)
    full = masks["full_attention"][0, 0]
    q, o = torch.tensor(qid), torch.tensor(oid)
    state = q == STATE
    assert not full[state][:, ~state].any()  # the state sees only the state
    assert full[:, state].all()  # everyone sees the state
    for i in range(len(ids)):
        seen = full[i]
        assert not seen[(q != q[i]) & ~state].any()  # no other question's tokens
        if o[i] != STATE:
            assert not seen[(q == q[i]) & (o != STATE) & (o != o[i])].any()  # no other option
    sliding = masks["sliding_attention"][0, 0]
    near = (t(pos)[0][:, None] - t(pos)[0][None, :]).abs() <= 64
    assert torch.equal(sliding, full & near)
    assert not sliding[pools[0][0], 0]  # position 0 is more than 64 from the options


def test_positions_restart_for_every_question_and_every_option():
    ids, pos, qid, _, pools = build_packed(
        [10] * 5, [([20, 21], [[30], [31, 32]]), ([22, 23], [[33]])], CLS, SEP
    )
    assert ids[:7] == [CLS, 10, 10, 10, 10, 10, SEP]
    starts = [i for i in range(len(ids)) if i > 0 and qid[i] != qid[i - 1]]
    assert [pos[i] for i in starts] == [7, 7]  # both questions start right after the state
    assert [pos[p] for pool in pools for p in pool] == [10, 10, 10]  # every option after "q [SEP]"
    assert all(ids[p] == CLS for pool in pools for p in pool)


def test_truncation_cuts_the_state_never_the_question_or_options():
    q, opts = [20] * 10, [[30] * 3, [31] * 7]
    assert state_budget(q, opts, max_len=40) == 40 - (10 + 7 + 4) - 1
    ids, pos, *_ = build_packed(list(range(100, 200)), [(q, opts)], CLS, SEP, max_len=40)
    assert ids[1:19] == list(range(100, 118))  # the state's start is kept
    assert ids.count(20) == 10 and ids.count(30) == 3 and ids.count(31) == 7
    assert max(pos) == 39  # the longest option's view fits in max_len positions
    with pytest.raises(ValueError, match="more than 40"):
        build_packed([], [([20] * 30, [[30] * 7])], CLS, SEP, max_len=40)


def test_a_question_scores_the_same_alone_and_packed(shared, model):
    # Equal up to fp32 rounding (~1e-6 relative): the masked rows have different lengths.
    questions = about_one_state()
    alone = predict_scores(model, shared, questions)
    packed = predict_scores(model, shared, questions, pack=True)
    assert len(shared.encode(questions, pack=True)) == 1  # one sequence for all five
    for a, p in zip(alone, packed, strict=True):
        np.testing.assert_allclose(a, p, rtol=1e-5, atol=1e-5)
    # and in any order: packing never lets one question see another
    reordered = predict_scores(model, shared, questions[::-1], pack=True)[::-1]
    for a, p in zip(alone, reordered, strict=True):
        np.testing.assert_allclose(a, p, rtol=1e-5, atol=1e-5)


def test_packing_keeps_different_states_apart(shared, model):
    first, second = about_one_state(3, seed=0), about_one_state(3, seed=1)
    mixed = [first[0], second[0], first[1], second[1], first[2]]
    items = shared.encode(mixed, pack=True)
    assert [p.questions for p in items] == [[0, 2, 4], [1, 3]]
    alone = predict_scores(model, shared, mixed)
    for a, p in zip(alone, predict_scores(model, shared, mixed, pack=True), strict=True):
        np.testing.assert_allclose(a, p, rtol=1e-5, atol=1e-5)


def test_changing_one_option_leaves_the_others_scores(shared, model):
    [ex] = about_one_state(1)
    other = example(0, ex.state, ex.question, [*ex.options[:-1], "cat dog red"])
    [before], [after] = predict_scores(model, shared, [ex]), predict_scores(model, shared, [other])
    np.testing.assert_allclose(before[:-1], after[:-1], atol=1e-5)
    assert not np.isclose(before[-1], after[-1])


def test_padded_options_get_zero_and_each_question_sums_to_one(shared, model):
    questions = about_one_state()
    batch = shared.collate(shared.encode(questions, pack=True))
    with torch.no_grad():
        probs = torch.softmax(model(**batch.model_inputs()), dim=-1)
    assert probs.shape == (5, max(ex.k for ex in questions))
    for row, ex in enumerate(questions):
        assert torch.all(probs[row, ex.k :] == 0)
        assert torch.all(probs[row, : ex.k] > 0)
        assert torch.isclose(probs[row].sum(), torch.tensor(1.0), atol=1e-6)


def test_shuffling_the_options_shuffles_the_probabilities(shared, model):
    rng = random.Random(1)
    questions = about_one_state()
    shuffled, perms = [], []
    for ex in questions:
        perm = list(range(ex.k))
        rng.shuffle(perm)
        perms.append(perm)
        shuffled.append(example(0, ex.state, ex.question, [ex.options[j] for j in perm]))
    before = predict_probs(model, shared, questions, pack=True)
    after = predict_probs(model, shared, shuffled, pack=True)
    for b, a, perm in zip(before, after, perms, strict=True):
        np.testing.assert_allclose(a, b[perm], atol=1e-6)


def test_checkpoint_keeps_the_architecture_and_predict_batch_packs(tmp_path, tokenizer):
    path = tmp_path / "run/best"
    make_model(architecture="shared").save(path, tokenizer, max_len=200)
    predictor = ModelPredictor.load(path, device="cpu")
    assert predictor.model.architecture == "shared"
    assert isinstance(predictor.encoder, SharedEncoder)
    model = trueodds.load(path, device="cpu")
    questions = about_one_state()
    pairs = [(ex.question, ex.options) for ex in questions]
    batch = model.predict_batch(questions[0].state, pairs)
    for (q, opts), out in zip(pairs, batch, strict=True):
        one = model.predict(questions[0].state, q, opts)
        np.testing.assert_allclose(list(out.values()), list(one.values()), atol=1e-5)


def test_a_checkpoint_without_an_architecture_is_a_cross_encoder(tmp_path, tokenizer):
    import json

    path = tmp_path / "run/best"
    make_model().save(path, tokenizer, max_len=32)
    meta = json.loads((path / "model.json").read_text())
    del meta["architecture"]  # as written before Phase 5
    (path / "model.json").write_text(json.dumps(meta))
    assert ModelPredictor.load(path, device="cpu").model.architecture == "cross"


def test_the_shared_model_overfits_a_known_set(shared):
    from trueodds.train import TrainConfig, fit

    rng = random.Random(0)
    colors = ["red", "blue", "green"]
    train = []
    for i in range(24):
        answer = rng.randrange(3)
        state = " ".join([*rng.choices(WORDS[:60], k=20), colors[answer]])
        train.append(example(i, state, "w1 w2", colors, label=answer))
    cfg = TrainConfig(
        run="tiny-shared",
        architecture="shared",
        lr=3e-3,
        warmup_ratio=0.1,
        max_steps=150,
        questions_per_step=6,
        eval_every=150,
        log_every=150,
        save=False,
    )
    model = make_model(architecture="shared").train()
    summary = fit(model, shared, train, {"train": train}, cfg)
    assert summary["final"]["train"]["accuracy"] == 1.0
