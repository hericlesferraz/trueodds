"""The properties the model has by construction (D2), checked on a tiny random model."""

import random

import pytest
import torch

from tests.conftest import WORDS, make_model
from trueodds.batching import collate
from trueodds.data.schema import Example
from trueodds.model import DecisionModel
from trueodds.predict import predict_probs


def question(i, k, rng, label=0):
    words = WORDS[:60]
    return Example(
        id=f"t/test/{i}",
        source="t",
        split="test",
        state=" ".join(rng.choices(words, k=rng.randint(0, 20))),
        question=" ".join(rng.choices(words, k=3)),
        options=[" ".join(rng.choices(words, k=rng.randint(1, 3))) + f" w{j}" for j in range(k)],
        label_idx=label,
        template_id="native",
    )


@pytest.fixture
def mixed_k():
    rng = random.Random(0)
    # distinct options per question: each ends in its own w{j}
    return [question(i, k, rng) for i, k in enumerate([2, 4, 3, 2, 10])]


@pytest.mark.parametrize("pooling", ["cls", "mean"])
def test_padded_options_get_zero_and_each_question_sums_to_one(encoder, mixed_k, pooling):
    model = make_model(pooling)
    batch = collate(encoder.encode(mixed_k), encoder.pad)
    with torch.no_grad():
        probs = torch.softmax(model(**batch.model_inputs()), dim=-1)
    assert probs.shape == (5, 10)
    for q, ex in enumerate(mixed_k):
        assert torch.all(probs[q, ex.k :] == 0)  # exactly 0, not small
        assert torch.all(probs[q, : ex.k] > 0)
        assert torch.isclose(probs[q].sum(), torch.tensor(1.0), atol=1e-6)


@pytest.mark.parametrize("pooling", ["cls", "mean"])
def test_shuffling_the_options_shuffles_the_probabilities(encoder, mixed_k, pooling):
    model = make_model(pooling)
    rng = random.Random(1)
    shuffled, perms = [], []
    for ex in mixed_k:
        perm = list(range(ex.k))
        rng.shuffle(perm)
        perms.append(perm)
        options = [ex.options[j] for j in perm]
        shuffled.append(Example(**{**ex.__dict__, "options": options, "label_idx": perm.index(0)}))
    before = predict_probs(model, encoder, mixed_k)
    after = predict_probs(model, encoder, shuffled)
    assert all(p.max() - p.min() > 0.05 for p in before)  # not ~1/K, which any order would match
    for p, q, perm in zip(before, after, perms, strict=True):
        assert torch.allclose(torch.tensor(q), torch.tensor(p[perm]), atol=1e-5)


def test_a_question_scores_the_same_alone_and_in_a_batch(encoder, mixed_k):
    model = make_model()
    together = predict_probs(model, encoder, mixed_k)
    for ex, p in zip(mixed_k, together, strict=True):
        [alone] = predict_probs(model, encoder, [ex])
        assert torch.allclose(torch.tensor(alone), torch.tensor(p), atol=1e-5)


def test_loss_and_gradients_are_finite_with_padded_options(encoder, mixed_k):
    model = make_model().train()
    batch = collate(encoder.encode(mixed_k), encoder.pad)
    scores = model(**batch.model_inputs())
    loss = torch.nn.functional.cross_entropy(scores, batch.labels)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.head.weight.grad.abs().sum() > 0


def test_save_and_load_give_the_same_probabilities(encoder, mixed_k, tmp_path):
    model = make_model("mean")
    model.save(tmp_path / "ckpt", encoder.tokenizer, max_len=encoder.max_len)
    loaded = DecisionModel.load(tmp_path / "ckpt")
    assert loaded.pooling == "mean"
    before = predict_probs(model, encoder, mixed_k)
    after = predict_probs(loaded, encoder, mixed_k)
    for p, q in zip(before, after, strict=True):
        assert torch.allclose(torch.tensor(p), torch.tensor(q), atol=1e-6)


def test_unknown_pooling_is_refused():
    with pytest.raises(ValueError, match="pooling"):
        make_model("max")
