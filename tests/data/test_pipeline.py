from collections import Counter

import pytest

from trueodds.data import pipeline as p
from trueodds.data.schema import Example, load_examples, write_examples
from trueodds.data.templates import TRAINED


def ex(i, *, source="ag_news", split="train", state=None, label=0, template_id="topic:0",
       question="What is this text about?", options=("a", "b", "c"), vars=None):  # fmt: skip
    return Example(
        id=f"{source}/{split}/{i}",
        source=source,
        split=split,
        state=f"text {i}" if state is None else state,
        question=question,
        options=list(options),
        label_idx=label,
        template_id=template_id,
        vars=vars or {},
    )


def native(i, question, options=("x", "y"), **kw):
    return ex(i, template_id="native", question=question, options=options, state="", **kw)


def test_dedup_key_ignores_template_and_case():
    a = ex(1, state="Hello  World", template_id="topic:0")
    b = p.with_template(ex(1, state="hello world"), "2")
    assert a.question != b.question
    assert p.dedup_key(a) == p.dedup_key(b)
    assert p.dedup_key(a) != p.dedup_key(ex(2))


def test_dedup_key_sorts_options():
    assert p.dedup_key(native(1, "Q?", ("x", "y"))) == p.dedup_key(native(2, "q?", ("Y", "X")))


def test_dedup_drops_eval_overlap_and_train_duplicates():
    test_row = native(0, "Which is a planet?", source="arc_easy", split="test")
    train = {
        "mmlu_aux": [native(1, "which is a  planet?", source="mmlu_aux"), native(2, "Other?", source="mmlu_aux")],
        "arc_easy": [native(3, "Other?", source="arc_easy"), native(4, "New?", source="arc_easy")],
    }  # fmt: skip
    out, report = p.dedup_train(train, [test_row])
    assert [e.id for e in out["mmlu_aux"]] == ["mmlu_aux/train/2"]
    assert [e.id for e in out["arc_easy"]] == ["arc_easy/train/4"]
    assert report.overlap_with_eval == Counter({"mmlu_aux": 1})
    assert report.within_train == Counter({"arc_easy": 1})
    assert p.overlap([e for v in out.values() for e in v], [test_row]) == 0


def test_carve_moves_examples_and_renames_split():
    data = [ex(i) for i in range(100)]
    rest, dev = p.carve(data, 10, seed=0, split="dev")
    assert len(dev) == 10 and len(rest) == 90
    assert all(e.split == "dev" and e.id.startswith("ag_news/dev/") for e in dev)
    assert {e.state for e in dev}.isdisjoint(e.state for e in rest)
    assert p.carve(data, 10, seed=0, split="dev")[1] == dev  # deterministic


def test_carve_by_group_keeps_groups_whole():
    data = [ex(i, state=f"passage {i // 4}") for i in range(40)]  # 10 passages x 4 questions
    rest, test = p.carve(data, 6, seed=1, split="test", group=lambda e: e.state)
    assert len(test) == 8  # two whole passages reach 6
    assert {e.state for e in test}.isdisjoint(e.state for e in rest)


def test_dev_size():
    assert p.dev_size(9427) == 942
    assert p.dev_size(392702) == 2000


def test_stratified_sample_keeps_label_shares():
    data = [ex(i, label=0) for i in range(600)] + [ex(i, label=1) for i in range(600, 1000)]
    out = p.stratified_sample(data, 100, seed=0)
    assert len(out) == 100
    assert Counter(e.label_idx for e in out) == {0: 60, 1: 40}
    assert p.stratified_sample(data, 5000, seed=0) == data


def test_sample():
    data = [ex(i) for i in range(50)]
    assert len(p.sample(data, 10, seed=0)) == 10
    assert p.sample(data, 10, seed=0) == p.sample(data, 10, seed=0)
    assert p.sample(data, 99, seed=0) == data


def test_assign_templates():
    train = [ex(i) for i in range(200)]
    test = [ex(i, split="test") for i in range(200)]
    used = Counter(e.template_id for e in p.assign_templates(train, seed=0))
    assert set(used) == {f"topic:{n}" for n in TRAINED["topic"]}
    first = p.assign_templates(test, seed=0)
    assert first == p.assign_templates(test, seed=123)  # stable by id, not by seed
    assert all(not e.template_id.endswith("heldout") for e in first)
    held = p.heldout_template_copy([*first, native(0, "Q?", split="test")])
    assert len(held) == 200 and {e.template_id for e in held} == {"topic:heldout"}


def test_mnli_template_rendering():
    e = ex(0, template_id="mnli:0", question="x", options=("yes", "maybe", "no"),
           vars={"hypothesis": "The cat sat."})  # fmt: skip
    assert p.with_template(e, "0").question == "Does it follow that the cat sat?"
    assert (
        p.with_template(e, "heldout").question
        == 'Is the claim "The cat sat." supported by the text?'
    )


def words(texts):
    return [len(t.split()) for t in texts]


def test_length_filter_drops_and_counts_truncation():
    fits = ex(0, state="w " * 5, question="q q", options=("o", "o o"))
    cut = ex(1, state="w " * 20, question="q q", options=("o", "o o"))
    too_long = ex(2, state="", question="q " * 9, options=("o", "o o"))
    kept, report = p.length_filter([fits, cut, too_long], words, max_tokens=14)
    # fixed = question + longest option + 4 special tokens: 2 + 2 + 4 = 8
    assert [e.id for e in kept] == [fits.id, cut.id]
    assert (report.kept, report.dropped, report.truncated) == (2, 1, 1)
    assert report.truncated_share == pytest.approx(0.5)


def test_write_and_load_roundtrip(tmp_path):
    rows = [ex(0), native(1, "Q?")]
    path = tmp_path / "x" / "test.jsonl"
    assert write_examples(path, rows) == 2
    assert load_examples(path) == rows


def test_validation_rejects_bad_rows():
    with pytest.raises(ValueError, match="out of range"):
        ex(0, label=3).validate()
    with pytest.raises(ValueError, match="duplicate"):
        ex(0, options=("a", "a")).validate()
    with pytest.raises(ValueError, match="native"):
        ex(0, template_id="native", vars={"q": "x"}).validate()
