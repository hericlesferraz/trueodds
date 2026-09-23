"""Each converter on hand-made rows shaped like the Hugging Face rows (checked 2026-09-23)."""

import pytest

from trueodds.data import converters as c


def test_boolq():
    row = {"question": "do iran and afghanistan speak the same language", "answer": True,
           "passage": "  Persian is spoken in Iran and Afghanistan. "}  # fmt: skip
    ex = c.boolq(row, "train", 7)
    assert ex.id == "boolq/train/7"
    assert ex.state == "Persian is spoken in Iran and Afghanistan."
    assert ex.question == "Do iran and afghanistan speak the same language?"
    assert ex.options == ["yes", "no"] and ex.label_idx == 0
    assert ex.template_id == "boolq:0"
    assert ex.vars == {"question": "Do iran and afghanistan speak the same language?"}
    assert c.boolq({**row, "answer": False}, "train", 7).label_idx == 1


@pytest.mark.parametrize(("label", "option"), [(0, "yes"), (1, "maybe"), (2, "no")])
def test_mnli(label, option):
    row = {
        "premise": "The cat sat.",
        "hypothesis": "A cat was sitting. ",
        "label": label,
        "idx": 12,
    }
    ex = c.mnli(row, "test", 0)
    assert ex.id == "mnli/test/12"
    assert ex.state == "The cat sat."
    assert ex.question == "Does it follow that a cat was sitting?"
    assert ex.label == option


def test_mnli_unlabelled_is_skipped():
    assert c.mnli({"premise": "a", "hypothesis": "b", "label": -1, "idx": 0}, "test", 0) is None


def test_arc():
    row = {"id": "Mercury_1", "question": "Which is a planet?",
           "choices": {"text": ["Sun", "Mars", "Moon"], "label": ["1", "2", "3"]}, "answerKey": "2"}  # fmt: skip
    ex = c.arc_challenge(row, "test", 0)
    assert ex.id == "arc_challenge/test/Mercury_1"
    assert ex.state == "" and ex.question == "Which is a planet?"
    assert ex.options == ["Sun", "Mars", "Moon"] and ex.label == "Mars"
    assert ex.template_id == "native" and ex.vars == {}
    assert c.arc_easy({**row, "answerKey": "E"}, "test", 0) is None


def test_commonsense_qa():
    row = {"id": "abc", "question": "Where is a revolving door a security measure?",
           "question_concept": "revolving door",
           "choices": {"label": ["A", "B", "C", "D", "E"],
                       "text": ["bank", "library", "department store", "mall", "new york"]},
           "answerKey": "A"}  # fmt: skip
    ex = c.commonsense_qa(row, "test", 0)
    assert ex.source == "commonsense_qa" and ex.k == 5 and ex.label == "bank"


def test_hellaswag():
    row = {"ind": 24, "ctx": "A man is sitting on a roof. he", "label": "3",
           "source_id": "activitynet~v_-JhWjGDPHMY",
           "endings": ["is using wrap.", "is ripping tiles off.", "is holding a cube.",
                       "starts pulling up roofing."]}  # fmt: skip
    ex = c.hellaswag(row, "train", 0)
    assert ex.id == "hellaswag/train/24"
    assert ex.state == "A man is sitting on a roof. he"
    assert ex.question == "What happens next?" and ex.template_id == "hellaswag:0"
    assert ex.label == "starts pulling up roofing."
    assert c.hellaswag({**row, "label": ""}, "test", 0) is None
    assert c.hellaswag({**row, "source_id": "wikihow~12345"}, "train", 0) is None  # D17


def test_repeated_options_are_merged():
    row = {"id": "q", "question": "Which?", "answerKey": "E",
           "choices": {"label": list("ABCDE"), "text": ["mortal", "dying", "death", "dead", "mortal"]}}  # fmt: skip
    ex = c.commonsense_qa(row, "test", 0)
    assert ex.options == ["mortal", "dying", "death", "dead"] and ex.label == "mortal"
    row = {"ind": 1, "ctx": "x", "label": "2", "endings": ["a", "a", "b", "c"]}
    ex = c.hellaswag(row, "train", 0)
    assert ex.options == ["a", "b", "c"] and ex.label == "b"
    assert c.hellaswag({**row, "endings": ["a", "a", "a", "a"], "label": "0"}, "train", 0) is None


def test_mmlu_aux_with_passage():
    passage = "Tom went to the market. " * 20
    row = {"train": {"question": passage + "Where did Tom go?   _  .", "subject": "",
                     "choices": ["home", "market", "school", "park"], "answer": 1}}  # fmt: skip
    ex = c.mmlu_aux(row, "train", 5)
    assert ex.id == "mmlu_aux/train/5"
    assert ex.state == passage.strip()
    assert ex.question == "Where did Tom go? _ ."
    assert ex.label == "market" and ex.template_id == "native"


def test_mmlu_aux_short_text_is_a_question():
    row = {"train": {"question": "What is 2 + 2? Pick one.", "choices": ["3", "4"], "answer": 1}}
    ex = c.mmlu_aux(row, "train", 0)
    assert ex.state == "" and ex.question == "What is 2 + 2? Pick one."


@pytest.mark.parametrize(
    "question",
    ["How does Mr. Cool manage to travel so fast?", "Why did J. K. Rowling write it?",
     "What did the U.S. team win?"],
)  # fmt: skip
def test_split_passage_keeps_abbreviations_in_the_question(question):
    passage, q = c.split_passage("B" * 300 + ". The end came. " + question)
    assert q == question and passage.endswith("The end came.")


def test_split_passage_skips_abbreviations():
    text = "A" * 300 + ". Then Mr. Smith left. What did Mr. Smith do?"
    passage, question = c.split_passage(text)
    assert question == "What did Mr. Smith do?"
    assert passage.endswith("Then Mr. Smith left.")


def test_ag_news():
    ex = c.ag_news({"text": "Unions are disappointed.", "label": 2}, "test", 3)
    assert ex.id == "ag_news/test/3"
    assert ex.options == c.AG_NEWS_LABELS and ex.label == "Business"
    assert ex.question == "What is this text about?" and ex.template_id == "topic:0"


def test_yahoo():
    row = {"id": 0, "topic": 8, "question_title": "What makes friendship click?",
           "question_content": "", "best_answer": "Good communication."}  # fmt: skip
    ex = c.yahoo(row, "test", 0)
    assert ex.state == "What makes friendship click?\nGood communication."
    assert ex.label == "Family & Relationships" and ex.k == 10


def test_dbpedia():
    row = {"label": 0, "title": "TY KU", "content": " TY KU is a beverage company."}
    ex = c.dbpedia(row, "test", 0)
    assert ex.state == "TY KU\nTY KU is a beverage company."
    assert ex.label == "Company" and ex.k == 14


def test_sources_registry():
    assert set(c.SOURCES) == {"boolq", "mnli", "arc_easy", "arc_challenge", "hellaswag",
                              "mmlu_aux", "ag_news", "yahoo", "commonsense_qa", "dbpedia"}  # fmt: skip
    heldout = {s.name for s in c.SOURCES.values() if s.role == "heldout"}
    assert heldout == {"commonsense_qa", "dbpedia"}
    for s in c.SOURCES.values():
        assert "test" in s.splits or s.name == "mmlu_aux"  # mmlu_aux carves its test (D14)
        assert (s.role == "train") == ("train" in s.splits)
