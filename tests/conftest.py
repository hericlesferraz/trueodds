"""A tiny random ModernBERT and a word-level tokenizer: the model tests run on the CPU in seconds."""

import pytest

WORDS = [f"w{i}" for i in range(60)] + ["yes", "no", "maybe", "red", "blue", "green", "cat", "dog"]
SPECIALS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]


def make_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {t: i for i, t in enumerate(SPECIALS + WORDS)}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return PreTrainedTokenizerFast(
        tokenizer_object=tok,
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
    )


def make_model(pooling="cls", seed=0, architecture="cross"):
    import torch
    from transformers import ModernBertConfig

    from trueodds.model import DecisionModel

    torch.manual_seed(seed)
    config = ModernBertConfig(
        vocab_size=len(SPECIALS) + len(WORDS),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=512,
        pad_token_id=0,
        cls_token_id=2,
        sep_token_id=3,
        bos_token_id=2,
        eos_token_id=3,
        initializer_range=0.5,
    )
    model = DecisionModel.from_config(config, pooling, architecture=architecture)
    # The default init gives near-equal scores, so every question would get ~1/K and a test of
    # "the probabilities follow the options" could pass by accident. A wider head spreads them.
    torch.nn.init.normal_(model.head.weight, std=1.0)
    return model.eval()


@pytest.fixture
def tokenizer():
    return make_tokenizer()


@pytest.fixture
def encoder(tokenizer):
    from trueodds.encode import Encoder

    return Encoder(tokenizer, max_len=32)
