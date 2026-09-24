"""`predict()`: ask the trained model about a state, get a probability per option (spec 003).

    import trueodds
    model = trueodds.load()      # the v1 checkpoint and its fitted temperature
    model.predict("The match ended 2-1.", "What is this text about?", ["Sports", "Business"])

A thin layer over `ModelPredictor`: the same scoring path as the harness, so the harness numbers
are the numbers `predict` gives.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from trueodds import paths
from trueodds.data.schema import Example
from trueodds.predict import ModelPredictor


def request_example(i: int, state: str, question: str, options: Sequence[str]) -> Example:
    """A question from a caller as an `Example`, checked by spec 001's rules (K >= 2, no duplicates)."""
    ex = Example(
        id=f"request/{i}",
        source="request",
        split="test",
        state=state,
        question=question,
        options=list(options),
        label_idx=0,  # unused: nothing is scored here
        template_id="native",
    )
    ex.validate()
    return ex


class TrueOdds:
    def __init__(self, predictor: ModelPredictor) -> None:
        self.predictor = predictor

    @property
    def temperature(self) -> float:
        return self.predictor.temperature

    def predict(self, state: str, question: str, options: Sequence[str]) -> dict[str, float]:
        """{option: probability}, in the order given, summing to 1."""
        return self.predict_batch(state, [(question, options)])[0]

    def predict_batch(
        self, state: str, questions: Sequence[tuple[str, Sequence[str]]]
    ) -> list[dict[str, float]]:
        """Many questions about one state in one call. A cross-encoder's sequences share
        micro-batches; a shared-state model reads the state once for all of them (spec 004)."""
        examples = [request_example(i, state, q, opts) for i, (q, opts) in enumerate(questions)]
        probs = self.predictor.predict(examples, pack=True)
        return [
            dict(zip(ex.options, map(float, p), strict=True))
            for ex, p in zip(examples, probs, strict=True)
        ]


def load(
    path: Path | str = paths.DEFAULT_CHECKPOINT, device: str | None = None, temperature: bool = True
) -> TrueOdds:
    """A checkpoint directory, at its fitted temperature unless `temperature=False`."""
    return TrueOdds(ModelPredictor.load(Path(path), device=device, temperature=temperature))
