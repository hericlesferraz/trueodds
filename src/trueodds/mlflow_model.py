"""The checkpoint as an MLflow pyfunc model, so a registered version can be loaded and asked (D24).

Input: a DataFrame with one row per question and the columns `state`, `question` and `options` (a
list of strings). Output: one list of probabilities per row, in the order of its options. A preview
of Phase 4's `predict()`, which can load `models:/trueodds@champion` through this class.
"""

from __future__ import annotations

import mlflow.pyfunc

from trueodds.data.schema import Example


class TrueOddsModel(mlflow.pyfunc.PythonModel):
    def __getstate__(self) -> dict:
        # log_model calls load_context on this instance (to infer the signature) and pickles it
        # afterwards; without this, the pickle would hold a second copy of the weights.
        return {k: v for k, v in self.__dict__.items() if k != "predictor"}

    def load_context(self, context) -> None:
        from pathlib import Path

        from trueodds.predict import ModelPredictor

        self.predictor = ModelPredictor.load(Path(context.artifacts["checkpoint"]))

    def predict(self, context, model_input, params=None) -> list[list[float]]:
        examples = [
            Example(
                id=f"request/{i}",
                source="request",
                split="test",
                state=row["state"],
                question=row["question"],
                options=list(row["options"]),
                label_idx=0,  # unused: nothing is scored here
                template_id="native",
            )
            for i, row in enumerate(model_input.to_dict("records"))
        ]
        return [p.tolist() for p in self.predictor.predict(examples)]
