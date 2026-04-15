import evaluate
import numpy as np
from datasets import Features, Value

_DESCRIPTION = "Simple accuracy metric"

_CITATION = """
@article{accuracy,
  title={Accuracy Metric},
  author={HuggingFace},
  year={2024}
}
"""

class Accuracy(evaluate.Metric):

    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            features=Features({
                "predictions": Value("int32"),
                "references": Value("int32"),
            }),
        )

    def _compute(self, predictions, references):
        predictions = np.array(predictions)
        references = np.array(references)
        return {"accuracy": float((predictions == references).mean())}