import torch

from torcheval.metrics import (
    BinaryAUROC,
    BinaryRecall,
    BinaryF1Score,
    BinaryAccuracy,
    BinaryPrecision,
    BinaryConfusionMatrix,
    # MulticlassAUROC,
    # MulticlassRecall,
    # MulticlassAccuracy,
    # MulticlassPrecision,
)


class MetricCompute:
    f1: torch.Tensor
    auroc: torch.Tensor
    recall: torch.Tensor
    accuracy: torch.Tensor
    precision: torch.Tensor
    confusion_matrix: torch.Tensor

    def __init__(
        self,
        **kwargs,
    ):
        self.__dict__.update(kwargs)


class Metric:

    f1: BinaryF1Score
    auroc: BinaryAUROC
    recall: BinaryRecall
    accuracy: BinaryAccuracy
    precision: BinaryPrecision
    confusion_matrix: BinaryConfusionMatrix

    def __init__(self, device: torch.device):

        self.f1 = BinaryF1Score(device=device)
        self.auroc = BinaryAUROC(device=device)
        self.recall = BinaryRecall(
            device=device,
        )
        self.accuracy = BinaryAccuracy(device=device)
        self.precision = BinaryPrecision(device=device)
        self.confusion_matrix = BinaryConfusionMatrix(device=device)

    def update(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        self.f1.update(input=y_pred, target=y_true)
        self.auroc.update(input=y_pred, target=y_true)
        self.recall.update(input=y_pred, target=y_true)
        self.accuracy.update(input=y_pred, target=y_true)
        self.precision.update(input=y_pred, target=y_true)
        self.confusion_matrix.update(input=y_pred, target=y_true)

    def reset(self):
        self.f1.reset()
        self.auroc.reset()
        self.recall.reset()
        self.accuracy.reset()
        self.precision.reset()
        self.confusion_matrix.reset()

    def compute(self) -> MetricCompute:
        return MetricCompute(
            f1=self.f1.compute(),
            auroc=self.auroc.compute(),
            recall=self.recall.compute(),
            accuracy=self.accuracy.compute(),
            precision=self.precision.compute(),
            confusion_matrix=self.confusion_matrix.compute(),
        )
