import torch
from abc import ABC, abstractmethod
from torchmetrics import Accuracy, Precision, Recall, F1Score


class Metric(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def update(self, preds, labels):
        pass

    @abstractmethod
    def compute(self):
        pass

    @abstractmethod
    def to_string(self, key_value_fromat=False):
        pass

    @abstractmethod
    def save_model_metric(self):
        # the evaluation metric determines whether the model should be saved
        pass


class ImageClassificationMetric(Metric):
    def __init__(self, num_classes, eval_metric, device):
        self.result = {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
        }
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes, average="macro").to(device)
        self.precision = Precision(task="multiclass", num_classes=num_classes, average="macro").to(device)
        self.recall = Recall(task="multiclass", num_classes=num_classes, average="macro").to(device)
        self.f1_score = F1Score(task="multiclass", num_classes=num_classes, average="macro").to(device)

        assert eval_metric in self.result.keys(), f"Invalid eval_metric: {eval_metric}. Please choose from {self.result.keys()}."
        self.eval_metric = eval_metric

    def reset(self):
        self.accuracy.reset()
        self.precision.reset()
        self.recall.reset()
        self.f1_score.reset()

    def update(self, preds, labels):
        preds = torch.argmax(preds, dim=1)
        self.accuracy.update(preds, labels)
        self.precision.update(preds, labels)
        self.recall.update(preds, labels)
        self.f1_score.update(preds, labels)

    def compute(self):
        self.result["accuracy"] = self.accuracy.compute()
        self.result["precision"] = self.precision.compute()
        self.result["recall"] = self.recall.compute()
        self.result["f1_score"] = self.f1_score.compute()

    def to_string(self, key_value_fromat=False):
        if key_value_fromat:
            return {
                "accuracy": f"{self.result['accuracy']:.6f}",
                "precision": f"{self.result['precision']:.6f}",
                "recall": f"{self.result['recall']:.6f}",
                "f1_score": f"{self.result['f1_score']:.6f}",
            }
        else:
            return f"Accuracy: {self.result['accuracy']:.6f}, Precision: {self.result['precision']:.6f}, Recall: {self.result['recall']:.6f}, F1-Score: {self.result['f1_score']:.6f}"

    def save_model_metric(self):
        return self.result[self.eval_metric]
