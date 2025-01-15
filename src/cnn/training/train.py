from sklearn.metrics import roc_auc_score
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader
import torch
from typing import Any, Generator, Type
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F


@dataclass
class LogPoint:
    X: Tensor
    y: Tensor
    y_hat: Tensor
    loss: Tensor
    batch_size: int

    def __str__(self) -> str:
        string: str = "LogPoint@\u007d"
        string = f"{string} loss                : {str(self.loss)}"
        string = f"{string} batch_size          : {str(self.batch_size)}"
        string = f"{string}\u007d\n"
        return string

    def __repr__(self) -> str:
        return self.__str__()

    def __hash__(self) -> int:
        return hash(f"{self.X}{self.y}{self.y_hat}{self.loss}")

    def __eq__(self, other):
        return hash(self) == hash(other)


@dataclass
class EpochLogs:
    model: nn.Module
    optimiser: optim.Optimizer
    epoch: int
    train_logs: list[LogPoint]
    val_logs: list[LogPoint]

    def __str__(self) -> str:

        string_train_logs: str = str(list(map(str, self.train_logs)))
        string_val_logs: str = str(list(map(str, self.val_logs)))

        string: str = "EpochLogs\u007d"
        string = f"{string} epoch: {str(self.epoch)},"
        string = f"{string} train_logs: {string_train_logs},"
        string = f"{string} val_logs: {string_val_logs}"
        string = f"{string}\u007d\n"

        return string

    def __repr__(self) -> str:
        return self.__str__()


class EvaluationMetric:
    def __init__(
        self,
        tp: int,
        fp: int,
        tn: int,
        fn: int,
        auc: float | None = None
    ):
        self.tp: int = tp
        self.fp: int = fp
        self.tn: int = tn
        self.fn: int = fn
        self.n: int = tp + fp + tn + fn
        self.auc: float | None = auc

        self.tp_percentage: float = self.tp/self.n
        self.fp_percentage: float = self.fp/self.n
        self.tn_percentage: float = self.tn/self.n
        self.fn_percentage: float = self.fn/self.n

    @classmethod
    def from_prediction(
        cls,
        y_hat_prob: torch.Tensor,
        y: torch.Tensor
    ) -> 'EvaluationMetric':
        """
        Create an EvaluationMetric object from predictions and ground truth.
        """

        y_hat_prob = torch.nn.functional.softmax(y_hat_prob, dim=-1)

        # Get predicted class indices
        y_hat_max: torch.Tensor = torch.argmax(y_hat_prob, dim=-1)
        y_max = torch.argmax(y, dim=-1)          # True class (index of max value)

        # Initialize counters for TP, FP, TN, FN
        tp = (y_hat_max == y_max) & (y_max == 1)  # True Positives (predicted class matches true class)
        tn = (y_hat_max == y_max) & (y_max == 0)  # True Negatives (predicted and true class both 0)
        fp = (y_hat_max != y_max) & (y_hat_max == 1)  # False Positives (predicted class is 1 but true is 0)
        fn = (y_hat_max != y_max) & (y_hat_max == 0)  # False Negatives (predicted class is 0 but true is 1)

        # Sum the number of TP, TN, FP, FN
        tp = tp.sum().item()
        tn = tn.sum().item()
        fp = fp.sum().item()
        fn = fn.sum().item()
        auc: float

        try:
            auc = roc_auc_score(
                y.detach().cpu().numpy(),
                y_hat_prob.detach().cpu().numpy()
            )
        except Exception:
            auc = -1.0

        return cls(tp=tp, fp=fp, tn=tn, fn=fn, auc=auc)

    @property
    def precision(self) -> float:
        """
        Precision = TP / (TP + FP)
        """
        if self.tp + self.fp == 0:
            return 0.0
        return self.tp / (self.tp + self.fp)

    @property
    def recall(self) -> float:
        """
        Recall = TP / (TP + FN)
        """
        if self.tp + self.fn == 0:
            return 0.0
        return self.tp / (self.tp + self.fn)

    @property
    def accuracy(self) -> float:
        """
        Accuracy = (TP + TN) / Total
        """
        return (self.tp + self.tn) / self.n

    @property
    def f1_score(self) -> float:
        """
        F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
        """
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * (self.precision * self.recall) / (self.precision + self.recall)

    @property
    def specificity(self) -> float:
        """
        Specificity = TN / (TN + FP)
        """
        if self.tn + self.fp == 0:
            return 0.0
        return self.tn / (self.tn + self.fp)

    @property
    def false_positive_rate(self) -> float:
        """
        False Positive Rate = FP / (FP + TN)
        """
        if self.fp + self.tn == 0:
            return 0.0
        return self.fp / (self.fp + self.tn)

    def as_dict(self) -> dict[str, int | float]:
        """
        Return all metrics as a dictionary for easy logging or export.
        """
        return {
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "n": self.n,
            "tp_percentage": self.tp_percentage,
            "fp_percentage": self.fp_percentage,
            "tn_percentage": self.tn_percentage,
            "fn_percentage": self.fn_percentage,
            "precision": self.precision,
            "recall": self.recall,
            "accuracy": self.accuracy,
            "f1_score": self.f1_score,
            "specificity": self.specificity,
            "false_positive_rate": self.false_positive_rate,
            "auc": self.auc,
        }

    def __str__(self):
        """
        Return a string representation of the metrics.
        """
        metrics = self.as_dict()
        return "\n".join(
            f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}" for key, value in metrics.items())


def evaluate(
    model: nn.Module,
    criterion: nn.Module,
    eval_dataloader: DataLoader,
    device: str = "cpu"
) -> Generator[
    LogPoint,
    None,
    None
]:
    model = model.to(device=device)
    model = model.eval()

    X: Tensor
    y: Tensor

    for X, y in eval_dataloader:

        X = X.to(device=device)
        y = y.to(device=device)

        y_hat: Tensor = model.forward(X)

        loss: Tensor = criterion(y_hat, y)

        yield LogPoint(
            X=None,
            y=y.detach().cpu(),
            y_hat=y_hat.detach().cpu(),
            loss=loss.detach().cpu(),
            batch_size=X.shape[0]
        )


def train(
    model: nn.Module,
    optimiser: optim.Optimizer,
    criterion: nn.Module,
    train_dataloader: DataLoader,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Generator[
    LogPoint,
    None,
    None
]:

    model = model.to(device=device)
    model = model.train(True)
    optimiser.zero_grad()

    X: Tensor
    y: Tensor
    for X, y in train_dataloader:

        torch.cuda.empty_cache()

        X = X.to(device=device)
        y = y.to(device=device)

        y_hat: Tensor = model.forward(X)

        pred: Tensor = F.softmax(y_hat, dim=-1)

        loss: Tensor = criterion(y_hat, y)

        loss.backward()

        optimiser.step()
        optimiser.zero_grad()

        yield LogPoint(
            X=None,
            y=y.detach().cpu(),
            y_hat=y_hat.detach().cpu(),
            loss=loss.detach().cpu(),
            batch_size=X.shape[0]
        )


def grid_search(
    model_factor: Type[nn.Module],
    all_model_parameters: list[dict[str, Any]],
    optim_factory: Type[optim.Optimizer],
    all_optim_params: list[dict[str, Any]],
    epochs: int,
    criterion: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    lr_decay_window_size: int = 10,
    lr_decay_minimum: float = 10**(-7),
    scheduler_scale: float = 0.5,
    device: str = "cpu",
    compile_model: bool = False
) -> Generator[
    EpochLogs,
    None,
    None
]:

    optimiser_params: dict[str, Any]
    for optimiser_params in all_optim_params:
        model_parameters: dict[str, Any]
        for model_parameters in all_model_parameters:

            model: nn.Module = model_factor(
                **model_parameters
            )
            model = model.train()

            if compile_model:
                model = torch.compile(model)

            model_optimiser: optim.Optimizer
            model_optimiser = optim_factory(
                params=model.parameters(),
                **optimiser_params
            )

            loss_window: list[float] = []

            for epoch in range(epochs):
                epoch_train_logs: list[LogPoint] = []
                epoch_val_logs: list[LogPoint] = []
                epoch_cur_loss: np.floating = 0.0

                optimiser_loss: float = model_optimiser.param_groups[0]["lr"]

                train_log: LogPoint
                for train_log in tqdm(
                    train(
                        model=model,
                        optimiser=model_optimiser,
                        criterion=criterion,
                        train_dataloader=train_dataloader,
                        device=device
                    ),
                    desc="Training model...",
                    total=len(train_dataloader)
                ):
                    epoch_train_logs.append(train_log)
                    epoch_cur_loss += np.sum(
                        train_log.loss.detach(
                        ).cpu().tolist()
                    )

                loss_window.append(epoch_cur_loss)

                if optimiser_loss < lr_decay_minimum:
                    break

                if len(loss_window) > lr_decay_window_size:

                    if np.argmin(loss_window) == 0:
                        for param_group in model_optimiser.param_groups:
                            param_group['lr'] *= scheduler_scale

                        loss_window = [loss_window[-1]]

                    loss_window.pop(0)

                val_log: LogPoint
                for val_log in tqdm(
                    iterable=evaluate(
                        model=model,
                        criterion=criterion,
                        eval_dataloader=val_dataloader,
                        device=device
                    ),
                    desc="Validating Model...",
                    total=len(val_dataloader)
                ):
                    epoch_val_logs.append(val_log)

                yield EpochLogs(
                    model=model,
                    optimiser=model_optimiser,
                    epoch=epoch,
                    train_logs=epoch_train_logs,
                    val_logs=epoch_val_logs
                )
