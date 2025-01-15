from sklearn.metrics import auc
from torch import nn, optim, Tensor, save, float32
from torch.utils.data import DataLoader
from torch.cuda import empty_cache
from typing import Any, Generator, Type
from dataclasses import dataclass
import math
import numpy as np
from tqdm import tqdm


@dataclass
class LogPoint:
    X: Tensor
    y: Tensor
    y_hat: Tensor
    loss: Tensor
    threshold_data: 'ThresholdData'
    batch_size: int

    def __str__(self) -> str:
        string: str = "TrainLog@\u007d"
        string = f"{string} loss        : {str(self.loss)}"
        string = f"{string} threshold: {str(self.threshold_data)}"
        string = f"{string} batch_size  : {str(self.batch_size)}"
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


@dataclass
class ThresholdData:
    thresholds: list[float]
    accuracies_counts: list[int]
    accuracies_percentage: list[float]
    thresholds_tp: list[int]
    thresholds_tn: list[int]
    thresholds_fp: list[int]
    thresholds_fn: list[int]

    @staticmethod
    def calculate_auc(
        threshold_datas: list['ThresholdData']
    ) -> float:

        all_tp_batches: list[int] = [
            threshold_data.thresholds_tp
            for threshold_data in threshold_datas
        ]

        all_fp_batches: list[int] = [
            threshold_data.thresholds_fp
            for threshold_data in threshold_datas
        ]

        all_tn_batches: list[int] = [
            threshold_data.thresholds_tn
            for threshold_data in threshold_datas
        ]

        all_fn_batches: list[int] = [
            threshold_data.thresholds_fn
            for threshold_data in threshold_datas
        ]

        tp_per_threshold: list[int] = [
            sum(threshold_tp)
            for threshold_tp in zip(*all_tp_batches)
        ]
        fp_per_threshold: list[int] = [
            sum(threshold_fp)
            for threshold_fp in zip(*all_fp_batches)
        ]
        tn_per_threshold: list[int] = [
            sum(threshold_tn)
            for threshold_tn in zip(*all_tn_batches)
        ]
        fn_per_threshold: list[int] = [
            sum(threshold_fn)
            for threshold_fn in zip(*all_fn_batches)
        ]

        true_positive_rates: list[float] = [
            tp / (tp + fn)
            if (tp + fn) > 0
            else 0
            for tp, fn in zip(
                tp_per_threshold,
                fn_per_threshold
            )
        ]

        false_positive_rates: list[float] = [
            fp / (fp + tn)
            if (fp + tn) > 0
            else 0 for fp, tn in
            zip(
                fp_per_threshold,
                tn_per_threshold
            )
        ]

        return auc(
            false_positive_rates,
            true_positive_rates
        )

    @staticmethod
    def from_prediction(
        y_hat_softmaxed: Tensor,
        y: Tensor,
        thresholds: list[float]
    ) -> 'ThresholdData':

        accuracies_counts: list[int] = []
        accuracies_percentage: list[float] = []
        thresholds_tp: list[int] = []
        thresholds_tn: list[int] = []
        thresholds_fp: list[int] = []
        thresholds_fn: list[int] = []

        threshold: float
        for threshold in thresholds:
            # Binarise predictions based on threshold
            y_hat_i: Tensor = (y_hat_softmaxed > threshold).to(
                dtype=float32
            )

            # Count total pixels
            total_pixels = math.prod(
                y_hat_i.shape
            )

            # Calculate accuracy
            matches: Tensor = (y_hat_i == y)
            accuracy_count: int = matches.sum().item()
            accuracies_counts.append(accuracy_count)
            accuracies_percentage.append(accuracy_count / total_pixels)

            # Calculate TP, TN, FP, and FN
            tp = ((y_hat_i == 1) & (y == 1)).sum().item()  # True Positives
            tn = ((y_hat_i == 0) & (y == 0)).sum().item()  # True Negatives
            fp = ((y_hat_i == 1) & (y == 0)).sum().item()  # False Positives
            fn = ((y_hat_i == 0) & (y == 1)).sum().item()  # False Negatives

            thresholds_tp.append(tp)
            thresholds_tn.append(tn)
            thresholds_fp.append(fp)
            thresholds_fn.append(fn)

        return ThresholdData(
            thresholds=thresholds,
            accuracies_counts=accuracies_counts,
            accuracies_percentage=accuracies_percentage,
            thresholds_tp=thresholds_tp,
            thresholds_tn=thresholds_tn,
            thresholds_fp=thresholds_fp,
            thresholds_fn=thresholds_fn
        )


def evaluate(
    model: nn.Module,
    criterion: nn.Module,
    eval_dataloader: DataLoader,
    device: str = "cpu",
    thresholds: list[float] = [
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9
    ]
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

        loss.backward()

        threshold_data: ThresholdData
        threshold_data = ThresholdData.from_prediction(
            y_hat_softmaxed=y_hat,
            y=y,
            thresholds=thresholds
        )

        yield LogPoint(
            X=X,
            y=y,
            y_hat=y_hat,
            loss=loss,
            threshold_data=threshold_data,
            batch_size=X.shape[0]
        )


def train(
    model: nn.Module,
    optimiser: optim.Optimizer,
    criterion: nn.Module,
    train_dataloader: DataLoader,
    device: str = "cpu",
    thresholds: list[float] = [
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9
    ]
) -> Generator[
    LogPoint,
    None,
    None
]:

    model = model.to(device=device)
    model = model.train(True)

    X: Tensor
    y: Tensor
    for X, y in train_dataloader:

        optimiser.zero_grad()
        empty_cache()

        X = X.to(device=device)
        y = y.to(device=device)

        y_hat: Tensor = model.forward(X)

        loss: Tensor = criterion(y_hat, y)

        loss.backward()

        optimiser.step()

        threshold_data: ThresholdData
        threshold_data = ThresholdData.from_prediction(
            y_hat_softmaxed=y_hat,
            y=y,
            thresholds=thresholds
        )

        yield LogPoint(
            X=X,
            y=y,
            y_hat=y_hat,
            loss=loss,
            threshold_data=threshold_data,
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
    device: str = "cpu",
) -> Generator[
    EpochLogs,
    None,
    None
]:

    optimiser_params: dict[str, Any]
    for optimiser_params in all_optim_params:
        model_parameters: dict[str, Any]
        for model_parameters in tqdm(
            iterable=all_model_parameters,
            desc="Grid Search...",
            total=len(all_model_parameters),
        ):

            model: nn.Module = model_factor(
                **model_parameters
            )
            model = model.train()
            model_optimiser: optim.Optimizer
            model_optimiser = optim_factory(
                params=model.parameters(),
                **optimiser_params
            )

            loss_window: list[float] = []

            for epoch in tqdm(
                iterable=range(epochs),
                desc="Iterating Epochs...",
                total=epochs
            ):
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
                    total=len(train_dataloader.dataset)//4
                ):
                    epoch_train_logs.append(train_log)
                    epoch_cur_loss += np.sum(train_log.loss.detach().cpu().tolist())

                loss_window.append(epoch_cur_loss)

                if optimiser_loss < lr_decay_minimum:
                    break

                if len(loss_window) > lr_decay_window_size:

                    if np.argmin(loss_window) == 0:
                        for param_group in model_optimiser.param_groups:
                            param_group['lr'] *= 0.5

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
                    total=len(val_dataloader.dataset)//4
                ):
                    epoch_val_logs.append(val_log)

                yield EpochLogs(
                    model=model,
                    optimiser=model_optimiser,
                    epoch=epoch,
                    train_logs=epoch_train_logs,
                    val_logs=epoch_val_logs
                )
