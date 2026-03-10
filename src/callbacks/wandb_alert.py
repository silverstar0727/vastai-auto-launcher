from typing import Literal

import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger

try:
    import wandb

    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


class WandbAlert(L.Callback):
    def __init__(self, monitor: str, mode: Literal["min", "max"]) -> None:
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.best_metric = float("inf") if mode == "min" else float("-inf")
        self.monitor_op = torch.lt if mode == "min" else torch.gt

    def on_train_epoch_end(self, trainer, pl_module):
        if not isinstance(trainer.logger, WandbLogger):
            return
        if self.monitor_op(trainer.callback_metrics[self.monitor], self.best_metric):
            self.best_metric = trainer.callback_metrics[self.monitor]
            val_loss = trainer.callback_metrics.get('val/loss', None)
            val_loss_str = f"\nval_loss={val_loss:.6f}" if val_loss is not None else ""
            trainer.logger.experiment.alert(
                title="Metric improved",
                text=f"{self.monitor}={self.best_metric:.4f}{val_loss_str}\nepoch={trainer.current_epoch}",
                wait_duration=1,
            )
