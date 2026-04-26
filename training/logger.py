import logging
from typing import Optional

logger = logging.getLogger(__name__)


class TrainLogger:
    def __init__(
        self,
        backend: str = "tensorboard",
        project: str = "ru-moonshine",
        name: str = "run",
        config: Optional[dict] = None,
    ):
        self.backend = backend
        self._tb_writer = None
        self._wandb_run = None

        if backend == "wandb":
            import wandb

            self._wandb_run = wandb.init(
                project=project, name=name, config=config, reinit=True
            )
        elif backend == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter

            self._tb_writer = SummaryWriter(log_dir=f"runs/{name}")
        else:
            raise ValueError(f"Unknown logging backend: {backend}")

    def log(self, metrics: dict, step: int):
        if self.backend == "wandb":
            import wandb

            wandb.log(metrics, step=step)
        elif self.backend == "tensorboard":
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self._tb_writer.add_scalar(k, v, step)

    def log_summary(self, metrics: dict):
        if self.backend == "wandb":
            import wandb

            wandb.summary.update(metrics)

    def close(self):
        if self.backend == "wandb":
            import wandb

            wandb.finish()
        elif self._tb_writer is not None:
            self._tb_writer.close()
