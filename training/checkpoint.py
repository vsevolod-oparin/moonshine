import json
import os
import signal
import torch
from pathlib import Path
from typing import Optional


def clean_state_dict(state_dict: dict) -> dict:
    return {k.replace("._orig_mod", ""): v for k, v in state_dict.items()}


class CheckpointManager:
    def __init__(
        self,
        save_dir: str,
        keep_top_k: int = 5,
        metric_mode: str = "min",
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.keep_top_k = keep_top_k
        self.metric_mode = metric_mode
        self._checkpoints: list[dict] = []
        self._load_index()
        self._preemption_handler = None

    def _index_path(self) -> Path:
        return self.save_dir / "index.json"

    def _load_index(self):
        path = self._index_path()
        if path.exists():
            with open(path) as f:
                self._checkpoints = json.load(f)

    def _save_index(self):
        with open(self._index_path(), "w") as f:
            json.dump(self._checkpoints, f, indent=2)

    def _is_better(self, current: float, best: float) -> bool:
        if self.metric_mode == "min":
            return current < best
        return current > best

    def save(
        self,
        model: torch.nn.Module,
        optimizer,
        scheduler,
        step: int,
        metric: float,
        scaler=None,
        path: Optional[str] = None,
    ) -> str:
        if path is None:
            path = str(self.save_dir / f"ckpt-step{step}.pt")

        state = {
            "step": step,
            "metric": metric,
            "model_state_dict": clean_state_dict(model.state_dict()),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        if scheduler is not None:
            state["scheduler_state_dict"] = scheduler.state_dict()
        if scaler is not None:
            state["scaler_state_dict"] = scaler.state_dict()

        try:
            state["rng_state"] = torch.random.get_rng_state()
            if torch.cuda.is_available():
                state["cuda_rng_state"] = torch.cuda.get_rng_state()
        except Exception:
            pass

        torch.save(state, path)

        self._checkpoints.append({"path": path, "step": step, "metric": metric})
        self._checkpoints.sort(
            key=lambda x: x["metric"],
            reverse=(self.metric_mode == "max"),
        )

        while len(self._checkpoints) > self.keep_top_k:
            worst = self._checkpoints.pop()
            if os.path.exists(worst["path"]):
                os.remove(worst["path"])

        self._save_index()
        return path

    def load_latest(
        self,
        model: torch.nn.Module,
        optimizer=None,
        scheduler=None,
        scaler=None,
        map_location=None,
    ) -> Optional[int]:
        latest_path = self.save_dir / "latest.pt"
        if not latest_path.exists():
            return None
        return self._load(
            str(latest_path), model, optimizer, scheduler, scaler, map_location
        )

    def load_best(
        self,
        model: torch.nn.Module,
        map_location=None,
    ) -> Optional[int]:
        if not self._checkpoints:
            return None
        best = self._checkpoints[0]
        if not os.path.exists(best["path"]):
            return None
        return self._load(best["path"], model, map_location=map_location)

    def _load(
        self,
        path: str,
        model: torch.nn.Module,
        optimizer=None,
        scheduler=None,
        scaler=None,
        map_location=None,
    ) -> int:
        state = torch.load(path, map_location=map_location, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in state:
            optimizer.load_state_dict(state["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in state:
            scheduler.load_state_dict(state["scheduler_state_dict"])
        if scaler is not None and "scaler_state_dict" in state:
            scaler.load_state_dict(state["scaler_state_dict"])
        if "rng_state" in state:
            torch.random.set_rng_state(state["rng_state"])
        if "cuda_rng_state" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state(state["cuda_rng_state"])
        return state["step"]

    def save_latest(self, model, optimizer, scheduler, step, scaler=None):
        path = str(self.save_dir / "latest.pt")
        state = {
            "step": step,
            "model_state_dict": clean_state_dict(model.state_dict()),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        if scheduler is not None:
            state["scheduler_state_dict"] = scheduler.state_dict()
        if scaler is not None:
            state["scaler_state_dict"] = scaler.state_dict()
        try:
            state["rng_state"] = torch.random.get_rng_state()
            if torch.cuda.is_available():
                state["cuda_rng_state"] = torch.cuda.get_rng_state()
        except Exception:
            pass
        torch.save(state, path)

    def install_preemption_handler(self, save_fn):
        def handler(signum, frame):
            save_fn()
            exit(143)

        self._preemption_handler = signal.signal(signal.SIGTERM, handler)

    def best_metric(self) -> Optional[float]:
        if self._checkpoints:
            return self._checkpoints[0]["metric"]
        return None

    @property
    def checkpoint_paths(self) -> list[str]:
        return [c["path"] for c in self._checkpoints]


def average_checkpoints(
    paths: list[str],
    output_path: str,
    map_location=None,
):
    if not paths:
        raise ValueError("No checkpoint paths provided")

    avg_state = None
    count = 0
    for path in paths:
        state = torch.load(path, map_location=map_location, weights_only=False)
        sd = state["model_state_dict"]
        if avg_state is None:
            avg_state = {k: v.float().clone() for k, v in sd.items()}
        else:
            for k, v in sd.items():
                avg_state[k].add_(v.float())
        count += 1

    for k in avg_state:
        avg_state[k].div_(count)

    torch.save({"model_state_dict": avg_state, "averaged_from": count}, output_path)
    return output_path
