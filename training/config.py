from typing import Literal, Optional

from pydantic import BaseModel, Field


class OptimizerConfig(BaseModel):
    name: str = "adamw"
    lr: float = 5e-4
    weight_decay: float = 0.01
    lr_schedule: Literal["wsd", "noam", "cosine"] = "cosine"
    warmup_steps: int = 2000
    min_lr_ratio: float = 0.1
    decay_start_step: int = 150000
    decay_steps: int = 50000
    plateau_start_step: int = 0
    plateau_steps: int = 0
    post_decay_steps: int = 0
    final_lr: float = 0.0
    final_decay_start_step: int = 0
    final_decay_steps: int = 0


class AugmentationConfig(BaseModel):
    spec_augment: bool = False
    speed_perturbation: bool = False


class BatchingConfig(BaseModel):
    max_tokens: Optional[int] = None
    frames_per_sec: float = 41.0
    max_batch_size: int = 512
    min_batch_size: int = 4


class ValidationConfig(BaseModel):
    every_n_steps: int = 4000
    max_batches: int = 200
    escape_wer_patience: int = 0
    escape_wer_min_steps: int = 5000


class CheckpointingConfig(BaseModel):
    every_n_steps: int = 4000
    save_top_k: int = 5


class DataConfig(BaseModel):
    train_manifest: str = "data/manifests/train.jsonl"
    val_manifest: str = "data/manifests/val.jsonl"
    tokenizer_model: str = "data/tokenizer_256.model"


class LoggingConfig(BaseModel):
    backend: Literal["tensorboard", "wandb"] = "tensorboard"
    project: str = "ru-moonshine"
    name: str = "run"


class ThermalConfig(BaseModel):
    gpu_temp_warn: float = 85.0
    gpu_temp_crit: float = 90.0
    cpu_temp_warn: float = 85.0
    cpu_temp_crit: float = 95.0
    resume_temp: float = 70.0
    poll_interval_sec: float = 15.0


class TrainingConfig(BaseModel):
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)
    batching: BatchingConfig = Field(default_factory=BatchingConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    checkpointing: CheckpointingConfig = Field(default_factory=CheckpointingConfig)
    thermal: ThermalConfig = Field(default_factory=ThermalConfig)

    batch_size: int = 16
    val_batch_size: Optional[int] = None
    accum_steps: int = 4
    max_steps: int = 50000
    grad_clip: float = 5.0
    precision: Literal["fp16", "bf16", "fp32"] = "bf16"
    num_workers: int = 4
    val_num_workers: int = 2
    prefetch_factor: int = 2
    num_buckets: int = 100
    compile: bool = True
    log_every: int = 100
    nonfinite_patience: int = 5
    average_top_n: int = 5
    num_threads: int = 4
    seed: int = 42

    def get_val_batch_size(self) -> int:
        return self.val_batch_size if self.val_batch_size is not None else self.batch_size


class FullConfig(BaseModel):
    model: dict = {}
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "FullConfig":
        import yaml

        with open(path) as f:
            raw = yaml.safe_load(f)

        model_data = raw.get("model", {})
        train_data = raw.get("training", {})
        data_data = raw.get("data", {})
        log_data = raw.get("logging", {})

        opt_data = train_data.pop("optimizer", {})
        aug_data = train_data.pop("augmentation", {})
        batch_data = train_data.pop("batching", {})
        val_data = train_data.pop("validation", {})
        ckpt_data = train_data.pop("checkpointing", {})
        thermal_data = train_data.pop("thermal", {})

        return cls(
            model=model_data,
            training=TrainingConfig(
                optimizer=OptimizerConfig(**{k: v for k, v in opt_data.items() if k in OptimizerConfig.model_fields}),
                augmentation=AugmentationConfig(**{k: v for k, v in aug_data.items() if k in AugmentationConfig.model_fields}),
                batching=BatchingConfig(**{k: v for k, v in batch_data.items() if k in BatchingConfig.model_fields}),
                validation=ValidationConfig(**{k: v for k, v in val_data.items() if k in ValidationConfig.model_fields}),
                checkpointing=CheckpointingConfig(**{k: v for k, v in ckpt_data.items() if k in CheckpointingConfig.model_fields}),
                thermal=ThermalConfig(**{k: v for k, v in thermal_data.items() if k in ThermalConfig.model_fields}),
                **{k: v for k, v in train_data.items() if k in TrainingConfig.model_fields},
            ),
            data=DataConfig(**{k: v for k, v in data_data.items() if k in DataConfig.model_fields}),
            logging=LoggingConfig(**{k: v for k, v in log_data.items() if k in LoggingConfig.model_fields}),
        )
