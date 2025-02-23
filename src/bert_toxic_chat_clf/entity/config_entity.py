from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path

@dataclass(frozen=True)
class SetupModelConfig:
    root_dir: Path
    model_path: Path
    params_classes: int
    params_dropout: float

@dataclass
class TrainingConfig:
    root_dir: Path
    model_path: Path
    trained_model_path: Path
    training_data_path: Path
    params_train_size: float
    params_max_len: int
    params_train_batch_size: int
    params_valid_batch_size: int
    params_epochs: int
    params_learning_rate: float
    params_train_num_workers: int
    params_valid_num_workers: int
    params_train_shuffle: bool
    params_valid_shuffle: bool