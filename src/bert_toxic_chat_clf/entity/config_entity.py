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