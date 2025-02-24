from bert_toxic_chat_clf.constants import *
from bert_toxic_chat_clf.utils.common import read_yaml, create_directories
from bert_toxic_chat_clf.entity.config_entity import (
    DataIngestionConfig,
    DataSplitConfig,
    SetupModelConfig,
    TrainingConfig,
    EvaluationConfig
)


class ConfigurationManager:
    
    def __init__(self, config_filepath = CONFIG_FILE_PATH, params_filepath = PARAMS_FILE_PATH):
        
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])
    

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        
        config = self.config.data_ingestion 

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file
        )

        return data_ingestion_config

    
    def get_data_split_config(self) -> DataSplitConfig:

        config = self.config.data_split

        create_directories([config.root_dir, config.train_dir, config.test_dir])

        data_split_config = DataSplitConfig(
            root_dir=Path(config.root_dir),
            data_path=Path(self.config.data_ingestion.local_data_file),
            train_dir=Path(config.train_dir),
            train_data_path=Path(config.train_data_path),
            test_dir=Path(config.test_dir),
            test_data_path=Path(config.test_data_path),
            params_train_size=self.params.TRAIN_SIZE
        )

        return data_split_config
    
    def get_setup_model_config(self) -> SetupModelConfig:
        
        config = self.config.setup_model

        create_directories([config.root_dir])

        setup_model_config = SetupModelConfig(
            root_dir=Path(config.root_dir),
            model_path=Path(config.model_path),
            params_classes=self.params.CLASSES,
            params_dropout=self.params.DROPOUT,
        )

        return setup_model_config

    
    def get_training_config(self) -> TrainingConfig:
        
        training = self.config.training
        setup_model = self.config.setup_model

        params = self.params
        
        create_directories([Path(training.root_dir)])

        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            model_path=Path(setup_model.model_path),
            trained_model_path=Path(training.trained_model_path),
            training_data_path=Path(self.config.data_split.train_data_path),
            params_max_len=params.MAX_LEN,
            params_train_batch_size=params.TRAIN_BATCH_SIZE,
            params_epochs=params.EPOCHS,
            params_learning_rate=params.LEARNING_RATE,
            params_train_num_workers=params.TRAIN_NUM_WORKERS,
            params_train_shuffle=params.TRAIN_SHUFFLE,
        )

        return training_config

    def get_evaluation_config(self) -> EvaluationConfig:

        eval_config = EvaluationConfig(
            model_path=Path(self.config.training.trained_model_path),
            testing_data_path=Path(self.config.data_split.test_data_path),
            all_params=self.params,
            mlflow_uri=self.config.evaluation.mlflow_uri,
            params_max_len=self.params.MAX_LEN,
            params_valid_batch_size=self.params.VALID_BATCH_SIZE,
            params_epochs=self.params.EPOCHS,
            params_valid_num_workers=self.params.VALID_NUM_WORKERS,
            params_valid_shuffle=self.params.VALID_SHUFFLE
        )

        return eval_config