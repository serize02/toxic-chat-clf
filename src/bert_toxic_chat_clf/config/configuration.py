from bert_toxic_chat_clf.constants import *
from bert_toxic_chat_clf.utils.common import read_yaml, create_directories
from bert_toxic_chat_clf.entity.config_entity import DataIngestionConfig, SetupModelConfig, TrainingConfig

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
            training_data_path=Path(self.config.data_ingestion.local_data_file),
            params_train_size=params.TRAIN_SIZE,
            params_max_len=params.MAX_LEN,
            params_train_batch_size=params.TRAIN_BATCH_SIZE,
            params_valid_batch_size=params.VALID_BATCH_SIZE,
            params_epochs=params.EPOCHS,
            params_learning_rate=params.LEARNING_RATE,
            params_train_num_workers=params.TRAIN_NUM_WORKERS,
            params_valid_num_workers=params.VALID_NUM_WORKERS,
            params_train_shuffle=params.TRAIN_SHUFFLE,
            params_valid_shuffle=params.VALID_SHUFFLE
        )

        return training_config