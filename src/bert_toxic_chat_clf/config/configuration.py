from bert_toxic_chat_clf.constants import *
from bert_toxic_chat_clf.utils.common import read_yaml, create_directories
from bert_toxic_chat_clf.entity.config_entity import DataIngestionConfig, SetupModelConfig

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