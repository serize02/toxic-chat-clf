from bert_toxic_chat_clf.config.configuration import ConfigurationManager
from bert_toxic_chat_clf.components.data_ingestion import DataIngestion
from bert_toxic_chat_clf.components.setup_model import SetupModel
from bert_toxic_chat_clf.components.trainer import Trainer
from bert_toxic_chat_clf import logger

try:
    STAGE_NAME = 'data-ingestion'
    logger.info(f'Stage {STAGE_NAME} started')
    config = ConfigurationManager()
    data_ingestion_config = config.get_data_ingestion_config()
    data_ingestion = DataIngestion(config=data_ingestion_config)
    data_ingestion.download_file()
    logger.info(f'{STAGE_NAME} completed successfully...')

    STAGE_NAME = 'setup-model'
    logger.info(f'Stage {STAGE_NAME} started')
    setup_model_config = config.get_setup_model_config()
    setup_model = SetupModel(setup_model_config)
    setup_model.get_model()
    logger.info(f'{STAGE_NAME} completed successfully...')

    STAGE_NAME = 'training'
    logger.info(f'Stage {STAGE_NAME} started')
    training_config = config.get_training_config()
    trainer = Trainer(config=training_config)
    trainer.setup_device()
    trainer.setup_model()
    trainer.setup_optimizer()
    trainer.split_data()
    trainer.train()
    trainer.run_validation()
    logger.info(f'{STAGE_NAME} completed successfully...')

except Exception as e:
    logger.exception(e)
    raise e

