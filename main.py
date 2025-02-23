from bert_toxic_chat_clf.config.configuration import ConfigurationManager
from bert_toxic_chat_clf.components.data_ingestion import DataIngestion
from bert_toxic_chat_clf.components.setup_model import SetupModel
from bert_toxic_chat_clf import logger

try:
    STAGE_NAME = 'data-ingestion'
    logger.info(f'Stage {STAGE_NAME} started')
    config = ConfigurationManager()
    data_ingestion_config = config.get_data_ingestion_config()
    data_ingestion = DataIngestion(config=data_ingestion_config)
    data_ingestion.download_file()
    logger.info(f'{STAGE_NAME} completed successfully...')

    logger.info(f'Stage {STAGE_NAME} started')
    STAGE_NAME = 'setup-model'
    setup_model_config = config.get_setup_model_config()
    setup_model = SetupModel(setup_model_config)
    setup_model.get_model()
    logger.info(f'{STAGE_NAME} completed successfully...')

except Exception as e:
    logger.exception(e)
    raise e

