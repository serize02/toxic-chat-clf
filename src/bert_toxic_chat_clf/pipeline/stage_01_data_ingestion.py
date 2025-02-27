from bert_toxic_chat_clf.config.configuration import ConfigurationManager
from bert_toxic_chat_clf.components.data_ingestion import DataIngestion
from bert_toxic_chat_clf import logger

STAGE_NAME = 'data-ingestion'

class DataIngestionTrainingPipeline:

    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()


if __name__ == '__main__':
    
    try:
        logger.info(f'Stage {STAGE_NAME} started')
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f'{STAGE_NAME} completed successfully...')
    
    except Exception as e:
        logger.exception(e)
        raise e