from bert_toxic_chat_clf.config.configuration import ConfigurationManager
from bert_toxic_chat_clf.components.data_split import DataSplit
from bert_toxic_chat_clf import logger

STAGE_NAME = 'data-split'

class DataSplitPipeline:

    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_split_config = config.get_data_split_config()
        data_split = DataSplit(config=data_ingestion_config)
        data_split.load_data()
        data_split.split()

if __name__ == '__main__':
    
    try:
        logger.info(f'Stage {STAGE_NAME} started')
        obj = DataSplitPipeline()
        obj.main()
        logger.info(f'{STAGE_NAME} completed successfully...')
    
    except Exception as e:
        logger.exception(e)
        raise e