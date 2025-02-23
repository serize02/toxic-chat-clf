import os
import urllib.request as request
import gdown
from bert_toxic_chat_clf import logger
from bert_toxic_chat_clf.utils.common import get_size
from bert_toxic_chat_clf.entity.config_entity import DataIngestionConfig

class DataIngestion:
    
    def __init__(self, config: DataIngestionConfig):
        self.config = config


    def download_file(self) -> str:
        """fetch data from the url
        """
        local_file = self.config.local_data_file

        try:
            dataset_url = self.config.source_URL
            os.makedirs('artifacts/data_ingestion', exist_ok=True)
            logger.info(f'Downloading data from {dataset_url} into file {local_file}')

            url = 'https://drive.google.com/uc?/export=download&id='+dataset_url.split('/')[-2]
            gdown.download(url, local_file)

            logger.info(f'Downloaded data from {dataset_url} into file {local_file}')

        except Exception as e:
            raise e