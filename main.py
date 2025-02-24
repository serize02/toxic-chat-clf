from bert_toxic_chat_clf.config.configuration import ConfigurationManager
from bert_toxic_chat_clf.components.data_ingestion import DataIngestion
from bert_toxic_chat_clf.components.setup_model import SetupModel
from bert_toxic_chat_clf.components.trainer import Trainer
from bert_toxic_chat_clf.components.data_split import DataSplit
from bert_toxic_chat_clf.components.evaluation import Evaluation
from bert_toxic_chat_clf import logger

try:

    config = ConfigurationManager()

    # STAGE_NAME = 'data-ingestion'
    # logger.info(f'Stage {STAGE_NAME} started')
    # data_ingestion_config = config.get_data_ingestion_config()
    # data_ingestion = DataIngestion(config=data_ingestion_config)
    # data_ingestion.download_file()
    # logger.info(f'{STAGE_NAME} completed successfully...')

    # STAGE_NAME = 'split-data'
    # logger.info(f'Stage {STAGE_NAME} started')
    # data_split_config = config.get_data_split_config()
    # data_split = DataSplit(config=data_split_config)
    # data_split.load_data()
    # data_split.split()
    # logger.info(f'{STAGE_NAME} completed successfully...')

    # STAGE_NAME = 'setup-model'
    # logger.info(f'Stage {STAGE_NAME} started')
    # setup_model_config = config.get_setup_model_config()
    # setup_model = SetupModel(setup_model_config)
    # setup_model.get_model()
    # logger.info(f'{STAGE_NAME} completed successfully...')

    # STAGE_NAME = 'training'
    # logger.info(f'Stage {STAGE_NAME} started')
    # training_config = config.get_training_config()
    # trainer = Trainer(config=training_config)
    # trainer.setup_device()
    # trainer.setup_model()
    # trainer.setup_optimizer()
    # trainer.load_data()
    # trainer.train()
    # logger.info(f'{STAGE_NAME} completed successfully...')

    STAGE_NAME='evaluation'
    logger.info(f'Stage {STAGE_NAME} started')
    evaluation_config = config.get_evaluation_config()
    evaluation = Evaluation(evaluation_config)
    evaluation.setup_device()
    evaluation.load_model()
    evaluation.load_data()
    evaluation.run_validation()
    evaluation.log_mlflow()
    logger.info(f'{STAGE_NAME} completed successfully...')

except Exception as e:
    logger.exception(e)
    raise e

